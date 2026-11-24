"""What uLLM's engine did during each measured chunk of a session, from its per-chunk
recordings: the figures that explain a gap to the baseline (decision 0026).

    uv run python -m bench.steps bench/results/m2/sweep/c8      # writes steps.md
"""

import argparse
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from bench.analysis import REGIMES, Session, load_session, nearest_rank
from bench.regimes import ChunkSpan

from ullm.telemetry import events as ev
from ullm.telemetry.trace import iter_frames

STEPS = "steps.md"


@dataclass(frozen=True)
class Chunk:
    """One chunk's steps and requests inside the client's measured span, in ms since the
    recording's epoch. A step counts when it starts and ends inside the span."""

    span_ms: float
    steps: list[ev.Step]
    lives: dict[str, tuple[float, float | None, float | None]]  # arrived, first token, finished
    dropped: int


def chunk(frames: Iterable[ev.Frame], span: ChunkSpan) -> Chunk:
    frames = list(frames)
    events = [e for f in frames for e in f.events]
    hello = next((e for e in events if isinstance(e, ev.Hello)), None)
    if hello is None:
        raise ValueError("a recording with no hello event: its server died before it began")
    start = span.start_unix * 1000 - hello.epoch_unix_ms
    end = span.end_unix * 1000 - hello.epoch_unix_ms
    steps = [e for e in events if isinstance(e, ev.Step) and e.t - e.ms >= start and e.t <= end]
    arrived = {e.rid: e.t for e in events if isinstance(e, ev.ReqArrived) and start <= e.t <= end}
    first = {e.rid: e.t for e in events if isinstance(e, ev.ReqFirstToken)}
    done = {e.rid: e.t for e in events if isinstance(e, ev.ReqFinished | ev.ReqAborted)}
    lives = {rid: (t, first.get(rid), done.get(rid)) for rid, t in arrived.items()}
    return Chunk(end - start, steps, lives, sum(f.dropped for f in frames))


@dataclass
class Split:
    decode_ms_by_rows: dict[int, list[float]] = field(default_factory=dict)
    prefill_ms_per_1k: list[float] = field(default_factory=list)
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    span_ms: float = 0.0
    gaps_ms: list[float] = field(default_factory=list)
    stall_shares: list[float] = field(default_factory=list)
    requests: int = 0
    dropped: int = 0


def split(chunks: Sequence[Chunk]) -> Split:
    out = Split()
    for c in chunks:
        out.span_ms += c.span_ms
        out.dropped += c.dropped
        out.requests += len(c.lives)
        for s in c.steps:
            if s.kind == "decode":
                out.decode_ms += s.ms
                out.decode_ms_by_rows.setdefault(len(s.rids), []).append(s.ms)
            else:
                out.prefill_ms += s.ms
                out.prefill_ms_per_1k.append(s.ms / s.tokens * 1000)
        for before, after in zip(c.steps, c.steps[1:], strict=False):
            # Host work between steps, only while some request is still live: otherwise the
            # engine waits on the client, which is not its cost.
            if any(a <= before.t and (f is None or f > before.t) for a, _, f in c.lives.values()):
                out.gaps_ms.append(after.t - after.ms - before.t)
        prefills = [s for s in c.steps if s.kind == "prefill"]
        for first, finished in ((f, d) for _, f, d in c.lives.values()):
            if first is None or finished is None or finished <= first:
                continue
            stalled = sum(p.ms for p in prefills if p.t - p.ms >= first and p.t <= finished)
            out.stall_shares.append(stalled / (finished - first))
    return out


def _ps(values: Sequence[float], digits: int = 1) -> str:
    if not values:
        return "n/a"
    return f"{nearest_rank(values, 50):.{digits}f}, {nearest_rank(values, 90):.{digits}f}"


def _table(s: Split) -> list[str]:
    span = s.span_ms or 1.0
    rows = [n for n, ms in s.decode_ms_by_rows.items() for _ in ms]
    mean_rows = f"{sum(rows) / len(rows):.2f}" if rows else "n/a"
    lines = [
        "| Figure | Value |",
        "|---|---|",
        f"| Requests | {s.requests} |",
        f"| Measured span in prefill, decode, outside steps | {s.prefill_ms / span:.1%}, "
        f"{s.decode_ms / span:.1%}, {1 - (s.prefill_ms + s.decode_ms) / span:.1%} |",
        f"| Prefill, ms per 1,000 prompt tokens, p50, p90 | {_ps(s.prefill_ms_per_1k)} |",
        f"| Gap between steps with a request live, ms, p50, p90 | {_ps(s.gaps_ms, 2)} |",
        f"| Others' prefill as a share of a request's decode span, p50, p90 | "
        f"{_ps(s.stall_shares, 3)} |",
        f"| Rows in a decode step, mean over steps | {mean_rows} |",
        "",
        "| Rows | Decode steps | Step ms, p50, p90 |",
        "|---|---|---|",
    ]
    for n in sorted(s.decode_ms_by_rows):
        ms = s.decode_ms_by_rows[n]
        lines.append(f"| {n} | {len(ms)} | {_ps(ms, 2)} |")
    return lines


def report(session: Session, engine: str) -> str:
    g = session.group
    lines = [
        f"# Steps: {g.name}, concurrency {g.concurrency}, {session.verdict}",
        "",
        f"`{engine}` from `{engine}-<block>.jsonl.gz`. A step counts when it starts and ends "
        "inside the client's measured span of its chunk, so warm-up requests never count. "
        "Step time includes packing the batch KV, the forward pass and sampling.",
        "",
    ]
    for regime in REGIMES:
        blocks = sorted(b for b, r in session.regimes.items() if r is regime)
        chunks = [
            chunk(iter_frames(session.path / f"{engine}-{b}.jsonl.gz"), session.spans[b, engine])
            for b in blocks
            if (b, engine) in session.spans
        ]
        if not chunks:
            continue
        s = split(chunks)
        lines += [f"## {regime.value}: {len(chunks)} blocks, {s.dropped} events dropped", ""]
        lines += [*_table(s), ""]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.steps")
    parser.add_argument("dirs", nargs="+", type=Path, help="session directories")
    parser.add_argument("--engine", default="ullm", help="the engine whose recordings to read")
    args = parser.parse_args(argv)
    for path in args.dirs:
        (path / STEPS).write_text(report(load_session(path), args.engine))
        print(f"wrote {path / STEPS}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
