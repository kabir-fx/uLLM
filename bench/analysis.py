"""Turn C5 run directories into summaries, charts and a comparison table (B1, decision 0013).

uv run python -m bench.analysis bench/results/<run> [...]          # summary.md, latency.svg
uv run python -m bench.analysis <run> <run> --compare              # and print one table across runs
"""

import argparse
import hashlib
import math
import statistics
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from bench import schema
from bench.schema import RequestRecord, RunConfig, RunEnv

# The first three categorical slots of the dataviz reference palette, on its light surface.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
SURFACE = "#fcfcfb"
INK = "#3d3d3a"


def nearest_rank(values: Sequence[float], q: float) -> float:
    """The smallest sample with at least q percent of the samples at or below it."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]


@dataclass(frozen=True)
class Percentiles:
    p50: float
    p99: float


def _percentiles(values: Sequence[float]) -> Percentiles | None:
    return Percentiles(nearest_rank(values, 50), nearest_rank(values, 99)) if values else None


def _tpot(r: RequestRecord) -> float | None:
    if r.usage is None or r.usage.completion_tokens < 2 or len(r.chunk_ms) < 2:
        return None
    return (r.chunk_ms[-1] - r.chunk_ms[0]) / (r.usage.completion_tokens - 1)


@dataclass(frozen=True)
class RepeatStats:
    repeat: int
    requests: int
    failed: int
    prompt_mismatches: int
    ttft_ms: Percentiles | None
    tpot_ms: Percentiles | None
    e2e_ms: Percentiles | None
    output_tokens: int
    wall_s: float
    tokens_per_s: float


def repeat_stats(
    records: Sequence[RequestRecord], prompt_tokens: Mapping[str, int]
) -> list[RepeatStats]:
    """Figures for each measured repeat. Failed requests count only as failures."""
    out = []
    for repeat in sorted({r.repeat for r in records if r.phase == "measure"}):
        rows = [r for r in records if r.phase == "measure" and r.repeat == repeat]
        ok = [r for r in rows if r.error is None and r.usage is not None]
        tpots = [t for t in map(_tpot, ok) if t is not None]
        tokens = sum(r.usage.completion_tokens for r in ok if r.usage is not None)
        wall = max(r.sent_s + r.end_ms / 1000 for r in rows) - min(r.sent_s for r in rows)
        out.append(
            RepeatStats(
                repeat=repeat,
                requests=len(rows),
                failed=len(rows) - len(ok),
                prompt_mismatches=sum(
                    r.usage is not None and r.usage.prompt_tokens != prompt_tokens[r.id] for r in ok
                ),
                ttft_ms=_percentiles([r.chunk_ms[0] for r in ok if r.chunk_ms]),
                tpot_ms=_percentiles(tpots),
                e2e_ms=_percentiles([r.end_ms for r in ok]),
                output_tokens=tokens,
                wall_s=wall,
                tokens_per_s=tokens / wall if wall > 0 else 0.0,
            )
        )
    return out


@dataclass(frozen=True)
class Run:
    path: Path
    config: RunConfig
    env: RunEnv
    records: list[RequestRecord]
    stats: list[RepeatStats]


def load_run(path: Path) -> Run:
    config = RunConfig.model_validate_json((path / schema.CONFIG).read_text())
    env = RunEnv.model_validate_json((path / schema.ENV).read_text())
    lines = (path / schema.RAW).read_text().splitlines()
    records = [RequestRecord.model_validate_json(line) for line in lines if line]
    workload_path = Path(config.workload)
    if hashlib.sha256(workload_path.read_bytes()).hexdigest() != config.workload_sha256:
        raise SystemExit(f"{workload_path} is not the workload this run used: its SHA-256 differs")
    workload = schema.load_workload(workload_path)
    prompt_tokens = {r.id: r.prompt_tokens for r in workload.requests}
    return Run(path, config, env, records, repeat_stats(records, prompt_tokens))


FIGURES = ("Output tok/s", "TTFT p50", "TTFT p99", "TPOT p50", "TPOT p99", "E2E p50", "E2E p99")


def _pair(p: Percentiles | None) -> tuple[float | None, float | None]:
    return (None, None) if p is None else (p.p50, p.p99)


def figures(s: RepeatStats) -> tuple[float | None, ...]:
    """One repeat's figures, in the order of ``FIGURES``."""
    return (s.tokens_per_s, *_pair(s.ttft_ms), *_pair(s.tpot_ms), *_pair(s.e2e_ms))


def headline(stats: Sequence[RepeatStats]) -> tuple[float | None, ...]:
    """The median over repeats of each figure (B5)."""
    columns = zip(*map(figures, stats), strict=True) if stats else [()] * len(FIGURES)
    return tuple(_median(column) for column in columns)


def _median(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return statistics.median(present) if present else None


def _cells(values: Sequence[float | None]) -> str:
    return " | ".join("n/a" if v is None else f"{v:.1f}" for v in values)


def summary(run: Run) -> str:
    c, e, power = run.config, run.env, run.env.power
    metrics = "snapshot taken" if (run.path / schema.METRICS).exists() else "not served"
    lines = [f"# {c.target}, concurrency {c.concurrency}", ""]
    if c.smoke:
        lines += ["**smoke: not a B5 run.** Use it for before and after numbers only.", ""]
    lines += [
        f"- Model: `{c.served_model}`. Workload: `{c.workload}`, {c.requests} requests, "
        f"`max_tokens` from the workload, `chat_template_kwargs` {c.chat_template_kwargs}, "
        f"temperature {c.temperature}.",
        f"- Warm-up: {c.warmup} requests, discarded. "
        f"Repeats: {c.repeats}, cooldown {c.cooldown_s:g} s.",
        f"- Commit `{e.commit[:12]}`{' (dirty tree)' if e.dirty else ''}. "
        f"{e.chip}, {e.cores} cores, {e.memory_gb:g} GB, macOS {e.macos}.",
        f"- Power: AC {power.on_ac}, battery {power.battery_percent}%, "
        f"Low Power Mode {power.low_power_mode}. "
        f"`powermetrics`: {'logged' if e.powermetrics else 'not logged'}. `/metrics`: {metrics}.",
        "",
        "| Repeat | Requests | Failed | Prompt mismatches | " + " | ".join(FIGURES) + " |",
        "|---" * (4 + len(FIGURES)) + "|",
    ]
    for s in run.stats:
        counts = f"| {s.repeat} | {s.requests} | {s.failed} | {s.prompt_mismatches}"
        lines.append(f"{counts} | {_cells(figures(s))} |")
    lines += [
        f"| **median** | | | | {_cells(headline(run.stats))} |",
        "",
        "Times in ms. TPOT = (last chunk - first chunk) / (completion tokens - 1). "
        "Percentiles by nearest rank over one repeat's successful requests.",
        "",
        f"![TTFT and TPOT distributions]({schema.CHART})",
        "",
    ]
    return "\n".join(lines)


def _ttft(r: RequestRecord) -> float | None:
    return r.chunk_ms[0] if r.chunk_ms else None


def chart(run: Run, path: Path) -> None:
    import matplotlib

    matplotlib.use("svg")
    import matplotlib.pyplot as plt

    measured = [r for r in run.records if r.phase == "measure" and r.error is None]
    repeats = sorted({r.repeat for r in measured})
    panels: tuple[tuple[str, Callable[[RequestRecord], float | None]], ...] = (
        ("TTFT (ms)", _ttft),
        ("TPOT (ms)", _tpot),
    )
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), facecolor=SURFACE)
    for ax, (label, get) in zip(axes, panels, strict=True):
        ax.set_facecolor(SURFACE)
        for i, repeat in enumerate(repeats):
            got = (get(r) for r in measured if r.repeat == repeat)
            values = sorted(v for v in got if v is not None)
            share = [(k + 1) / len(values) for k in range(len(values))]
            color = SERIES[i % len(SERIES)]
            ax.step(values, share, where="post", linewidth=2, color=color, label=f"repeat {repeat}")
        ax.set_xlabel(label, color=INK)
        ax.set_ylabel("share of requests at or below", color=INK)
        ax.set_ylim(0, 1.02)
        ax.grid(color="#e4e3df", linewidth=0.8)
        ax.tick_params(colors=INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if len(repeats) > 1:
            ax.legend(frameon=False, labelcolor=INK)
    fig.suptitle(f"{run.config.target}, concurrency {run.config.concurrency}", color=INK)
    fig.tight_layout()
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)


def compare(runs: Sequence[Run]) -> str:
    lines = [
        "| Target | Concurrency | Requests | Repeats | "
        + " | ".join(FIGURES)
        + " | Failed | Run |",
        "|---" * (6 + len(FIGURES)) + "|",
    ]
    for run in runs:
        c = run.config
        target = f"{c.target} (smoke)" if c.smoke else c.target
        failed = sum(s.failed for s in run.stats)
        lines.append(
            f"| {target} | {c.concurrency} | {c.requests} | {c.repeats} "
            f"| {_cells(headline(run.stats))} | {failed} | `{run.path}` |"
        )
    return "\n".join(lines) + "\n"


def write(run: Run) -> None:
    (run.path / schema.SUMMARY).write_text(summary(run))
    chart(run, run.path / schema.CHART)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.analysis")
    parser.add_argument("runs", nargs="+", type=Path, help="C5 run directories")
    parser.add_argument("--compare", action="store_true", help="also print a table across runs")
    args = parser.parse_args(argv)
    runs = [load_run(p) for p in args.runs]
    for run in runs:
        write(run)
        print(f"wrote {run.path / schema.SUMMARY}")
    if args.compare:
        print(compare(runs), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
