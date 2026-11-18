"""Turn C5 run directories and session directories into summaries and comparison tables (B1,
decisions 0013 and 0016). A session directory holds group.json.

uv run python -m bench.analysis bench/results/<run> [...]          # summary.md, latency.svg
uv run python -m bench.analysis <session> <session> --compare      # and print one table across them
"""

import argparse
import hashlib
import itertools
import json
import math
import statistics
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from bench import schema
from bench.paired import Estimate, bootstrap, claimed, floor, ratio
from bench.regimes import ChunkSpan, GpuSample, Regime, classify, read_powermetrics
from bench.schema import (
    Group,
    Probe,
    RequestRecord,
    RunConfig,
    RunEnv,
    Sent,
    SessionInfo,
    SessionRecord,
)

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
        f"temperature {c.temperature}."
        + (f" This target alone also got {c.extra_body}." if c.extra_body else ""),
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
        target = c.target
        if c.smoke:
            target += " (smoke)"
        elif not run.env.powermetrics:
            target += " (not B5: no powermetrics log)"
        failed = sum(s.failed for s in run.stats)
        lines.append(
            f"| {target} | {c.concurrency} | {c.requests} | {c.repeats} "
            f"| {_cells(headline(run.stats))} | {failed} | `{run.path}` |"
        )
    return "\n".join(lines) + "\n"


def write(run: Run) -> None:
    (run.path / schema.SUMMARY).write_text(summary(run))
    chart(run, run.path / schema.CHART)


# --- Sessions (decision 0016) ------------------------------------------------------------

GB = 2**30
STALL_MS = 100.0
METRICS = ("TTFT", "TPOT", "E2E", "Output tok/s")
REGIMES = (Regime.PEAK, Regime.SUSTAINED)


@dataclass(frozen=True)
class Gate:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class Session:
    path: Path
    group: Group
    info: SessionInfo
    env: RunEnv
    records: list[SessionRecord]
    probes: list[Probe]
    prompt_tokens: dict[str, int]
    spans: dict[tuple[int, str], ChunkSpan]
    samples: list[GpuSample]
    regimes: dict[int, Regime]
    gates: list[Gate]

    @property
    def verdict(self) -> str:
        if self.info.smoke:
            return "SMOKE"
        return "VALID" if all(g.passed for g in self.gates) else "DISCARDED"

    @property
    def baseline(self) -> str:
        return self.group.baseline.name

    @property
    def twin(self) -> str:
        return self.group.twin.name

    def measured(self, engine: str, blocks: Sequence[int]) -> list[SessionRecord]:
        wanted = set(blocks)
        return [
            r
            for r in self.records
            if r.phase == "measure" and r.engine == engine and r.block in wanted
        ]


def _sent_tpot(s: Sent) -> float | None:
    if s.usage is None or s.usage.completion_tokens < 2 or len(s.chunk_ms) < 2:
        return None
    return (s.chunk_ms[-1] - s.chunk_ms[0]) / (s.usage.completion_tokens - 1)


def _ok(s: Sent) -> bool:
    return s.error is None and s.usage is not None


def per_request(metric: str, s: Sent) -> float | None:
    if not _ok(s):
        return None
    if metric == "TTFT":
        return s.chunk_ms[0] if s.chunk_ms else None
    if metric == "TPOT":
        return _sent_tpot(s)
    return s.end_ms


def _spans(records: Sequence[SessionRecord]) -> dict[tuple[int, str], ChunkSpan]:
    edges: dict[tuple[int, str], list[float]] = {}
    for r in records:
        if r.phase == "measure":
            end = r.sent.sent_unix + r.sent.end_ms / 1000
            edges.setdefault((r.block, r.engine), []).extend([r.sent.sent_unix, end])
    return {key: ChunkSpan(key[1], min(t), max(t)) for key, t in edges.items()}


def _samples(path: Path) -> list[GpuSample]:
    gz = path / schema.POWERMETRICS_GZ
    try:
        return read_powermetrics(gz) if gz.exists() else []
    except (OSError, EOFError, KeyError, ValueError):
        return []


def _conditions_gate(
    info: SessionInfo, probes: Sequence[Probe], samples: Sequence[GpuSample]
) -> Gate:
    problems = []
    if info.smoke:
        problems.append("a smoke session")
    if not samples:
        problems.append("no clock log parsed")
    off_ac = sum(p.power.on_ac is not True for p in probes)
    if off_ac:
        problems.append(f"{off_ac} probes not on AC power")
    low = sum(p.power.low_power_mode is not False for p in probes)
    if low:
        problems.append(f"{low} probes with Low Power Mode on or unknown")
    detail = "; ".join(problems) or f"{len(samples)} clock samples, {len(probes)} probes on AC"
    return Gate("Conditions", not problems, detail)


def _liveness_gate(
    info: SessionInfo, records: Sequence[SessionRecord], prompt_tokens: Mapping[str, int]
) -> Gate:
    failed = sum(r.sent.error is not None for r in records)
    mismatched = sum(
        r.sent.usage is not None and r.sent.usage.prompt_tokens != prompt_tokens[r.sent.id]
        for r in records
    )
    passed = (
        info.ended == "completed"
        and info.blocks_run == info.blocks_planned
        and not failed
        and not mismatched
    )
    detail = (
        f"{info.blocks_run} of {info.blocks_planned} blocks, ended {info.ended}; "
        f"{failed} of {len(records)} requests failed; {mismatched} prompt lengths differ"
    )
    return Gate("Liveness", passed, detail)


def _memory_gate(group: Group, probes: Sequence[Probe]) -> Gate:
    problems, peaks = [], []
    for spec in group.servers_in_order:
        mine = [p for p in probes if p.engine == spec.name]
        unread = sum(p.peak_bytes is None for p in mine)
        if unread:
            problems.append(f"{spec.name}: {unread} probes without a footprint")
        read = [p.peak_bytes for p in mine if p.peak_bytes is not None]
        if read:
            peak = max(read) / GB
            peaks.append(f"{spec.name} {peak:.1f}/{spec.footprint_gb:g}")
            if peak > spec.footprint_gb:
                problems.append(f"{spec.name} peaked at {peak:.2f} GB, over {spec.footprint_gb:g}")
    if probes:
        first = probes[0].swap_used_bytes
        grown = max(p.swap_used_bytes for p in probes) - first
        if grown > 0:
            problems.append(f"swap grew {grown / 2**20:.0f} MB above the first probe's")
    detail = "; ".join(problems) or "peak GB of limit: " + ", ".join(peaks) + "; swap never grew"
    return Gate("Memory", not problems, detail)


def ratios(session: Session, engine: str, regime: Regime) -> dict[str, Estimate]:
    """``engine`` over the baseline per metric, prompt by prompt in the same block; output
    tokens per second chunk by chunk."""
    blocks = [b for b, r in session.regimes.items() if r is regime]
    base = {(r.block, r.sent.id): r.sent for r in session.measured(session.baseline, blocks)}
    out: dict[str, Estimate] = {}
    for metric in METRICS[:3]:
        per_block: dict[int, list[float]] = {}
        for r in session.measured(engine, blocks):
            mine = per_request(metric, r.sent)
            theirs = base.get((r.block, r.sent.id))
            other = None if theirs is None else per_request(metric, theirs)
            if mine is not None and other:
                per_block.setdefault(r.block, []).append(mine / other)
        if per_block:
            out[metric] = ratio(per_block, session.group.seed)
    rates = {
        b: [_rate(session, engine, b) / _rate(session, session.baseline, b)]
        for b in blocks
        if _rate(session, session.baseline, b) > 0
    }
    if rates:
        out[METRICS[3]] = ratio(rates, session.group.seed)
    return out


def _rate(session: Session, engine: str, block: int) -> float:
    span = session.spans.get((block, engine))
    if span is None or span.end_unix <= span.start_unix:
        return 0.0
    tokens = sum(
        r.sent.usage.completion_tokens
        for r in session.measured(engine, [block])
        if r.sent.usage is not None and r.sent.error is None
    )
    return tokens / (span.end_unix - span.start_unix)


def _twin_gate(group: Group, estimates: Mapping[Regime, Estimate | None]) -> Gate:
    limits = {
        Regime.PEAK: group.gates.twin_tpot_peak,
        Regime.SUSTAINED: group.gates.twin_tpot_sustained,
    }
    problems, seen = [], []
    for regime, e in estimates.items():
        if e is None:
            continue
        gap = abs(e.point - 1)
        seen.append(f"{regime.value} {e.point:.4f} (limit {limits[regime]:g})")
        if gap > limits[regime]:
            problems.append(f"{regime.value} twin TPOT median {e.point:.4f}, off by {gap:.2%}")
    detail = "; ".join(problems) or ("; ".join(seen) or "no Peak or Sustained block")
    return Gate("Twin", not problems, detail)


def load_session(path: Path) -> Session:
    group = Group.model_validate_json((path / schema.GROUP).read_text())
    info = SessionInfo.model_validate_json((path / schema.SESSION).read_text())
    plan = json.loads((path / schema.PLAN).read_text())
    workload_path = Path(plan["workload"])
    if hashlib.sha256(workload_path.read_bytes()).hexdigest() != plan["workload_sha256"]:
        raise SystemExit(f"{workload_path} is not the workload this session used")
    prompt_tokens = {r.id: r.prompt_tokens for r in schema.load_workload(workload_path).requests}
    records = [
        SessionRecord.model_validate_json(line)
        for line in (path / schema.RAW).read_text().splitlines()
        if line
    ]
    probes = [
        Probe.model_validate_json(line)
        for line in (path / schema.PROBES).read_text().splitlines()
        if line
    ]
    spans = _spans(records)
    samples = _samples(path)
    deciders = (group.baseline.name, group.twin.name)
    regimes = {}
    if samples:
        for block in sorted({b for b, _ in spans}):
            block_spans = [s for (b, _), s in spans.items() if b == block]
            regimes[block] = classify(block_spans, samples, deciders)
    gates = [
        _conditions_gate(info, probes, samples),
        _liveness_gate(info, records, prompt_tokens),
        _memory_gate(group, probes),
    ]
    env = RunEnv.model_validate_json((path / schema.ENV).read_text())
    session = Session(
        path, group, info, env, records, probes, prompt_tokens, spans, samples, regimes, gates
    )
    twin = {regime: ratios(session, session.twin, regime).get("TPOT") for regime in REGIMES}
    gates.append(_twin_gate(group, twin))
    return session


def _estimate(e: Estimate | None, decimals: int = 3) -> str:
    if e is None:
        return "n/a"
    interval = (
        "no interval"
        if e.low is None or e.high is None
        else f"[{e.low:.{decimals}f}, {e.high:.{decimals}f}]"
    )
    return f"{e.point:.{decimals}f} {interval}, {e.prompts}/{e.blocks}"


def _clock_mix(session: Session, engine: str, blocks: Sequence[int]) -> str:
    weight: dict[int, float] = {}
    for b in blocks:
        span = session.spans.get((b, engine))
        if span is None:
            continue
        for s in session.samples:
            if s.end_unix > span.start_unix and s.end_unix - s.elapsed_s < span.end_unix:
                for mhz, share in s.steps.items():
                    weight[mhz] = weight.get(mhz, 0.0) + share * s.elapsed_s
    total = sum(weight.values())
    if not total:
        return "n/a"
    top = sorted(weight.items(), key=lambda kv: -kv[1])[:3]
    return ", ".join(f"{mhz} MHz {w / total:.0%}" for mhz, w in top)


def _ratio_table(session: Session, regime: Regime) -> list[str]:
    floors = {m: floor(e) for m, e in ratios(session, session.twin, regime).items()}
    lines = [
        f"Ratio to `{session.baseline}`: median [95% block-bootstrap interval], prompts/blocks. "
        "Below 1 is faster for TTFT, TPOT and E2E; above 1 is faster for output tok/s.",
        "",
        "| Engine | " + " | ".join(METRICS) + " |",
        "|---" * (1 + len(METRICS)) + "|",
    ]
    others = [e.name for e in session.group.engines]
    for engine in [*others, session.twin]:
        est = ratios(session, engine, regime)
        cells = []
        for m in METRICS:
            cell = _estimate(est.get(m))
            if engine != session.twin and m in est and not claimed(est[m], floors.get(m)):
                cell += ", within noise"
            cells.append(cell)
        lines.append(f"| {engine} | " + " | ".join(cells) + " |")
    lines.append(
        "| floor | "
        + " | ".join("n/a" if floors.get(m) is None else f"±{floors[m]:.3f}" for m in METRICS)
        + " |"
    )
    return lines


def _absolute_table(session: Session, regime: Regime) -> list[str]:
    blocks = [b for b, r in session.regimes.items() if r is regime]
    lines = [
        "Absolute, ms: p50, p95, p99 [95% block-bootstrap interval of p99].",
        "",
        "| Engine | Requests | TTFT | TPOT | E2E | Clock mix |",
        "|---|---|---|---|---|---|",
    ]
    for spec in session.group.servers_in_order:
        rows = session.measured(spec.name, blocks)
        cells = []
        for metric in METRICS[:3]:
            per_block: dict[int, list[float]] = {}
            for r in rows:
                v = per_request(metric, r.sent)
                if v is not None:
                    per_block.setdefault(r.block, []).append(v)
            values = [v for vs in per_block.values() for v in vs]
            if not values:
                cells.append("n/a")
                continue
            p99 = bootstrap(per_block, session.group.seed, lambda v: nearest_rank(v, 99))
            interval = "" if p99.low is None else f" [{p99.low:.1f}, {p99.high:.1f}]"
            cells.append(
                f"{nearest_rank(values, 50):.1f}, {nearest_rank(values, 95):.1f}, "
                f"{p99.point:.1f}{interval}"
            )
        mix = _clock_mix(session, spec.name, blocks)
        lines.append(f"| {spec.name} | {len(rows)} | " + " | ".join(cells) + f" | {mix} |")
    return lines


def _streams_table(session: Session, regime: Regime) -> list[str]:
    blocks = [b for b, r in session.regimes.items() if r is regime]
    lines = [
        f"Streams: a stall is a gap over {STALL_MS:g} ms between chunks. The footprint is the "
        "session's peak.",
        "",
        "| Engine | Largest gap (ms) | Stream time in stalls | Peak footprint (GB) |",
        "|---|---|---|---|",
    ]
    for spec in session.group.servers_in_order:
        rows = [r.sent for r in session.measured(spec.name, blocks)]
        gaps = [b - a for s in rows for a, b in itertools.pairwise(s.chunk_ms)]
        stream = sum(s.chunk_ms[-1] - s.chunk_ms[0] for s in rows if s.chunk_ms)
        stalled = sum(g for g in gaps if g > STALL_MS)
        peaks = [p.peak_bytes for p in session.probes if p.engine == spec.name and p.peak_bytes]
        largest = f"{max(gaps):.0f}" if gaps else "n/a"
        share = f"{stalled / stream:.2%}" if stream else "n/a"
        peak = f"{max(peaks) / GB:.2f}" if peaks else "n/a"
        lines.append(f"| {spec.name} | {largest} | {share} | {peak} |")
    return lines


def _same_length(session: Session) -> list[str]:
    base = {
        (r.block, r.sent.id): r.sent.usage.completion_tokens
        for r in session.records
        if r.phase == "measure" and r.engine == session.baseline and r.sent.usage is not None
    }
    lines = []
    for spec in session.group.servers_in_order[1:]:
        pairs = [
            (r.sent.usage.completion_tokens, base[r.block, r.sent.id])
            for r in session.records
            if r.phase == "measure"
            and r.engine == spec.name
            and r.sent.usage is not None
            and (r.block, r.sent.id) in base
        ]
        same = sum(a == b for a, b in pairs)
        lines.append(f"- {spec.name}: {same} of {len(pairs)} prompts")
    return lines


def _conditions(session: Session) -> str:
    e, n = session.env, len(session.probes)
    libraries = ", ".join(f"{k} {v}" for k, v in e.libraries.items() if k in ("mlx", "mlx-lm"))
    charging = sum(p.power.charging is True for p in session.probes)
    lid = sum(p.power.lid_open is True for p in session.probes)
    return (
        f"- Commit `{e.commit[:12]}`{' (dirty tree)' if e.dirty else ''}. {e.chip}, "
        f"{e.cores} cores, {e.memory_gb:g} GB, macOS {e.macos}; {libraries}. At the start: "
        f"AC {e.power.on_ac}, battery {e.power.battery_percent}%, Low Power Mode "
        f"{e.power.low_power_mode}. Over {n} probes: charging in {charging}, lid open in {lid}."
    )


def _lifecycle(group: Group) -> str:
    return "servers stay loaded" if group.servers == "resident" else "one server per chunk"


def session_summary(session: Session) -> str:
    g, info = session.group, session.info
    counts = {r: sum(v is r for v in session.regimes.values()) for r in Regime}
    lines = [
        f"# {session.verdict}: {g.name}, concurrency {g.concurrency}",
        "",
        f"- Baseline `{g.baseline.name}` and its twin; engines "
        + ", ".join(f"`{e.name}`" for e in g.engines)
        + f". {_lifecycle(g).capitalize()}.",
        f"- {g.prompts_per_chunk} prompts and {g.warmup_per_chunk} warm-up requests a chunk; "
        f"settle {info.settle_s:g} s; seed {g.seed}.",
        _conditions(session),
        "",
        "## Gates",
        "",
        *(
            f"- **{gate.name}**: {'passed' if gate.passed else 'FAILED'}. {gate.detail}."
            for gate in session.gates
        ),
        "",
        "## Regimes",
        "",
        ", ".join(f"{r.value} {counts[r]}" for r in Regime) + " blocks. Transition blocks are in "
        "raw.jsonl and in no table.",
        "",
    ]
    for regime in REGIMES:
        if not counts[regime]:
            continue
        lines += [f"## {regime.value}", ""]
        lines += [*_ratio_table(session, regime), "", *_absolute_table(session, regime), ""]
        lines += [*_streams_table(session, regime), ""]
    if g.concurrency == 1:
        lines += [
            f"## Same completion length as `{g.baseline.name}`, prompt by prompt",
            "",
            *_same_length(session),
            "",
        ]
    return "\n".join(lines)


def write_session(session: Session) -> None:
    (session.path / schema.SUMMARY).write_text(session_summary(session))


def compare_sessions(sessions: Sequence[Session]) -> str:
    lines: list[str] = []
    for regime in REGIMES:
        for session in sessions:
            if not any(r is regime for r in session.regimes.values()):
                continue
            g = session.group
            lines += [
                f"### {regime.value}: {g.name}, concurrency {g.concurrency}, "
                f"{_lifecycle(g)}, {session.verdict}",
                "",
                *_ratio_table(session, regime),
                "",
            ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.analysis")
    parser.add_argument("dirs", nargs="+", type=Path, help="C5 run or session directories")
    parser.add_argument("--compare", action="store_true", help="also print a table across them")
    args = parser.parse_args(argv)
    runs = [load_run(p) for p in args.dirs if not (p / schema.GROUP).exists()]
    sessions = [load_session(p) for p in args.dirs if (p / schema.GROUP).exists()]
    for run in runs:
        write(run)
        print(f"wrote {run.path / schema.SUMMARY}", file=sys.stderr)
    for session in sessions:
        write_session(session)
        print(f"wrote {session.path / schema.SUMMARY}", file=sys.stderr)
    if args.compare:
        if runs:
            print(compare(runs), end="")
        if sessions:
            print(compare_sessions(sessions), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
