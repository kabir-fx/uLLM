"""The README's benchmark table, from the committed sweep sessions (P1, decision 0030).

uv run python -m bench.readme        # rewrite the table between the markers in README.md
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from bench.analysis import Session, load_session, ratios
from bench.paired import Estimate, claimed, floor
from bench.regimes import Regime

SESSIONS = tuple(Path("bench/results/m2/sweep") / c for c in ("c1", "c2", "c4", "c8"))
BEGIN = "<!-- bench.readme: generated from bench/results/m2/sweep by `make readme` -->"
END = "<!-- bench.readme: end -->"
ENGINE = "ullm"
ROWS = ("Output tok/s", "TTFT", "TPOT", "E2E")


def table(sessions: Sequence[Session]) -> str:
    ordered = sorted(sessions, key=lambda s: s.group.concurrency)
    if bad := [f"c{s.group.concurrency} {s.verdict}" for s in ordered if s.verdict != "VALID"]:
        raise SystemExit(f"the README reports VALID sessions only: {', '.join(bad)}")
    baselines = {s.baseline for s in ordered}
    if len(baselines) != 1:
        raise SystemExit(f"one baseline for the whole table, got {sorted(baselines)}")
    setups = {
        (
            s.env.commit,
            s.env.dirty,
            s.env.chip,
            s.env.memory_gb,
            s.env.macos,
            s.env.libraries["mlx"],
            s.env.libraries["mlx-lm"],
            s.info.served_models[ENGINE],
            s.group.baseline.launch,
            tuple(e.launch for e in s.group.engines if e.name == ENGINE),
        )
        for s in ordered
    }
    if len(setups) != 1:
        raise SystemExit(f"one setup (machine, commit, libraries, launches) per table: {setups}")
    env, model = ordered[0].env, ordered[0].info.served_models[ENGINE]
    commit = f"`{env.commit[:7]}`" + (", dirty tree" if env.dirty else "")
    estimates = [ratios(s, ENGINE, Regime.SUSTAINED) for s in ordered]
    noise = [ratios(s, s.twin, Regime.SUSTAINED) for s in ordered]
    lines = [
        f"| uLLM ÷ `{ordered[0].baseline}`, Sustained | "
        + " | ".join(f"c{s.group.concurrency}" for s in ordered)
        + " |",
        "|---" * (1 + len(ordered)) + "|",
    ]
    for m in ROWS:
        cells = []
        for est, twin in zip(estimates, noise, strict=True):
            e = est[m]
            cell = _cell(e)
            if not claimed(e, floor(twin[m]) if m in twin else None):
                cell += ", within noise"
            cells.append(cell)
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines.append(
        "| Prompts / blocks | "
        + " | ".join(f"{e['TTFT'].prompts} / {e['TTFT'].blocks}" for e in estimates)
        + " |"
    )
    lines += [
        "",
        f"Median [95% block-bootstrap interval]; below 1 is faster for TTFT, TPOT and E2E, "
        f"above 1 for output tok/s. `{model}`, workload A, closed loop. "
        f"{env.chip}, {env.memory_gb:g} GB, macOS {env.macos}; mlx {env.libraries['mlx']}, "
        f"mlx-lm {env.libraries['mlx-lm']}; commit {commit}. "
        "Every session VALID under the run conditions (decision 0016).",
    ]
    return "\n".join(lines) + "\n"


def _cell(e: Estimate) -> str:
    if e.low is None or e.high is None:
        return f"{e.point:.3f}, no interval"
    return f"{e.point:.3f} [{e.low:.3f}, {e.high:.3f}]"


def splice(readme: str, block: str) -> str:
    if (
        readme.count(BEGIN) != 1
        or readme.count(END) != 1
        or readme.index(BEGIN) > readme.index(END)
    ):
        raise SystemExit("README.md needs one pair of bench.readme markers, BEGIN before END")
    head, rest = readme.split(BEGIN)
    _, tail = rest.split(END)
    return f"{head}{BEGIN}\n{block}{END}{tail}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.readme")
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    args = parser.parse_args(argv)
    block = table([load_session(p) for p in SESSIONS])
    args.readme.write_text(splice(args.readme.read_text(), block))
    print(f"wrote the benchmark table in {args.readme}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
