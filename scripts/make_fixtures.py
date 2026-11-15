"""Write the synthetic telemetry traces in terminal/fixtures/telemetry/, or check them.

    uv run python scripts/make_fixtures.py          # write every scenario
    uv run python scripts/make_fixtures.py --check  # exit 1 if a committed trace is stale

Traces are compared by decompressed text: gzip bytes can differ between zlib builds.
"""

import argparse
import sys
from pathlib import Path

from ullm.telemetry import trace
from ullm.telemetry.check import check_trace
from ullm.telemetry.synthetic import SCENARIOS, generate

OUT_DIR = Path(__file__).resolve().parents[1] / "terminal" / "fixtures" / "telemetry"


def main() -> int:
    parser = argparse.ArgumentParser(description="Write or check the synthetic telemetry traces.")
    parser.add_argument("--check", action="store_true", help="exit 1 if a trace is stale")
    args = parser.parse_args()

    stale = []
    for name, scenario in SCENARIOS.items():
        frames = generate(scenario)
        report = check_trace(frames)
        if not report.ok:
            print(f"{name}: generator produced an inconsistent trace:", file=sys.stderr)
            for problem in report.problems[:20]:
                print(f"  {problem}", file=sys.stderr)
            return 1
        path = OUT_DIR / f"{name}.jsonl.gz"
        if args.check:
            if not path.exists() or trace.read_text(path) != trace.dumps(frames):
                stale.append(path)
            continue
        trace.write(path, frames)
        print(f"wrote {path} ({len(frames)} frames, {report.requests} requests)")
    if stale:
        for path in stale:
            print(f"{path} is stale: run `make fixtures`", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
