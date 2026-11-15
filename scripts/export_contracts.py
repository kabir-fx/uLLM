"""Write the telemetry JSON Schema (C2), or check that the committed one is fresh.

    uv run python scripts/export_contracts.py          # write it
    uv run python scripts/export_contracts.py --check  # exit 1 if it is stale

The TypeScript types are generated from this file by ``pnpm -C terminal gen:contracts``.
``make contracts`` runs both steps.
"""

import argparse
import sys

from ullm.telemetry.export import SCHEMA_PATH, render_schema


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the telemetry JSON Schema.")
    parser.add_argument("--check", action="store_true", help="exit 1 if the file is stale")
    args = parser.parse_args()

    fresh = render_schema()
    if args.check:
        current = SCHEMA_PATH.read_text() if SCHEMA_PATH.exists() else ""
        if current != fresh:
            print(f"{SCHEMA_PATH} is stale: run `make contracts`", file=sys.stderr)
            return 1
        return 0
    SCHEMA_PATH.write_text(fresh)
    print(f"wrote {SCHEMA_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
