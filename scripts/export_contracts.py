"""Write the telemetry JSON Schema (C2) and the load control schema (decision 0025), or check
that the committed ones are fresh.

    uv run python -m scripts.export_contracts          # write them
    uv run python -m scripts.export_contracts --check  # exit 1 if one is stale

The TypeScript types are generated from these files by ``pnpm -C terminal gen:contracts``.
``make contracts`` runs both steps.
"""

import argparse
import sys

from bench import control

from ullm.telemetry import export

SCHEMAS = ((export.SCHEMA_PATH, export.render_schema), (control.SCHEMA_PATH, control.render_schema))


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the contract JSON Schemas.")
    parser.add_argument("--check", action="store_true", help="exit 1 if a file is stale")
    args = parser.parse_args()

    stale = 0
    for path, render in SCHEMAS:
        fresh = render()
        if not args.check:
            path.write_text(fresh)
            print(f"wrote {path}")
        elif (path.read_text() if path.exists() else "") != fresh:
            print(f"{path} is stale: run `make contracts`", file=sys.stderr)
            stale = 1
    return stale


if __name__ == "__main__":
    sys.exit(main())
