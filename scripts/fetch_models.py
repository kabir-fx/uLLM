"""Download pinned models into the Hugging Face cache (F4).

    uv run python scripts/fetch_models.py                    # the default model
    uv run python scripts/fetch_models.py qwen3-0.6b-4bit    # one or more by key
    uv run python scripts/fetch_models.py --all
    uv run python scripts/fetch_models.py --list

Each model is fetched at the exact revision in ``ullm.models``. Prints the
local path of each snapshot.
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from ullm.models import DEFAULT_MODEL, MODELS, ModelSpec

# Only what loading needs; repos sometimes carry extra files.
ALLOW = ["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "*.tiktoken"]


def local_path(spec: ModelSpec) -> Path | None:
    """The downloaded snapshot of a model, or None if it is not fully downloaded."""
    try:
        return Path(
            snapshot_download(
                spec.repo, revision=spec.revision, allow_patterns=ALLOW, local_files_only=True
            )
        )
    except LocalEntryNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Download pinned models.")
    parser.add_argument("keys", nargs="*", help=f"model keys (default: {DEFAULT_MODEL.key})")
    parser.add_argument("--all", action="store_true", help="every model in ullm.models")
    parser.add_argument("--list", action="store_true", help="list the models and exit")
    args = parser.parse_args()

    if args.list:
        for spec in MODELS.values():
            have = "downloaded" if local_path(spec) else "-"
            print(f"{spec.key:18} {spec.approx_gb:5.2f} GB  {have:10}  {spec.repo}  {spec.role}")
        return 0
    keys = list(MODELS) if args.all else (args.keys or [DEFAULT_MODEL.key])
    unknown = [k for k in keys if k not in MODELS]
    if unknown:
        print(f"unknown model keys: {unknown}; see --list", file=sys.stderr)
        return 2
    for key in keys:
        spec = MODELS[key]
        path = snapshot_download(spec.repo, revision=spec.revision, allow_patterns=ALLOW)
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
