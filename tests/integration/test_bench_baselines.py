"""bench/baselines.sh runs decision 0014's six runs, as ``make bench-baselines`` calls it: no
``ARGS``. A stand-in ``uv`` on ``PATH`` records each command instead of running it."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

FAKE_UV = """#!/bin/sh
case "$*" in
  *fetch_models.py*) echo "qwen3-0.6b-bf16: $MODEL_DIR" ;;
  *bench.analysis*) echo "| row |" ;;
  *) printf '%s\\037' "$@" >>"$CALLS"; echo >>"$CALLS" ;;
esac
"""


@pytest.mark.parametrize("args", [None, "--smoke"])
def test_baselines_runs_the_six_runs_with_or_without_args(tmp_path: Path, args: str | None) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "uv").write_text(FAKE_UV)
    (bin_dir / "uv").chmod(0o755)
    model = tmp_path / "snapshot"
    model.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    calls = tmp_path / "calls"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "MODEL_DIR": str(model),
        "CALLS": str(calls),
        "OUT": str(out),
        "GAP": "0",
    }
    env.pop("ARGS", None)
    if args is not None:
        env["ARGS"] = args

    done = subprocess.run(
        ["bash", "bench/baselines.sh"], cwd=ROOT, env=env, capture_output=True, text=True
    )

    assert done.returncode == 0, done.stderr
    runs = [line.split("\x1f")[:-1] for line in calls.read_text().splitlines()]
    targets = [(r[r.index("--target") + 1], r[r.index("--concurrency") + 1]) for r in runs]
    assert targets == [
        ("ullm", "1"),
        ("mlx-lm-sequential-no-cache", "1"),
        ("mlx-lm-sequential", "1"),
        ("ullm", "8"),
        ("mlx-lm-batched", "8"),
        ("mlx-lm-batched-no-cache", "8"),
    ]
    assert all(("--smoke" in r) == (args is not None) for r in runs)
    assert [("--extra-body" in r) for r in runs] == [False, True, True, False, False, False]
    assert all(r[r.index("--model") + 1] == str(model) for r in runs if "--model" in r)
    assert (out / "comparison.md").read_text() == "| row |\n"
