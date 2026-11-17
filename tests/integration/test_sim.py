"""``ullm sim``: the real engine on FakeRunner with many clients, checked by the trace checker."""

import subprocess
import sys
from pathlib import Path

from ullm.cli import main
from ullm.telemetry import events as ev
from ullm.telemetry.check import check_trace
from ullm.telemetry.trace import iter_frames


def test_a_short_sim_exits_zero_and_writes_a_trace_that_passes_the_checker(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sim.jsonl.gz"
    args = ["sim", "--clients", "12", "--seed", "3", "--disconnect", "0.5", "--trace", str(path)]
    assert main(args) == 0
    frames = list(iter_frames(path))
    assert check_trace(frames).problems == []
    hello = frames[0].events[0]
    assert isinstance(hello, ev.Hello)
    assert hello.source == "sim"
    events = [e for f in frames for e in f.events]
    assert len([e for e in events if isinstance(e, ev.ReqArrived)]) == 12
    assert [e for e in events if isinstance(e, ev.ReqAborted)], "the sim aborts some requests"


def test_the_ullm_command_runs_the_sim() -> None:
    done = subprocess.run(
        [sys.executable, "-m", "ullm.cli", "sim", "--clients", "4"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "trace check: ok" in done.stdout
