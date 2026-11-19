"""O1: the recorder writes frames as they come, in the trace format (decision 0023)."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from ullm.telemetry import events as ev
from ullm.telemetry import trace

FRAMES = [
    ev.Frame(
        replica="r1",
        seq=i,
        t=float(i),
        events=[ev.ReqArrived(t=float(i), rid=f"r{i}", prompt_tokens=1, max_tokens=1)],
    )
    for i in range(3)
]


async def _frames() -> AsyncIterator[ev.Frame]:
    for frame in FRAMES:
        yield frame


def test_a_recording_reads_back_as_the_frames_and_matches_a_trace_written_at_once(
    tmp_path: Path,
) -> None:
    recorded, written = tmp_path / "recorded.jsonl.gz", tmp_path / "written.jsonl.gz"
    asyncio.run(trace.record(_frames(), recorded.open("wb")))
    trace.write(written, FRAMES)
    assert list(trace.iter_frames(recorded)) == FRAMES
    assert trace.read_text(recorded) == trace.read_text(written)
