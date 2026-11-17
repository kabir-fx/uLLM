"""Trace files: telemetry frames as gzip-compressed JSONL (C2, decision 0003).

One frame per line, exactly as sent on ``/telemetry/stream``, so the terminal's
live and replay sources read the same thing. Files are written with a zero
gzip timestamp so the same frames always give the same file on one machine.
Compressed bytes can still differ between zlib builds, so compare traces by
their decompressed text, never by file bytes.
"""

import gzip
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

from ullm.telemetry.events import Event, Frame

FRAME_MS = 1000 / 15  # frames are flushed about 15 times a second (O1)


def dumps(frames: Iterable[Frame]) -> str:
    return "".join(frame.model_dump_json() + "\n" for frame in frames)


def write(path: Path, frames: Iterable[Frame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz:
        gz.write(dumps(frames).encode())


def read_text(path: Path) -> str:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return f.read()


def iter_frames(path: Path) -> Iterator[Frame]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield Frame.model_validate_json(line)


def frames_from(events: Sequence[Event], replica: str) -> list[Frame]:
    """Events in time order, grouped into one frame per window they fall in."""
    frames: list[Frame] = []
    for event in events:
        end = (event.t // FRAME_MS + 1) * FRAME_MS
        if frames and frames[-1].t == end:
            frames[-1].events.append(event)
        else:
            frames.append(Frame(replica=replica, seq=len(frames), t=end, events=[event]))
    return frames
