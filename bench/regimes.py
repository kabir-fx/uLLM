"""Sort a session's blocks into Peak, Sustained and Transition from the powermetrics log
(decision 0016). The rule is fixed before any session runs: a sample is capped when the GPU
got a lower state than it asked for."""

import calendar
import gzip
import plistlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from xml.parsers.expat import ExpatError

CAP_MARGIN = 0.3
SUSTAINED_SHARE = 0.5
# Below this share of a sample in any requested or applied state the GPU barely ran, and the
# mean state index means nothing.
MIN_STATE_SHARE = 0.05


@dataclass(frozen=True)
class GpuSample:
    end_unix: float
    elapsed_s: float
    busy: float
    requested: float | None
    applied: float | None
    steps: Mapping[int, float]


@dataclass(frozen=True)
class ChunkSpan:
    engine: str
    start_unix: float
    end_unix: float


class Regime(Enum):
    PEAK = "Peak"
    SUSTAINED = "Sustained"
    TRANSITION = "Transition"


def _mean_state(states: Sequence[Mapping[str, Any]], key: str, prefix: str) -> float | None:
    share = sum(float(s["used_ratio"]) for s in states)
    if share <= MIN_STATE_SHARE:
        return None
    total = sum(int(str(s[key]).removeprefix(prefix)) * float(s["used_ratio"]) for s in states)
    return total / share


def read_powermetrics(path: Path) -> list[GpuSample]:
    """Samples in log order. Timestamps have one-second resolution, so the clock is rebuilt
    from the first timestamp plus each sample's ``elapsed_ns``."""
    data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    samples: list[GpuSample] = []
    clock = 0.0
    for part in data.split(b"\x00"):
        if not part.strip():
            continue
        try:
            raw = plistlib.loads(part)
        except (plistlib.InvalidFileException, ExpatError):
            continue  # the sample powermetrics was writing when it stopped
        gpu = raw["gpu"]
        elapsed = raw["elapsed_ns"] / 1e9
        clock = calendar.timegm(raw["timestamp"].timetuple()) if not samples else clock + elapsed
        samples.append(
            GpuSample(
                end_unix=clock,
                elapsed_s=elapsed,
                busy=1.0 - float(gpu["idle_ratio"]),
                requested=_mean_state(gpu.get("sw_requested_state", []), "sw_req_state", "P"),
                applied=_mean_state(gpu.get("sw_state", []), "sw_state", "SW_P"),
                steps={
                    int(d["freq"]): d["used_ratio"]
                    for d in gpu["dvfm_states"]
                    if d["used_ratio"] > 0
                },
            )
        )
    return samples


def capped(sample: GpuSample) -> bool:
    if sample.requested is None or sample.applied is None:
        return False
    return sample.applied < sample.requested - CAP_MARGIN


def _overlapping(span: ChunkSpan, samples: Sequence[GpuSample]) -> list[GpuSample]:
    return [
        s
        for s in samples
        if s.end_unix > span.start_unix and s.end_unix - s.elapsed_s < span.end_unix
    ]


def capped_share(span: ChunkSpan, samples: Sequence[GpuSample]) -> float:
    """The share of the span's busy GPU time spent in capped samples."""
    inside = _overlapping(span, samples)
    busy = sum(s.busy * s.elapsed_s for s in inside)
    held = sum(s.busy * s.elapsed_s for s in inside if capped(s))
    return held / busy if busy else 0.0


def classify(
    block: Sequence[ChunkSpan], samples: Sequence[GpuSample], deciders: tuple[str, str]
) -> Regime:
    """``deciders`` are the baseline and its twin: ``mlx_lm.server`` asks for the top state
    whenever it is busy, so the cap always shows on it."""
    covered = [_overlapping(span, samples) for span in block]
    if not all(covered):
        # The log ended or started late: nothing shows the clock was full.
        return Regime.TRANSITION
    if not any(capped(s) for inside in covered for s in inside):
        return Regime.PEAK
    spans = [span for span in block if span.engine in deciders]
    if len(spans) == len(deciders) and all(
        capped_share(span, samples) >= SUSTAINED_SHARE for span in spans
    ):
        return Regime.SUSTAINED
    return Regime.TRANSITION
