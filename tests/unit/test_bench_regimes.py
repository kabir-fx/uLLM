"""Blocks sort into Peak, Sustained and Transition by the capped rule (decision 0016)."""

import gzip
import plistlib
from datetime import UTC, datetime
from pathlib import Path

import pytest
from bench.regimes import (
    ChunkSpan,
    GpuSample,
    Regime,
    capped,
    capped_share,
    classify,
    read_powermetrics,
)


def _sample(
    end: float,
    *,
    requested: float | None = 9.0,
    applied: float | None = 9.0,
    busy: float = 1.0,
    elapsed: float = 1.0,
) -> GpuSample:
    return GpuSample(
        end_unix=end, elapsed_s=elapsed, busy=busy, requested=requested, applied=applied, steps={}
    )


def test_a_sample_is_capped_only_when_applied_is_below_requested_by_more_than_the_margin() -> None:
    assert not capped(_sample(1, requested=9.0, applied=9.0))
    assert not capped(_sample(1, requested=9.0, applied=8.75))
    assert capped(_sample(1, requested=9.0, applied=8.65))
    assert not capped(_sample(1, requested=None, applied=5.0))
    assert not capped(_sample(1, requested=9.0, applied=None))


def test_capped_share_weights_samples_by_busy_time_and_counts_overlapping_samples() -> None:
    samples = [
        _sample(10.0),
        _sample(11.0, applied=7.0, busy=0.5),
        _sample(12.0, applied=7.0, busy=1.0),
        _sample(13.0, busy=0.5),
        _sample(20.0, applied=7.0),
    ]
    span = ChunkSpan("baseline", start_unix=10.5, end_unix=12.5)
    # Samples ending at 11, 12 and 13 overlap [10.5, 12.5]; the one ending at 10 covers [9, 10].
    assert capped_share(span, samples) == pytest.approx(1.5 / 2.0)
    assert capped_share(ChunkSpan("x", 14.0, 15.0), samples) == 0.0


def _block(shares: dict[str, list[bool]]) -> tuple[list[ChunkSpan], list[GpuSample]]:
    """One chunk per engine, back to back; each bool is one busy second, capped or not."""
    spans, samples, t = [], [], 0.0
    for engine, caps in shares.items():
        spans.append(ChunkSpan(engine, t + 0.1, t + len(caps) - 0.1))
        for c in caps:
            t += 1.0
            samples.append(_sample(t, applied=7.0 if c else 9.0))
    return spans, samples


DECIDERS = ("base", "twin")


def test_a_block_with_no_capped_sample_is_peak() -> None:
    spans, samples = _block({"ullm": [False] * 4, "base": [False] * 4, "twin": [False] * 4})
    assert classify(spans, samples, DECIDERS) is Regime.PEAK


def test_one_capped_sample_in_any_chunk_leaves_peak() -> None:
    spans, samples = _block(
        {"ullm": [False, True, False, False], "base": [False] * 4, "twin": [False] * 4}
    )
    assert classify(spans, samples, DECIDERS) is Regime.TRANSITION


def test_a_block_is_sustained_when_baseline_and_twin_are_each_capped_half_their_busy_time() -> None:
    spans, samples = _block(
        {"ullm": [False] * 4, "base": [True, True, False, False], "twin": [True, True, True, False]}
    )
    assert classify(spans, samples, DECIDERS) is Regime.SUSTAINED


def test_spike_1_block_5_is_transition_read_each_though_pooled_it_would_be_sustained() -> None:
    # Spike 1, block 5: uLLM 0.52, baseline 0.41, twin 0.71. Pooled, the deciders give 0.56.
    def caps(share: int) -> list[bool]:
        return [True] * share + [False] * (100 - share)

    spans, samples = _block({"ullm": caps(52), "base": caps(41), "twin": caps(71)})
    assert classify(spans, samples, DECIDERS) is Regime.TRANSITION


def _plist(
    timestamp: datetime,
    elapsed_ns: int,
    requested: dict[int, float],
    applied: dict[int, float],
    idle: float,
) -> bytes:
    sample = {
        "timestamp": timestamp,
        "elapsed_ns": elapsed_ns,
        "gpu": {
            "idle_ratio": idle,
            "dvfm_states": [{"freq": 1470, "used_ratio": 1 - idle}, {"freq": 338, "used_ratio": 0}],
            "sw_requested_state": [
                {"sw_req_state": f"P{p}", "used_ratio": r} for p, r in requested.items()
            ],
            "sw_state": [{"sw_state": f"SW_P{p}", "used_ratio": r} for p, r in applied.items()],
        },
    }
    return plistlib.dumps(sample)


def test_read_powermetrics_rebuilds_the_clock_and_skips_a_cut_off_last_sample(
    tmp_path: Path,
) -> None:
    start = datetime(2026, 11, 18, 11, 5, 50)
    log = b"\x00".join(
        [
            _plist(start, 1_000_000_000, {9: 0.8, 8: 0.2}, {9: 0.5, 7: 0.5}, idle=0.0),
            _plist(start, 1_500_000_000, {9: 0.01}, {9: 0.01}, idle=0.99),
            _plist(start, 1_000_000_000, {9: 1.0}, {9: 1.0}, idle=0.0)[:-40],
        ]
    )
    path = tmp_path / "powermetrics.plist.gz"
    path.write_bytes(gzip.compress(log))
    first, second = read_powermetrics(path)
    # powermetrics stamps UTC without a zone.
    assert first.end_unix == start.replace(tzinfo=UTC).timestamp()
    assert second.end_unix - first.end_unix == pytest.approx(1.5)
    assert first.requested == pytest.approx(8.8)
    assert first.applied == pytest.approx(8.0)
    assert first.busy == pytest.approx(1.0)
    assert first.steps == {1470: 1.0}
    assert second.requested is None and second.applied is None
