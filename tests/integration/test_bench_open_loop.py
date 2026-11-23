"""The open-loop load generator: Poisson arrivals from a seed, sent whatever the server does
(decision 0025)."""

import asyncio
import statistics
import time

import pytest
from bench.loadgen import arrival_offsets, open_loop
from bench.schema import Message, Sent, WorkloadRequest
from tests.recording import RecordingRunner
from tests.serving import SERVED, serving

from ullm.runner.base import RunnerInfo
from ullm.runner.fake_runner import FakeRunner, FakeTokenizer, Latency
from ullm.telemetry import events as ev

INFO = RunnerInfo(model_id="fake", vocab_size=300, eos_token_ids=frozenset({0}), max_model_len=512)
# Decode is slow next to the arrival gaps, so a closed loop of one would never overlap.
SLOW = Latency(
    prefill_base_ms=0.2, prefill_per_token_ms=0.0, decode_base_ms=20.0, decode_per_row_ms=0.0
)
KWARGS = {"enable_thinking": False}


def _request(i: int) -> WorkloadRequest:
    content = "x" * (i + 1)
    prompt = len(FakeTokenizer().render([{"role": "user", "content": content}], {}))
    return WorkloadRequest(
        id=f"r{i}",
        messages=(Message(role="user", content=content),),
        max_tokens=8,
        prompt_tokens=prompt,
    )


def test_the_same_seed_gives_the_same_arrivals() -> None:
    assert arrival_offsets(50, 4.0, seed=7) == arrival_offsets(50, 4.0, seed=7)
    assert arrival_offsets(50, 4.0, seed=7) != arrival_offsets(50, 4.0, seed=8)


def test_arrivals_rise_from_zero_at_the_mean_rate() -> None:
    offsets = arrival_offsets(4000, 5.0, seed=0)
    assert offsets[0] > 0
    gaps = [b - a for a, b in zip([0.0, *offsets], offsets, strict=False)]
    assert all(g > 0 for g in gaps)
    assert statistics.fmean(gaps) == pytest.approx(1 / 5.0, rel=0.05)
    # Exponential gaps: the spread equals the mean. A fixed interval has none.
    assert statistics.stdev(gaps) == pytest.approx(1 / 5.0, rel=0.1)


@pytest.mark.parametrize("rate", [0.0, -1.0, float("inf"), float("nan")])
def test_a_rate_that_is_not_positive_and_finite_is_refused(rate: float) -> None:
    with pytest.raises(ValueError, match="rate"):
        arrival_offsets(3, rate, seed=0)


def _run(requests: list[WorkloadRequest], rate: float) -> tuple[list[Sent], float]:
    async def main() -> tuple[list[Sent], float]:
        events: list[ev.Event] = []
        async with serving(RecordingRunner(FakeRunner(INFO, SLOW)), events) as client:
            start = time.time()
            sent = await open_loop(client, SERVED, requests, rate, 0, KWARGS, extra_body={})
            return sent, start

    return asyncio.run(asyncio.wait_for(main(), timeout=30))


def test_requests_go_at_their_arrival_times_without_waiting_for_replies() -> None:
    requests = [_request(i) for i in range(10)]
    rate = 50.0
    records, start = _run(requests, rate)
    assert [r.id for r in records] == [q.id for q in requests]
    assert all(r.error is None for r in records)
    for record, offset in zip(records, arrival_offsets(10, rate, seed=0), strict=True):
        assert record.sent_unix - start == pytest.approx(offset, abs=0.05)
    # Eight decode steps of 20 ms each outlast the mean gap of 20 ms, so replies overlap.
    ends = [r.sent_unix + r.end_ms / 1000 for r in records]
    assert any(records[i + 1].sent_unix < ends[i] for i in range(len(records) - 1))
