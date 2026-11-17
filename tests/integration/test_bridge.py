"""E5's proof without HTTP: ``EngineClient`` on an event loop, the engine on its thread."""

import asyncio
import random
import time
from collections.abc import Awaitable, Callable

import pytest
from tests.recording import RecordingRunner, tokens_by_rid

from ullm.core.client import EngineClient, RequestStream, Unavailable
from ullm.core.engine import Delta, EngineConfig, Failed, Finished, Output
from ullm.runner.base import RunnerInfo, SamplingParams
from ullm.runner.fake_runner import FakeDetokenizer, FakeRunner, Latency
from ullm.telemetry import events as ev
from ullm.telemetry.check import check_trace
from ullm.telemetry.trace import frames_from

INFO = RunnerInfo(model_id="fake", vocab_size=50, eos_token_ids=frozenset({0}), max_model_len=64)
FAST = Latency(
    prefill_base_ms=0.2, prefill_per_token_ms=0.01, decode_base_ms=0.2, decode_per_row_ms=0.05
)
SLOW_PREFILL = Latency(
    prefill_base_ms=500.0, prefill_per_token_ms=0.0, decode_base_ms=0.2, decode_per_row_ms=0.0
)
GREEDY = SamplingParams(temperature=0.0, top_p=1.0, top_k=0, seed=None)


def _run[T](main: Callable[[], Awaitable[T]]) -> T:
    async def bounded() -> T:
        return await asyncio.wait_for(main(), timeout=20)

    return asyncio.run(bounded())


def _start(
    runner: RecordingRunner, events: list[ev.Event], config: EngineConfig | None = None
) -> EngineClient:
    return EngineClient.start(runner, FakeDetokenizer, config or EngineConfig(), events.append)


async def _drain(stream: RequestStream) -> list[Output]:
    async with stream:
        return [out async for out in stream]


def _text(outputs: list[Output]) -> str:
    return "".join(o.text for o in outputs if isinstance(o, Delta | Finished))


def test_fifty_concurrent_clients_each_get_their_own_reply() -> None:
    runner = RecordingRunner(FakeRunner(INFO, FAST))
    events: list[ev.Event] = []

    async def main() -> dict[str, list[Output]]:
        engine = _start(runner, events)
        rng = random.Random(7)
        streams = [
            engine.submit(
                tuple(rng.randrange(50) for _ in range(rng.randint(1, 20))),
                SamplingParams(temperature=0.0, top_p=1.0, top_k=0, seed=i),
                max_tokens=rng.randint(1, 20),
            )
            for i in range(50)
        ]
        replies = await asyncio.gather(*(_drain(s) for s in streams))
        await engine.close()
        return {s.rid: r for s, r in zip(streams, replies, strict=True)}

    replies = _run(main)
    sampled = tokens_by_rid(events, runner.steps)
    assert len(replies) == 50
    for rid, outputs in replies.items():
        assert isinstance(outputs[-1], Finished)
        assert all(isinstance(o, Delta) for o in outputs[:-1])
        assert _text(outputs) == "".join(f" t{t}" for t in sampled[rid])
    assert check_trace(frames_from(events, "r1")).problems == []
    assert runner.adds == runner.releases == 50


def test_every_client_that_leaves_early_frees_its_sequence() -> None:
    runner = RecordingRunner(FakeRunner(INFO, FAST))
    events: list[ev.Event] = []

    async def leave_after_one(stream: RequestStream) -> None:
        async with stream:
            async for _ in stream:
                break

    async def main() -> None:
        engine = _start(runner, events)
        streams = [engine.submit((1, 2, 3), GREEDY, max_tokens=40) for _ in range(20)]
        await asyncio.gather(*(leave_after_one(s) for s in streams))
        await engine.close()

    _run(main)
    assert runner.adds == runner.releases
    assert runner.live == set()
    aborted = [e for e in events if isinstance(e, ev.ReqAborted)]
    assert {e.reason for e in aborted} == {"client_disconnect"}
    assert check_trace(frames_from(events, "r1")).problems == []


def test_a_stream_left_after_its_end_sends_no_abort() -> None:
    runner = RecordingRunner(FakeRunner(INFO, FAST))
    events: list[ev.Event] = []

    async def main() -> None:
        engine = _start(runner, events)
        await _drain(engine.submit((1,), GREEDY, max_tokens=3))
        await engine.close()

    _run(main)
    assert not [e for e in events if isinstance(e, ev.ReqAborted)]


def test_the_event_loop_and_health_stay_fast_during_a_long_prefill() -> None:
    runner = RecordingRunner(FakeRunner(INFO, SLOW_PREFILL))
    events: list[ev.Event] = []

    async def main() -> tuple[float, float, list[Output]]:
        engine = _start(runner, events)
        stream = engine.submit((1, 2), GREEDY, max_tokens=2)
        await asyncio.sleep(0.1)  # the engine thread is now inside the 500 ms prefill
        started = time.perf_counter()
        await asyncio.sleep(0.01)
        slept = time.perf_counter() - started
        started = time.perf_counter()
        health = engine.health()
        asked = time.perf_counter() - started
        assert health.alive
        assert health.in_flight == 1
        reply = await _drain(stream)
        await engine.close()
        return slept, asked, reply

    slept, asked, reply = _run(main)
    assert slept < 0.1  # a blocked loop would wake only after the prefill, about 0.4 s later
    assert asked < 0.01
    assert isinstance(reply[-1], Finished)


class _DiesOnAdd(RecordingRunner):
    def add(self, seq: int, sampling: SamplingParams) -> None:
        raise RuntimeError("a bug in the core")


def test_an_engine_that_dies_fails_every_open_stream_and_reports_not_alive() -> None:
    events: list[ev.Event] = []

    async def main() -> None:
        engine = _start(_DiesOnAdd(FakeRunner(INFO, FAST)), events)
        streams = [engine.submit((1,), GREEDY, max_tokens=3) for _ in range(3)]
        replies = await asyncio.gather(*(_drain(s) for s in streams))
        assert replies == [[Failed("error")]] * 3
        assert not engine.health().alive
        with pytest.raises(Unavailable) as refused:
            engine.submit((1,), GREEDY, max_tokens=3)
        assert refused.value.why == "stopped"
        await engine.close()

    _run(main)
    assert check_trace(frames_from(events, "r1")).problems == []


def test_open_requests_are_bounded_at_max_in_flight() -> None:
    runner = RecordingRunner(FakeRunner(INFO, SLOW_PREFILL))
    events: list[ev.Event] = []

    async def main() -> None:
        engine = _start(runner, events, EngineConfig(max_in_flight=2))
        first = [engine.submit((1,), GREEDY, max_tokens=1) for _ in range(2)]
        with pytest.raises(Unavailable) as refused:
            engine.submit((1,), GREEDY, max_tokens=1)
        assert refused.value.why == "full"
        await asyncio.gather(*(_drain(s) for s in first))
        assert engine.health().in_flight == 0
        await _drain(engine.submit((1,), GREEDY, max_tokens=1))
        await engine.close()

    _run(main)


@pytest.mark.parametrize(
    ("prompt", "max_tokens", "match"),
    [
        ((), 1, "prompt"),
        ((1, 50), 1, "vocabulary"),
        ((1,) * 60, 5, "max_model_len"),
        ((1,), 0, "max_tokens"),
    ],
)
def test_a_request_the_model_cannot_run_is_refused(
    prompt: tuple[int, ...], max_tokens: int, match: str
) -> None:
    async def main() -> None:
        engine = _start(RecordingRunner(FakeRunner(INFO, FAST)), [])
        with pytest.raises(ValueError, match=match):
            engine.submit(prompt, GREEDY, max_tokens=max_tokens)
        assert engine.health().in_flight == 0
        await engine.close()

    _run(main)


def test_close_fails_open_streams_with_shutdown_and_refuses_new_ones() -> None:
    events: list[ev.Event] = []

    async def main() -> None:
        engine = _start(RecordingRunner(FakeRunner(INFO, SLOW_PREFILL)), events)
        streams = [engine.submit((1,), GREEDY, max_tokens=3) for _ in range(3)]
        await engine.close()
        await engine.close()
        replies = await asyncio.gather(*(_drain(s) for s in streams))
        assert all(r[-1] == Failed("shutdown") for r in replies)
        assert not engine.health().alive
        with pytest.raises(Unavailable) as refused:
            engine.submit((1,), GREEDY, max_tokens=3)
        assert refused.value.why == "stopped"

    _run(main)
    assert check_trace(frames_from(events, "r1")).problems == []
