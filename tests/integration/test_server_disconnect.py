"""A client that leaves frees its request, through a real uvicorn server (A1, decision 0011)."""

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence

import httpx2
import pytest
from tests.recording import RecordingRunner
from tests.serving import SERVED, serving

from ullm.runner.base import RunnerInfo, TemplateMessage
from ullm.runner.fake_runner import FakeRunner, FakeTokenizer, Latency
from ullm.telemetry import events as ev

INFO = RunnerInfo(model_id="fake", vocab_size=300, eos_token_ids=frozenset({0}), max_model_len=4096)
SLOW_DECODE = Latency(
    prefill_base_ms=0.2, prefill_per_token_ms=0.0, decode_base_ms=20.0, decode_per_row_ms=0.0
)
SLOW_PREFILL = Latency(
    prefill_base_ms=500.0, prefill_per_token_ms=0.0, decode_base_ms=20.0, decode_per_row_ms=0.0
)
BODY = {"model": SERVED, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1000}


def _run(main: Callable[[], Awaitable[None]]) -> None:
    async def bounded() -> None:
        await asyncio.wait_for(main(), timeout=20)

    asyncio.run(bounded())


async def _until_idle(client: httpx2.AsyncClient) -> None:
    while (await client.get("/health")).json()["in_flight"] != 0:
        await asyncio.sleep(0.02)


def _aborted(events: list[ev.Event]) -> list[str]:
    return [e.reason for e in events if isinstance(e, ev.ReqAborted)]


def test_leaving_a_stream_aborts_the_request() -> None:
    runner = RecordingRunner(FakeRunner(INFO, SLOW_DECODE))
    events: list[ev.Event] = []

    async def main() -> None:
        async with serving(runner, events) as client:
            request = {**BODY, "stream": True}
            async with client.stream("POST", "/v1/chat/completions", json=request) as response:
                lines = response.aiter_lines()
                while not (await anext(lines)).startswith('data: {"id"'):
                    pass
                assert (await client.get("/health")).json()["in_flight"] == 1
            await _until_idle(client)

    _run(main)
    assert _aborted(events) == ["client_disconnect"]
    assert runner.adds == runner.releases == 1
    assert len(runner.steps) < 100  # max_tokens is 1000: the engine stopped early


def test_leaving_a_plain_request_during_its_prefill_aborts_it() -> None:
    runner = RecordingRunner(FakeRunner(INFO, SLOW_PREFILL))
    events: list[ev.Event] = []

    async def main() -> None:
        async with serving(runner, events) as client:
            with pytest.raises(httpx2.ReadTimeout):  # the client leaves during the prefill
                await client.post("/v1/chat/completions", json=BODY, timeout=0.1)
            await _until_idle(client)

    _run(main)
    assert _aborted(events) == ["client_disconnect"]
    assert runner.adds == runner.releases == 1
    assert len(runner.steps) <= 2  # the prefill, and at most one decode before the abort


class _SlowRender(FakeTokenizer):
    def render(
        self, messages: Sequence[TemplateMessage], template_kwargs: Mapping[str, object]
    ) -> tuple[int, ...]:
        time.sleep(0.5)  # a long prompt through the real tokenizer: about 2.8 us a token
        return super().render(messages, template_kwargs)


def test_health_stays_fast_while_a_long_prompt_renders() -> None:
    runner = RecordingRunner(FakeRunner(INFO, SLOW_DECODE))
    events: list[ev.Event] = []

    async def main() -> None:
        async with serving(runner, events, _SlowRender()) as client:
            # The client shares the server's loop, so time from the start: a blocked loop
            # answers the health check only after the 0.5 s render.
            began = time.perf_counter()
            request = asyncio.create_task(
                client.post("/v1/chat/completions", json={**BODY, "max_tokens": 1})
            )
            await asyncio.sleep(0.1)  # the render has begun
            assert (await client.get("/health")).status_code == 200
            answered = time.perf_counter() - began
            assert (await request).status_code == 200
            assert answered < 0.4

    _run(main)
