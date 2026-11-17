"""A client that leaves frees its request, through a real uvicorn server (A1, decision 0011)."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import httpx2
import uvicorn
from tests.recording import RecordingRunner

from ullm.runner.base import RunnerInfo
from ullm.runner.fake_runner import FakeRunner, FakeTokenizer, Latency
from ullm.server.app import ServeConfig, create_app
from ullm.telemetry import events as ev

SERVED = "fake-model"
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


@asynccontextmanager
async def serving(
    runner: RecordingRunner, events: list[ev.Event]
) -> AsyncIterator[httpx2.AsyncClient]:
    app = create_app(runner, FakeTokenizer(), ServeConfig(served_model=SERVED), events.append)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    async with httpx2.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
        yield client
    server.should_exit = True
    await task


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
            try:
                await client.post("/v1/chat/completions", json=BODY, timeout=0.1)
            except httpx2.ReadTimeout:
                pass
            else:
                raise AssertionError("the reply came before the prefill could end")
            await _until_idle(client)

    _run(main)
    assert _aborted(events) == ["client_disconnect"]
    assert runner.adds == runner.releases == 1
    assert len(runner.steps) <= 2  # the prefill, and at most one decode before the abort
