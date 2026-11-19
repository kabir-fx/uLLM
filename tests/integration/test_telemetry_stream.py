"""O1: ``/telemetry/stream`` and ``--record`` on a real uvicorn server (decision 0023)."""

import asyncio
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import httpx2
import jsonschema
from fastapi import FastAPI
from tests.serving import SERVED, serving_app

from ullm.runner.base import RunnerInfo
from ullm.runner.fake_runner import FakeRunner, FakeTokenizer, Latency
from ullm.server.app import ServeConfig, create_app
from ullm.telemetry import events as ev
from ullm.telemetry.check import check_trace
from ullm.telemetry.export import render_schema
from ullm.telemetry.trace import iter_frames

INFO = RunnerInfo(model_id="fake", vocab_size=300, eos_token_ids=frozenset({0}), max_model_len=4096)
FAST = Latency(
    prefill_base_ms=1.0, prefill_per_token_ms=0.0, decode_base_ms=1.0, decode_per_row_ms=0.0
)
BODY = {"model": SERVED, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}


def _app(record: Path | None = None, keepalive_s: float = 10.0) -> FastAPI:
    config = ServeConfig(served_model=SERVED, record=record, telemetry_keepalive_s=keepalive_s)
    return create_app(FakeRunner(INFO, FAST), FakeTokenizer(), config)


def _run(main: Callable[[], Awaitable[None]]) -> None:
    asyncio.run(asyncio.wait_for(main(), timeout=20))


async def _data_lines(response: httpx2.Response) -> AsyncIterator[str]:
    async for line in response.aiter_lines():
        if line.startswith("data: "):
            yield line.removeprefix("data: ")


async def _frames_until_finished(lines: AsyncIterator[str], requests: int) -> list[str]:
    wire: list[str] = []
    finished = 0
    while finished < requests:
        wire.append(await anext(lines))
        finished += wire[-1].count('"type":"req.finished"')
    return wire


def test_a_tab_gets_hello_then_schema_valid_frames_that_tell_a_whole_story() -> None:
    wire: list[str] = []

    async def main() -> None:
        async with (
            serving_app(_app()) as client,
            client.stream("GET", "/telemetry/stream") as response,
        ):
            assert response.headers["content-type"].startswith("text/event-stream")
            lines = _data_lines(response)
            post = client.post
            replies = asyncio.gather(*(post("/v1/chat/completions", json=BODY) for _ in "ab"))
            wire.extend(await _frames_until_finished(lines, 2))
            assert all(r.status_code == 200 for r in await replies)

    _run(main)
    validator = jsonschema.Draft202012Validator(json.loads(render_schema()))
    for line in wire:
        validator.validate(json.loads(line))
        assert ev.Frame.model_validate_json(line).model_dump_json() == line
    frames = [ev.Frame.model_validate_json(line) for line in wire]
    assert [f.seq for f in frames] == list(range(len(frames)))
    assert isinstance(frames[0].events[0], ev.Hello)
    assert check_trace(frames).problems == []


def test_each_tab_starts_its_own_numbering_with_hello() -> None:
    async def main() -> None:
        async with serving_app(_app()) as client:
            for _ in range(2):
                async with client.stream("GET", "/telemetry/stream") as response:
                    first = ev.Frame.model_validate_json(await anext(_data_lines(response)))
                    assert (first.seq, first.events[0].type) == (0, "hello")

    _run(main)


def test_a_closed_tab_stops_being_fed() -> None:
    app = _app()

    async def main() -> None:
        async with serving_app(app) as client:
            async with client.stream("GET", "/telemetry/stream") as response:
                await anext(_data_lines(response))
                assert app.state.hub.readers == 1
            deadline = time.monotonic() + 5
            while app.state.hub.readers and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            assert app.state.hub.readers == 0

    _run(main)


def test_an_idle_tab_gets_keepalive_comments() -> None:
    async def main() -> None:
        async with (
            serving_app(_app(keepalive_s=0.05)) as client,
            client.stream("GET", "/telemetry/stream") as response,
        ):
            lines = response.aiter_lines()
            seen = [await anext(lines) for _ in range(6)]
            assert ": keepalive" in seen

    _run(main)


def test_a_recording_holds_the_whole_run_and_passes_the_trace_checker(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl.gz"

    async def main() -> None:
        async with serving_app(_app(record=path)) as client:
            replies = await asyncio.gather(
                *(client.post("/v1/chat/completions", json=BODY) for _ in range(4))
            )
            assert all(r.status_code == 200 for r in replies)

    _run(main)
    frames = list(iter_frames(path))
    assert isinstance(frames[0].events[0], ev.Hello)
    assert check_trace(frames).problems == []
    finished = [e for f in frames for e in f.events if isinstance(e, ev.ReqFinished)]
    assert len(finished) == 4
