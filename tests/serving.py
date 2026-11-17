"""A real uvicorn server on 127.0.0.1 and a free port, for tests that need a socket."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx2
import uvicorn
from tests.recording import RecordingRunner

from ullm.runner.base import Tokenizer
from ullm.runner.fake_runner import FakeTokenizer
from ullm.server.app import ServeConfig, create_app
from ullm.telemetry import events as ev

SERVED = "fake-model"


@asynccontextmanager
async def serving(
    runner: RecordingRunner, events: list[ev.Event], tokenizer: Tokenizer | None = None
) -> AsyncIterator[httpx2.AsyncClient]:
    config = ServeConfig(served_model=SERVED)
    app = create_app(runner, tokenizer or FakeTokenizer(), config, events.append)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    async with httpx2.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
        yield client
    server.should_exit = True
    await task
