"""The API thread's side of the bridge (E5, decision 0010).

Every method runs on the event loop's thread. The engine thread reaches the
client only through ``call_soon_threadsafe``, so all client state lives on one
thread and needs no lock. Every accepted ``submit`` ends with exactly one
``Finished`` or ``Failed``: the engine sends one for each request it ends, and
the exit callback fails the rest.
"""

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from queue import SimpleQueue
from threading import Thread
from typing import Literal

from ullm.core.block_manager import KvUsage
from ullm.core.engine import (
    Abort,
    Engine,
    EngineConfig,
    Failed,
    Finished,
    Load,
    Output,
    Shutdown,
    Submit,
    Ticket,
    pool_blocks,
)
from ullm.core.request import Emit
from ullm.runner.base import Detokenizer, Runner, RunnerInfo, SamplingParams
from ullm.telemetry.bus import TelemetryClock

_log = logging.getLogger(__name__)


class Unavailable(Exception):
    def __init__(self, why: Literal["full", "stopped"]) -> None:
        super().__init__(why)
        self.why = why


@dataclass(frozen=True, slots=True)
class Health:
    alive: bool
    in_flight: int
    waiting: int
    running: int
    kv: KvUsage


class EngineClient:
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        info: RunnerInfo,
        config: EngineConfig,
        inbound: "SimpleQueue[Ticket]",
    ) -> None:
        """Use ``start``."""
        self._loop = loop
        self._info = info
        self._config = config
        self._inbound = inbound
        self._thread: Thread | None = None
        self._open: dict[str, asyncio.Queue[Output]] = {}
        self._idle = Load(0, 0, KvUsage.idle(pool_blocks(config, info), info.block_size))
        self._load = self._idle
        self._submitted = 0
        self._closing = False
        self._exited = asyncio.Event()

    @classmethod
    def start(
        cls,
        runner: Runner,
        new_detokenizer: Callable[[], Detokenizer],
        config: EngineConfig,
        emit: Emit,
        clock: TelemetryClock | None = None,
    ) -> "EngineClient":
        """Binds the client to the running loop and starts the engine thread."""
        inbound: SimpleQueue[Ticket] = SimpleQueue()
        client = cls(asyncio.get_running_loop(), runner.info(), config, inbound)
        engine = Engine(runner, new_detokenizer, config, inbound, client._deliver, emit, clock)
        client._thread = Thread(
            target=client._run_engine, args=(engine,), name="ullm-engine", daemon=True
        )
        client._thread.start()
        return client

    def submit(
        self,
        prompt: tuple[int, ...],
        sampling: SamplingParams,
        *,
        max_tokens: int,
        stop: tuple[str, ...] = (),
        trace: str | None = None,
    ) -> "RequestStream":
        """Raises ``ValueError`` for a request this model cannot run, ``Unavailable`` when the
        engine cannot take it now."""
        rid = f"{self._config.replica}-{self._submitted}"
        ticket = Submit(rid, prompt, sampling, max_tokens, stop, trace)
        if max(prompt) >= self._info.vocab_size:
            raise ValueError(
                f"prompt token ids must be in the vocabulary [0, {self._info.vocab_size})"
            )
        if len(prompt) + max_tokens > self._info.max_model_len:
            raise ValueError(
                f"prompt ({len(prompt)} tokens) plus max_tokens ({max_tokens}) exceeds "
                f"max_model_len {self._info.max_model_len}"
            )
        if self._closing or self._exited.is_set():
            raise Unavailable("stopped")
        if len(self._open) >= self._config.max_in_flight:
            raise Unavailable("full")
        self._submitted += 1
        queue: asyncio.Queue[Output] = asyncio.Queue()
        self._open[rid] = queue
        self._inbound.put(ticket)
        return RequestStream(self, rid, queue)

    def health(self) -> Health:
        """Reads only loop-thread state, so it never waits on the engine."""
        return Health(
            alive=not self._exited.is_set()
            and self._thread is not None
            and self._thread.is_alive(),
            in_flight=len(self._open),
            waiting=self._load.waiting,
            running=self._load.running,
            kv=self._load.kv,
        )

    async def close(self) -> None:
        if not self._closing and not self._exited.is_set():
            self._inbound.put(Shutdown())
        self._closing = True
        await self._exited.wait()

    def _abandon(self, rid: str) -> None:
        """The caller left before the end. The request stays open until the engine ends it."""
        if rid in self._open and not self._exited.is_set():
            self._inbound.put(Abort(rid))

    def _run_engine(self, engine: Engine) -> None:
        """The engine thread's body."""
        error: BaseException | None = None
        try:
            engine.run()
        except BaseException as e:
            error = e
            _log.exception("the engine thread died")
        with contextlib.suppress(RuntimeError):  # the loop has closed: no stream is left
            self._loop.call_soon_threadsafe(self._engine_exited, error)

    def _deliver(self, batch: Sequence[tuple[str, Output]], load: Load) -> None:
        """Called on the engine thread once per turn: one wakeup of the loop per step."""
        self._loop.call_soon_threadsafe(self._on_outputs, batch, load)

    def _on_outputs(self, batch: Sequence[tuple[str, Output]], load: Load) -> None:
        self._load = load
        for rid, out in batch:
            queue = self._open.get(rid)
            if queue is None:
                continue
            queue.put_nowait(out)
            if isinstance(out, Finished | Failed):
                del self._open[rid]

    def _engine_exited(self, error: BaseException | None) -> None:
        # After a Shutdown the engine has ended every request it was sent, so any left open
        # here were cut off by a dying engine.
        for queue in self._open.values():
            queue.put_nowait(Failed("error"))
        self._open.clear()
        self._load = self._idle
        self._exited.set()


class RequestStream(AsyncIterator[Output]):
    """Zero or more ``Delta``, then one ``Finished`` or ``Failed``. Use it with ``async with``:
    leaving early sends an abort."""

    def __init__(self, client: EngineClient, rid: str, queue: "asyncio.Queue[Output]") -> None:
        self.rid = rid
        self._client = client
        self._queue = queue
        self._ended = False

    async def __anext__(self) -> Output:
        if self._ended:
            raise StopAsyncIteration
        out = await self._queue.get()
        self._ended = isinstance(out, Finished | Failed)
        return out

    async def __aenter__(self) -> "RequestStream":
        return self

    async def __aexit__(self, *exc: object) -> None:
        if not self._ended:
            self._client._abandon(self.rid)
