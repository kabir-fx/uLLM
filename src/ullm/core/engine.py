"""The engine thread's loop, and the types that cross the bridge (E5, decision 0010).

Only the engine thread runs ``Engine.run``, and only it touches the runner, the
scheduler and the requests (rule 2). Tickets arrive on a ``SimpleQueue``; each
loop turn hands everything it produced to ``deliver`` in one call.
"""

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from queue import Empty, SimpleQueue
from typing import Literal

from ullm.core.request import AbortReason, Emit, Request, RequestState
from ullm.core.scheduler import Scheduled, Scheduler
from ullm.core.stop import FinishReason, StopCheck
from ullm.runner.base import (
    DecodePlan,
    Detokenizer,
    PrefillPlan,
    Runner,
    SamplingParams,
    require_ints,
)
from ullm.telemetry import events as ev

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Submit:
    """API thread to engine thread: one new request. Plain data, checked when built."""

    rid: str
    prompt: tuple[int, ...]
    sampling: SamplingParams
    max_tokens: int
    stop: tuple[str, ...]
    trace: str | None = None  # W3C traceparent, carried across the thread boundary (C4)
    tenant: str | None = None  # DEC-9 hook; set from M8

    def __post_init__(self) -> None:
        if type(self.rid) is not str or not self.rid:
            raise ValueError(f"rid must be a non-empty str, got {self.rid!r}")
        if type(self.prompt) is not tuple or not self.prompt:
            raise ValueError(f"prompt must be a non-empty tuple, got {self.prompt!r}")
        require_ints("prompt token ids", self.prompt)
        if min(self.prompt) < 0:
            raise ValueError(f"prompt token ids must be >= 0, got {self.prompt!r}")
        if not isinstance(self.sampling, SamplingParams):
            raise ValueError(f"sampling must be SamplingParams, got {self.sampling!r}")
        if type(self.max_tokens) is not int or self.max_tokens < 1:
            raise ValueError(f"max_tokens must be an int >= 1, got {self.max_tokens!r}")
        if type(self.stop) is not tuple or not all(type(s) is str and s for s in self.stop):
            raise ValueError(f"stop must be a tuple of non-empty strings, got {self.stop!r}")
        for name in ("trace", "tenant"):
            value = getattr(self, name)
            if value is not None and type(value) is not str:
                raise ValueError(f"{name} must be a str or None, got {value!r}")


@dataclass(frozen=True, slots=True)
class Abort:
    rid: str  # the client went away


@dataclass(frozen=True, slots=True)
class Shutdown:
    pass


Ticket = Submit | Abort | Shutdown


@dataclass(frozen=True, slots=True)
class Delta:
    text: str  # never empty


@dataclass(frozen=True, slots=True)
class Finished:
    text: str  # the reply's last piece, possibly empty: the deltas plus this are the reply
    reason: FinishReason
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True, slots=True)
class Failed:
    reason: AbortReason


Output = Delta | Finished | Failed


@dataclass(frozen=True, slots=True)
class Load:
    waiting: int
    running: int


Deliver = Callable[[Sequence[tuple[str, Output]], Load], None]  # once per loop turn; never blocks


@dataclass(frozen=True, slots=True)
class EngineConfig:
    max_num_seqs: int = 1
    max_in_flight: int = 64  # submitted and not yet ended; beyond it, Unavailable
    replica: str = "r1"
    source: Literal["engine", "sim"] = "engine"

    def __post_init__(self) -> None:
        if self.max_num_seqs != 1:
            raise ValueError(
                f"max_num_seqs is 1 until batched decode arrives in M2-T2, got {self.max_num_seqs}"
            )
        if type(self.max_in_flight) is not int or self.max_in_flight < 1:
            raise ValueError(f"max_in_flight must be an int >= 1, got {self.max_in_flight!r}")


class Engine:
    def __init__(
        self,
        runner: Runner,
        new_detokenizer: Callable[[], Detokenizer],
        config: EngineConfig,
        inbound: "SimpleQueue[Ticket]",
        deliver: Deliver,
        emit: Emit,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._runner = runner
        self._info = runner.info()
        self._new_detokenizer = new_detokenizer
        self._config = config
        self._inbound = inbound
        self._deliver = deliver
        self._emit = emit
        self._clock = clock
        self._epoch = clock()
        self._epoch_unix_ms = int(time.time() * 1000)
        self._scheduler = Scheduler(config.max_num_seqs)
        self._live: dict[str, Request] = {}
        self._next_seq = 0
        self._steps = 0

    def run(self) -> None:
        """The engine thread's body. Returns after a Shutdown ticket."""
        self._emit(self._hello())
        try:
            while True:
                out: list[tuple[str, Output]] = []
                running = self._take_tickets(out)
                if running:
                    self._step(out)
                self._deliver(out, Load(self._scheduler.waiting, self._scheduler.running))
                if not running:
                    return
        except BaseException:
            # A bug in the core. End every request in the trace, then let the thread die.
            t = self._now()
            for req in self._live.values():
                req.abort("error", t)
            raise

    def _take_tickets(self, out: list[tuple[str, Output]]) -> bool:
        """Every ticket queued now; waits for one only when idle. False after a Shutdown."""
        idle = not self._live
        while True:
            try:
                ticket = self._inbound.get() if idle else self._inbound.get_nowait()
            except Empty:
                return True
            idle = False
            match ticket:
                case Submit():
                    self._arrive(ticket)
                case Abort(rid=rid):
                    if rid in self._live:  # else it ended before the abort arrived
                        self._end(self._live[rid], "client_disconnect", out)
                case Shutdown():
                    for req in list(self._live.values()):
                        self._end(req, "shutdown", out)
                    return False

    def _arrive(self, ticket: Submit) -> None:
        stop = StopCheck(
            self._new_detokenizer(), self._info.eos_token_ids, ticket.max_tokens, ticket.stop
        )
        req = Request(ticket, self._next_seq, stop, self._emit, self._now())
        self._next_seq += 1
        self._live[req.rid] = req
        self._scheduler.add(req)

    def _step(self, out: list[tuple[str, Output]]) -> None:
        scheduled = self._scheduler.schedule(self._now())
        if scheduled is None:
            return
        if scheduled.admitted is not None:
            self._runner.add(scheduled.admitted.seq, scheduled.admitted.sampling)
        try:
            result = self._runner.execute(scheduled.plan)
        except Exception:
            # The runner has failed these sequences (decision 0008, invariant 4).
            _log.exception("step failed; failing %s", [r.rid for r in scheduled.rows])
            for req in scheduled.rows:
                self._end(req, "error", out)
            return
        t = self._now()
        self._emit(self._step_event(scheduled, result.total_ms, t))
        for req, token in zip(scheduled.rows, result.tokens, strict=True):
            chunk = req.append(token, t)
            if chunk.finish_reason is None:
                if chunk.text:
                    out.append((req.rid, Delta(chunk.text)))
                continue
            self._forget(req)
            out.append(
                (
                    req.rid,
                    Finished(chunk.text, chunk.finish_reason, len(req.prompt), req.generated),
                )
            )

    def _end(self, req: Request, reason: AbortReason, out: list[tuple[str, Output]]) -> None:
        self._forget(req)
        req.abort(reason, self._now())
        out.append((req.rid, Failed(reason)))

    def _forget(self, req: Request) -> None:
        if req.state is not RequestState.WAITING:
            self._runner.release(req.seq)
        self._scheduler.remove(req)
        del self._live[req.rid]

    def _step_event(self, scheduled: Scheduled, ms: float, t: float) -> ev.Step:
        match scheduled.plan:
            case PrefillPlan(token_ids=token_ids):
                kind: Literal["prefill", "decode"] = "prefill"
                tokens = len(token_ids)
            case DecodePlan(seqs=seqs):
                kind = "decode"
                tokens = len(seqs)
        self._steps += 1
        return ev.Step(
            t=t,
            n=self._steps,
            kind=kind,
            rids=[r.rid for r in scheduled.rows],
            tokens=tokens,
            ms=ms,
            waiting=self._scheduler.waiting,
            running=self._scheduler.running,
            # Before the rows append: the KV before the step, plus what the step wrote.
            kv_tokens=sum(r.kv_tokens for r in self._live.values()) + tokens,
        )

    def _hello(self) -> ev.Hello:
        return ev.Hello(
            t=self._now(),
            replica=self._config.replica,
            model=self._info.model_id,
            source=self._config.source,
            version=version("ullm"),
            commit=None,
            epoch_unix_ms=self._epoch_unix_ms,
            limits=ev.Limits(
                max_num_seqs=self._config.max_num_seqs,
                # One step prefills one whole prompt (DEC-7) until chunked prefill (E13).
                max_prefill_tokens=self._info.max_model_len,
                max_model_len=self._info.max_model_len,
            ),
            kv=ev.KvInfo(),
            machine=None,
        )

    def _now(self) -> float:
        return (self._clock() - self._epoch) * 1000
