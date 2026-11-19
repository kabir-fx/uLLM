"""One request and its state machine (E4, decision 0010). Engine thread only.

Every state change goes through ``_move``, which checks ``TRANSITIONS``, and
each public method emits its event, so no transition happens without one.
"""

from collections.abc import Callable, Mapping
from enum import Enum
from typing import TYPE_CHECKING, Final, Literal

from ullm.core.stop import Chunk, StopCheck
from ullm.runner.base import PrefillPlan, SamplingParams, SeqId
from ullm.telemetry import events as ev

if TYPE_CHECKING:
    from ullm.core.engine import Submit


class RequestState(Enum):
    WAITING = "waiting"
    PREFILLING = "prefilling"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED = "finished"
    ABORTED = "aborted"


_S = RequestState
TRANSITIONS: Final[Mapping[RequestState, frozenset[RequestState]]] = {
    _S.WAITING: frozenset({_S.PREFILLING, _S.ABORTED}),
    _S.PREFILLING: frozenset({_S.RUNNING, _S.ABORTED}),
    _S.RUNNING: frozenset({_S.FINISHED, _S.PREEMPTED, _S.ABORTED}),
    _S.PREEMPTED: frozenset({_S.WAITING, _S.ABORTED}),
    _S.FINISHED: frozenset(),
    _S.ABORTED: frozenset(),
}

AbortReason = Literal["client_disconnect", "shutdown", "error"]  # the C2 req.aborted reasons
Emit = Callable[[ev.Event], None]  # appends; never blocks (rule 5)


class IllegalTransition(ValueError):
    pass


class Request:
    def __init__(self, ticket: "Submit", seq: SeqId, stop: StopCheck, emit: Emit, t: float) -> None:
        self._ticket = ticket
        self._seq = seq
        self._stop = stop
        self._emit = emit
        self._state = _S.WAITING
        self._output: list[int] = []
        self.prefilled = 0  # prompt tokens in the runner's KV: PREFILLING's progress (E13)
        self._arrived = t
        self._admitted = t
        self._first = t
        emit(
            ev.ReqArrived(
                t=t,
                rid=ticket.rid,
                prompt_tokens=len(ticket.prompt),
                max_tokens=ticket.max_tokens,
                tenant=ticket.tenant,
            )
        )

    @property
    def rid(self) -> str:
        return self._ticket.rid

    @property
    def seq(self) -> SeqId:
        return self._seq

    @property
    def prompt(self) -> tuple[int, ...]:
        return self._ticket.prompt

    @property
    def sampling(self) -> SamplingParams:
        return self._ticket.sampling

    @property
    def trace(self) -> str | None:
        return self._ticket.trace

    @property
    def tenant(self) -> str | None:
        return self._ticket.tenant

    @property
    def state(self) -> RequestState:
        return self._state

    @property
    def generated(self) -> int:
        """Tokens sampled for this request, the end token included."""
        return len(self._output)

    @property
    def kv_tokens(self) -> int:
        """Tokens in the runner's KV: the last sampled token is fed by the next decode."""
        return self.prefilled + max(len(self._output) - 1, 0)

    @property
    def kv_reservation(self) -> int:
        """The most slots this request's KV row can ever hold (decision 0020)."""
        return len(self.prompt) + self._ticket.max_tokens

    def admit(self, t: float) -> None:
        self._move(_S.PREFILLING)
        self._admitted = t
        self._emit(ev.ReqAdmitted(t=t, rid=self.rid, queue_ms=t - self._arrived))

    def append(self, token: int, t: float) -> Chunk:
        if self._state not in {_S.PREFILLING, _S.RUNNING}:
            raise IllegalTransition(f"{self.rid}: a token for a request that is {self._state}")
        chunk = self._stop.add(token)
        self._output.append(token)
        if self._state is _S.PREFILLING:
            self._move(_S.RUNNING)
            self.prefilled = len(self.prompt)
            self._first = t
            self._emit(ev.ReqFirstToken(t=t, rid=self.rid, ttft_ms=t - self._arrived))
        if chunk.finish_reason is not None:
            self._move(_S.FINISHED)
            self._emit(self._finished(chunk.finish_reason, t))
        return chunk

    def abort(self, reason: AbortReason, t: float) -> None:
        self._move(_S.ABORTED)
        self._emit(
            ev.ReqAborted(t=t, rid=self.rid, reason=reason, completion_tokens=len(self._output))
        )

    def prefill_plan(self) -> PrefillPlan:
        return PrefillPlan(
            seq=self._seq, token_ids=self.prompt[self.prefilled :], start=self.prefilled
        )

    def decode_input(self) -> int:
        return self._output[-1]

    def _finished(self, reason: Literal["stop", "length"], t: float) -> ev.ReqFinished:
        n = len(self._output)
        return ev.ReqFinished(
            t=t,
            rid=self.rid,
            reason=reason,
            usage=ev.TokenUsage(
                prompt_tokens=len(self.prompt), completion_tokens=n, cached_tokens=0
            ),
            timings=ev.Timings(
                queue_ms=self._admitted - self._arrived,
                ttft_ms=self._first - self._arrived,
                tpot_ms=(t - self._first) / (n - 1) if n > 1 else None,
                e2e_ms=t - self._arrived,
            ),
        )

    def _move(self, to: RequestState) -> None:
        if to not in TRANSITIONS[self._state]:
            raise IllegalTransition(f"{self.rid}: {self._state} cannot move to {to}")
        self._state = to
