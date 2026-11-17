"""E4: one request's state machine. Every move is in ``TRANSITIONS`` and emits its event."""

import pytest

from ullm.core.engine import Submit
from ullm.core.request import TRANSITIONS, IllegalTransition, Request, RequestState
from ullm.core.stop import StopCheck
from ullm.runner.base import DecodePlan, PrefillPlan, SamplingParams
from ullm.runner.fake_runner import FakeDetokenizer
from ullm.telemetry import events as ev

S = RequestState
GREEDY = SamplingParams(temperature=0.0, top_p=1.0, top_k=0, seed=None)
EOS = 0


def _request(
    events: list[ev.Event], max_tokens: int = 3, stop: tuple[str, ...] = (), t: float = 10.0
) -> Request:
    ticket = Submit(rid="r1-0", prompt=(5, 6, 7), sampling=GREEDY, max_tokens=max_tokens, stop=stop)
    check = StopCheck(FakeDetokenizer(), frozenset({EOS}), max_tokens, stop)
    return Request(ticket, seq=4, stop=check, emit=events.append, t=t)


def _in_state(state: RequestState) -> tuple[Request, list[ev.Event]]:
    events: list[ev.Event] = []
    req = _request(events, max_tokens=2)
    if state is S.WAITING:
        return req, events
    if state is S.ABORTED:
        req.abort("client_disconnect", 11.0)
        return req, events
    req.admit(11.0)
    if state is S.PREFILLING:
        return req, events
    req.append(1, 12.0)
    if state is S.PREEMPTED:
        req._move(S.PREEMPTED)
    if state is S.FINISHED:
        req.append(2, 13.0)
    return req, events


def test_transitions_name_every_state() -> None:
    assert set(TRANSITIONS) == set(RequestState)


@pytest.mark.parametrize(
    ("before", "to"),
    [(a, b) for a in RequestState for b in RequestState if b not in TRANSITIONS[a]],
    ids=str,
)
def test_an_illegal_move_raises_and_changes_nothing(before: RequestState, to: RequestState) -> None:
    req, events = _in_state(before)
    assert req.state is before
    seen = len(events)
    with pytest.raises(IllegalTransition):
        req._move(to)
    assert req.state is before
    assert len(events) == seen


def test_a_reply_emits_its_life_cycle_with_timings() -> None:
    events: list[ev.Event] = []
    req = _request(events, max_tokens=3, t=10.0)
    states = [req.state]
    req.admit(12.0)
    states.append(req.state)
    first = req.append(1, 15.0)
    states.append(req.state)
    req.append(2, 17.0)
    last = req.append(3, 21.0)
    states.append(req.state)
    assert states == [S.WAITING, S.PREFILLING, S.RUNNING, S.FINISHED]
    assert (first.text, first.finish_reason) == (" t1", None)
    assert (last.text, last.finish_reason) == (" t3", "length")
    assert events == [
        ev.ReqArrived(t=10.0, rid="r1-0", prompt_tokens=3, max_tokens=3),
        ev.ReqAdmitted(t=12.0, rid="r1-0", queue_ms=2.0),
        ev.ReqFirstToken(t=15.0, rid="r1-0", ttft_ms=5.0),
        ev.ReqFinished(
            t=21.0,
            rid="r1-0",
            reason="length",
            usage=ev.TokenUsage(prompt_tokens=3, completion_tokens=3, cached_tokens=0),
            timings=ev.Timings(queue_ms=2.0, ttft_ms=5.0, tpot_ms=3.0, e2e_ms=11.0),
        ),
    ]


def test_one_token_finishes_on_its_first_append_without_a_tpot() -> None:
    events: list[ev.Event] = []
    req = _request(events, max_tokens=1)
    req.admit(11.0)
    chunk = req.append(EOS, 12.0)
    assert (chunk.text, chunk.finish_reason) == ("", "stop")
    assert req.state is S.FINISHED
    assert [e.type for e in events] == [
        "req.arrived",
        "req.admitted",
        "req.first_token",
        "req.finished",
    ]
    finished = events[-1]
    assert isinstance(finished, ev.ReqFinished)
    assert finished.usage.completion_tokens == 1  # the end token counts
    assert finished.timings.tpot_ms is None


def test_a_stop_string_finishes_the_reply() -> None:
    events: list[ev.Event] = []
    req = _request(events, max_tokens=10, stop=(" t2",))
    req.admit(11.0)
    req.append(1, 12.0)
    chunk = req.append(2, 13.0)
    assert (chunk.text, chunk.finish_reason) == ("", "stop")
    assert req.state is S.FINISHED


@pytest.mark.parametrize("state", [S.WAITING, S.PREFILLING, S.RUNNING, S.PREEMPTED])
def test_a_live_request_aborts_with_its_token_count(state: RequestState) -> None:
    req, events = _in_state(state)
    req.abort("shutdown", 20.0)
    assert req.state is S.ABORTED
    sampled = 0 if state in {S.WAITING, S.PREFILLING} else 1
    assert events[-1] == ev.ReqAborted(
        t=20.0, rid="r1-0", reason="shutdown", completion_tokens=sampled
    )


@pytest.mark.parametrize("state", [S.FINISHED, S.ABORTED])
def test_an_ended_request_cannot_abort(state: RequestState) -> None:
    req, events = _in_state(state)
    seen = len(events)
    with pytest.raises(IllegalTransition):
        req.abort("error", 20.0)
    assert len(events) == seen


@pytest.mark.parametrize("state", [S.WAITING, S.FINISHED, S.ABORTED, S.PREEMPTED])
def test_append_outside_prefilling_or_running_raises_and_changes_nothing(
    state: RequestState,
) -> None:
    req, events = _in_state(state)
    seen = len(events)
    with pytest.raises(IllegalTransition):
        req.append(1, 20.0)
    assert req.state is state
    assert len(events) == seen


@pytest.mark.parametrize("state", [S.PREFILLING, S.RUNNING, S.FINISHED, S.ABORTED])
def test_admit_outside_waiting_raises(state: RequestState) -> None:
    req, _ = _in_state(state)
    with pytest.raises(IllegalTransition):
        req.admit(20.0)


def test_plans_feed_the_prompt_then_the_last_token() -> None:
    req, _ = _in_state(S.PREFILLING)
    assert req.prefill_plan() == PrefillPlan(seq=4, token_ids=(5, 6, 7), start=0)
    assert req.kv_tokens == 0
    req.append(9, 12.0)
    assert req.prefilled == 3
    assert req.kv_tokens == 3
    assert DecodePlan(seqs=(req.seq,), token_ids=(req.decode_input(),)) == DecodePlan(
        seqs=(4,), token_ids=(9,)
    )
    req.append(8, 13.0)
    assert req.kv_tokens == 4  # the token just sampled is not in the KV yet
    assert req.decode_input() == 8


def test_the_ticket_fields_are_carried() -> None:
    events: list[ev.Event] = []
    ticket = Submit(
        rid="r1-9",
        prompt=(1,),
        sampling=GREEDY,
        max_tokens=1,
        stop=(),
        trace="00-abc-def-01",
        tenant="acme",
    )
    req = Request(
        ticket,
        seq=0,
        stop=StopCheck(FakeDetokenizer(), frozenset(), 1, ()),
        emit=events.append,
        t=0.0,
    )
    assert (req.rid, req.prompt, req.sampling, req.trace, req.tenant) == (
        "r1-9",
        (1,),
        GREEDY,
        "00-abc-def-01",
        "acme",
    )
    assert events == [
        ev.ReqArrived(t=0.0, rid="r1-9", prompt_tokens=1, max_tokens=1, tenant="acme")
    ]
