"""E6 at ``max_num_seqs = 1``: FCFS admission, one prefill per step, then decode."""

import pytest

from ullm.core.engine import Submit
from ullm.core.request import Request, RequestState
from ullm.core.scheduler import Scheduler
from ullm.core.stop import StopCheck
from ullm.runner.base import DecodePlan, PrefillPlan, SamplingParams
from ullm.runner.fake_runner import FakeDetokenizer
from ullm.telemetry import events as ev

GREEDY = SamplingParams(temperature=0.0, top_p=1.0, top_k=0, seed=None)


def _request(seq: int, events: list[ev.Event] | None = None) -> Request:
    ticket = Submit(rid=f"r1-{seq}", prompt=(seq, 1), sampling=GREEDY, max_tokens=8, stop=())
    stop = StopCheck(FakeDetokenizer(), frozenset(), 8, ())
    sink = events if events is not None else []
    return Request(ticket, seq=seq, stop=stop, emit=sink.append, t=0.0)


def test_nothing_to_do_gives_none() -> None:
    assert Scheduler(max_num_seqs=1).schedule(0.0) is None


def test_the_oldest_waiting_request_is_admitted_and_prefilled_first() -> None:
    scheduler = Scheduler(max_num_seqs=1)
    a, b = _request(0), _request(1)
    scheduler.add(a)
    scheduler.add(b)
    step = scheduler.schedule(1.0)
    assert step is not None
    assert step.admitted is a
    assert step.rows == (a,)
    assert step.plan == PrefillPlan(seq=0, token_ids=(0, 1), start=0)
    assert a.state is RequestState.PREFILLING
    assert b.state is RequestState.WAITING
    assert (scheduler.waiting, scheduler.running) == (1, 1)


def test_a_running_request_decodes_while_the_next_waits_for_room() -> None:
    scheduler = Scheduler(max_num_seqs=1)
    a, b = _request(0), _request(1)
    scheduler.add(a)
    scheduler.add(b)
    scheduler.schedule(1.0)
    a.append(5, 2.0)
    step = scheduler.schedule(3.0)
    assert step is not None
    assert step.admitted is None
    assert step.rows == (a,)
    assert step.plan == DecodePlan(seqs=(0,), token_ids=(5,))
    assert b.state is RequestState.WAITING


def test_removing_the_running_request_admits_the_next() -> None:
    scheduler = Scheduler(max_num_seqs=1)
    a, b = _request(0), _request(1)
    scheduler.add(a)
    scheduler.add(b)
    scheduler.schedule(1.0)
    scheduler.remove(a)
    step = scheduler.schedule(2.0)
    assert step is not None
    assert step.admitted is b
    assert (scheduler.waiting, scheduler.running) == (0, 1)


def test_a_removed_waiting_request_is_never_admitted() -> None:
    scheduler = Scheduler(max_num_seqs=1)
    a, b, c = _request(0), _request(1), _request(2)
    for req in (a, b, c):
        scheduler.add(req)
    scheduler.remove(b)
    admitted = []
    while (step := scheduler.schedule(0.0)) is not None:
        assert step.admitted is not None
        admitted.append(step.admitted)
        scheduler.remove(step.admitted)
    assert admitted == [a, c]


def test_removing_an_unknown_request_raises() -> None:
    with pytest.raises(ValueError, match="r1-0"):
        Scheduler(max_num_seqs=1).remove(_request(0))


def test_admission_is_first_come_first_served() -> None:
    scheduler = Scheduler(max_num_seqs=1)
    reqs = [_request(i) for i in range(5)]
    for req in reqs:
        scheduler.add(req)
    order = []
    while (step := scheduler.schedule(0.0)) is not None:
        assert step.admitted is not None
        order.append(step.admitted.seq)
        scheduler.remove(step.admitted)
    assert order == [0, 1, 2, 3, 4]
