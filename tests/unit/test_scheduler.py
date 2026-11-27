"""E6 (decisions 0010 and 0020): FCFS admission within ``max_num_seqs`` and the padded KV
slot budget, one prefill per step, else one decode of every running request."""

import pytest

from ullm.core.engine import Submit
from ullm.core.request import Request, RequestState
from ullm.core.scheduler import Scheduler
from ullm.core.stop import StopCheck
from ullm.runner.base import DecodePlan, PrefillPlan, SamplingParams
from ullm.runner.fake_runner import FakeDetokenizer
from ullm.telemetry import events as ev

GREEDY = SamplingParams(temperature=0.0, top_p=1.0, top_k=0, seed=None)


BIG = 1 << 20  # a slot budget that never binds


def _scheduler(max_num_seqs: int = 1, max_kv_slots: int = BIG) -> Scheduler:
    return Scheduler(max_num_seqs=max_num_seqs, max_kv_slots=max_kv_slots, block_size=4)


def _request(
    seq: int, events: list[ev.Event] | None = None, prompt_len: int = 2, max_tokens: int = 8
) -> Request:
    """Reserves ``prompt_len + max_tokens`` slots."""
    prompt = (seq, *([1] * (prompt_len - 1)))
    ticket = Submit(rid=f"r1-{seq}", prompt=prompt, sampling=GREEDY, max_tokens=max_tokens, stop=())
    stop = StopCheck(FakeDetokenizer(), frozenset(), max_tokens, ())
    sink = events if events is not None else []
    return Request(ticket, seq=seq, stop=stop, emit=sink.append, t=0.0)


def test_nothing_to_do_gives_none() -> None:
    assert _scheduler().schedule(0.0) is None


def test_the_oldest_waiting_request_is_admitted_and_prefilled_first() -> None:
    scheduler = _scheduler()
    a, b = _request(0), _request(1)
    scheduler.add(a)
    scheduler.add(b)
    step = scheduler.schedule(1.0)
    assert step is not None
    assert step.admitted is a
    assert step.rows == (a,)
    assert step.plan == PrefillPlan(seq=0, token_ids=(0, 1), start=0, blocks=(0,))
    assert a.state is RequestState.PREFILLING
    assert b.state is RequestState.WAITING
    assert (scheduler.waiting, scheduler.running) == (1, 1)


def test_a_running_request_decodes_while_the_next_waits_for_room() -> None:
    scheduler = _scheduler()
    a, b = _request(0), _request(1)
    scheduler.add(a)
    scheduler.add(b)
    scheduler.schedule(1.0)
    a.append(5, 2.0)
    step = scheduler.schedule(3.0)
    assert step is not None
    assert step.admitted is None
    assert step.rows == (a,)
    assert step.plan == DecodePlan(seqs=(0,), token_ids=(5,), blocks=((0,),))
    assert b.state is RequestState.WAITING


def test_removing_the_running_request_admits_the_next() -> None:
    scheduler = _scheduler()
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
    scheduler = _scheduler()
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
        _scheduler().remove(_request(0))


def test_admission_is_first_come_first_served() -> None:
    scheduler = _scheduler()
    reqs = [_request(i) for i in range(5)]
    for req in reqs:
        scheduler.add(req)
    order = []
    while (step := scheduler.schedule(0.0)) is not None:
        assert step.admitted is not None
        order.append(step.admitted.seq)
        scheduler.remove(step.admitted)
    assert order == [0, 1, 2, 3, 4]


def _admit_and_prefill(scheduler: Scheduler, token: int) -> Request:
    step = scheduler.schedule(0.0)
    assert step is not None
    assert step.admitted is not None
    step.admitted.append(token, 0.0)
    return step.admitted


def test_requests_join_one_per_step_up_to_max_num_seqs_then_decode_together() -> None:
    scheduler = _scheduler(max_num_seqs=2)
    a, b, c = _request(0), _request(1), _request(2)
    for req in (a, b, c):
        scheduler.add(req)
    assert _admit_and_prefill(scheduler, 5) is a
    assert _admit_and_prefill(scheduler, 6) is b
    step = scheduler.schedule(0.0)
    assert step is not None
    assert step.admitted is None
    assert step.rows == (a, b)
    assert step.plan == DecodePlan(seqs=(0, 1), token_ids=(5, 6), blocks=((0,), (1,)))
    assert c.state is RequestState.WAITING
    assert (scheduler.waiting, scheduler.running) == (1, 2)


def test_a_new_request_prefills_alone_while_the_running_ones_wait_a_step() -> None:
    scheduler = _scheduler(max_num_seqs=4)
    a, b = _request(0), _request(1)
    scheduler.add(a)
    _admit_and_prefill(scheduler, 5)
    scheduler.add(b)
    step = scheduler.schedule(0.0)
    assert step is not None
    assert step.admitted is b
    assert step.rows == (b,)
    assert step.plan == PrefillPlan(seq=1, token_ids=(1, 1), start=0, blocks=(1,))


def test_the_head_waits_while_its_reservation_would_overfill_the_batch() -> None:
    scheduler = _scheduler(max_num_seqs=4, max_kv_slots=39)
    a = _request(0, prompt_len=2, max_tokens=8)  # reserves 10
    b = _request(1, prompt_len=4, max_tokens=16)  # reserves 20: 2 rows x 20 > 39
    scheduler.add(a)
    scheduler.add(b)
    _admit_and_prefill(scheduler, 5)
    step = scheduler.schedule(0.0)
    assert step is not None
    assert step.admitted is None
    assert step.rows == (a,)
    scheduler.remove(a)
    step = scheduler.schedule(0.0)
    assert step is not None
    assert step.admitted is b


def test_a_batch_that_fills_the_budget_exactly_admits() -> None:
    scheduler = _scheduler(max_num_seqs=4, max_kv_slots=40)
    scheduler.add(_request(0, prompt_len=2, max_tokens=8))
    scheduler.add(_request(1, prompt_len=4, max_tokens=16))
    _admit_and_prefill(scheduler, 5)
    step = scheduler.schedule(0.0)
    assert step is not None
    assert step.admitted is not None
    assert step.admitted.seq == 1


def test_a_short_request_counts_at_the_longest_reservation_in_the_batch() -> None:
    scheduler = _scheduler(max_num_seqs=4, max_kv_slots=39)
    long = _request(0, prompt_len=4, max_tokens=16)  # reserves 20
    short = _request(1, prompt_len=1, max_tokens=1)  # reserves 2, padded to 20
    scheduler.add(long)
    scheduler.add(short)
    _admit_and_prefill(scheduler, 5)
    step = scheduler.schedule(0.0)
    assert step is not None
    assert step.admitted is None
    assert short.state is RequestState.WAITING


def test_no_request_skips_ahead_of_a_head_that_does_not_fit() -> None:
    scheduler = _scheduler(max_num_seqs=4, max_kv_slots=40)
    a = _request(0, prompt_len=2, max_tokens=8)  # reserves 10
    head = _request(1, prompt_len=10, max_tokens=20)  # reserves 30: 2 x 30 > 40
    behind = _request(2, prompt_len=1, max_tokens=1)  # reserves 2: 2 x 10 would fit
    for req in (a, head, behind):
        scheduler.add(req)
    _admit_and_prefill(scheduler, 5)
    step = scheduler.schedule(0.0)
    assert step is not None
    assert step.admitted is None
    assert (head.state, behind.state) == (RequestState.WAITING, RequestState.WAITING)


# Block tables (decision 0034)


def test_each_plan_carries_the_block_table_after_its_step() -> None:
    scheduler = _scheduler(max_num_seqs=2)
    a = _request(0, prompt_len=7)
    scheduler.add(a)
    assert _admit_and_prefill(scheduler, 5) is a
    step = scheduler.schedule(0.0)  # writes token 8 of 8: still two blocks of 4
    assert step is not None
    assert step.plan == DecodePlan(seqs=(0,), token_ids=(5,), blocks=((0, 1),))
    a.append(6, 0.0)
    step = scheduler.schedule(0.0)  # token 9 takes a third block
    assert step is not None
    assert step.plan == DecodePlan(seqs=(0,), token_ids=(6,), blocks=((0, 1, 2),))


def test_removing_a_running_request_frees_its_blocks_and_a_waiting_one_has_none() -> None:
    scheduler = _scheduler()
    a, b = _request(0, prompt_len=5), _request(1)
    scheduler.add(a)
    scheduler.add(b)
    _admit_and_prefill(scheduler, 5)
    assert scheduler.remove(b) == ()
    assert sorted(scheduler.remove(a)) == [0, 1]
