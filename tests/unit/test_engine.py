"""E5: the engine loop, run on the test thread with tickets queued in advance."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from queue import SimpleQueue
from typing import Any

import pytest
from tests.recording import RecordingRunner, tokens_by_rid

from ullm.core.engine import (
    Abort,
    Delta,
    Engine,
    EngineConfig,
    Failed,
    Finished,
    Load,
    Output,
    Shutdown,
    Submit,
    Ticket,
)
from ullm.runner.base import Detokenizer, Runner, RunnerInfo, SamplingParams
from ullm.runner.fake_runner import FakeDetokenizer, FakeRunner, Latency
from ullm.telemetry import events as ev
from ullm.telemetry.check import TraceReport, check_trace
from ullm.telemetry.trace import frames_from

GREEDY = SamplingParams(temperature=0.0, top_p=1.0, top_k=0, seed=None)


def _submit(**changes: Any) -> Submit:
    fields: dict[str, Any] = {
        "rid": "r1-0",
        "prompt": (1, 2),
        "sampling": GREEDY,
        "max_tokens": 4,
        "stop": (),
    }
    return Submit(**(fields | changes))


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"rid": ""}, "rid"),
        ({"rid": 3}, "rid"),
        ({"prompt": ()}, "prompt"),
        ({"prompt": [1, 2]}, "prompt"),
        ({"prompt": (1, True)}, "prompt"),
        ({"prompt": (1, -1)}, "prompt"),
        ({"sampling": {"temperature": 0.0}}, "sampling"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"max_tokens": 2.0}, "max_tokens"),
        ({"stop": ["x"]}, "stop"),
        ({"stop": ("",)}, "stop"),
        ({"trace": 1}, "trace"),
        ({"tenant": b"acme"}, "tenant"),
    ],
    ids=repr,
)
def test_a_malformed_submit_is_refused(changes: dict[str, Any], field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _submit(**changes)


def test_a_well_formed_submit_is_built() -> None:
    assert _submit(trace="00-a-b-01", tenant="acme").prompt == (1, 2)


# --- the loop ---------------------------------------------------------------------

INFO = RunnerInfo(model_id="fake", vocab_size=50, eos_token_ids=frozenset({0}), max_model_len=256)
FAST = Latency(
    prefill_base_ms=1.0, prefill_per_token_ms=0.1, decode_base_ms=2.0, decode_per_row_ms=0.5
)
OnDeliver = Callable[[Sequence[tuple[str, Output]], "SimpleQueue[Ticket]"], None]


@dataclass
class Ran:
    runner: RecordingRunner
    events: list[ev.Event]
    outputs: dict[str, list[Output]]
    delivers: list[tuple[tuple[str, Output], ...]]

    def report(self) -> TraceReport:
        return check_trace(frames_from(self.events, "r1"))

    def text(self, rid: str) -> str:
        return "".join(o.text for o in self.outputs[rid] if isinstance(o, Delta | Finished))


def _shutdown_when_all_end(submits: int) -> OnDeliver:
    ended = 0

    def on_deliver(batch: Sequence[tuple[str, Output]], inbound: "SimpleQueue[Ticket]") -> None:
        nonlocal ended
        ended += sum(isinstance(o, Finished | Failed) for _, o in batch)
        if ended == submits:
            inbound.put(Shutdown())

    return on_deliver


def _run(
    tickets: list[Ticket],
    on_deliver: OnDeliver | None = None,
    runner: Runner | None = None,
    new_detokenizer: Callable[[], Detokenizer] = FakeDetokenizer,
    inbound: "SimpleQueue[Ticket] | None" = None,
    config: EngineConfig | None = None,
) -> Ran:
    recording = RecordingRunner(runner or FakeRunner(INFO, FAST, sleep=lambda s: None))
    inbound = inbound if inbound is not None else SimpleQueue()
    for ticket in tickets:
        inbound.put(ticket)
    hook = on_deliver or _shutdown_when_all_end(sum(isinstance(t, Submit) for t in tickets))
    ran = Ran(recording, [], {}, [])

    def deliver(batch: Sequence[tuple[str, Output]], load: Load) -> None:
        ran.delivers.append(tuple(batch))
        for rid, out in batch:
            ran.outputs.setdefault(rid, []).append(out)
        hook(batch, inbound)

    engine = Engine(
        recording, new_detokenizer, config or EngineConfig(), inbound, deliver, ran.events.append
    )
    engine.run()
    return ran


def _ends_once(outputs: list[Output]) -> Output:
    """The one terminal output, last, after only deltas."""
    *deltas, last = outputs
    assert all(isinstance(o, Delta) and o.text for o in deltas)
    assert isinstance(last, Finished | Failed)
    return last


def test_every_submit_finishes_with_its_own_tokens() -> None:
    tickets: list[Ticket] = [
        _submit(rid=f"r1-{i}", max_tokens=1 + 3 * i, sampling=_seeded(i)) for i in range(4)
    ]
    ran = _run(tickets)
    sampled = tokens_by_rid(ran.events, ran.runner.steps)
    for i in range(4):
        rid = f"r1-{i}"
        last = _ends_once(ran.outputs[rid])
        assert last == Finished(
            text=f" t{sampled[rid][-1]}",
            reason="length",
            prompt_tokens=2,
            completion_tokens=1 + 3 * i,
        )
        assert ran.text(rid) == "".join(f" t{t}" for t in sampled[rid])
        assert len(sampled[rid]) == 1 + 3 * i
    assert ran.report().problems == []
    assert ran.runner.adds == ran.runner.releases == 4
    assert ran.runner.live == set()


def test_step_events_count_the_kv_after_each_step() -> None:
    ran = _run([_submit(prompt=(1, 2, 3), max_tokens=4)])
    steps = [e for e in ran.events if isinstance(e, ev.Step)]
    assert [(e.kind, e.tokens, e.kv_tokens) for e in steps] == [
        ("prefill", 3, 3),
        ("decode", 1, 4),
        ("decode", 1, 5),
        ("decode", 1, 6),
    ]


class _OddTokensOnly:
    """Holds back every even token, as a detokenizer holds an unfinished character."""

    def add(self, token: int) -> str:
        return f" t{token}" if token % 2 else ""

    def flush(self) -> str:
        return ""


def test_a_chunk_with_no_text_sends_no_delta() -> None:
    ran = _run([_submit(max_tokens=30)], new_detokenizer=_OddTokensOnly)
    sampled = tokens_by_rid(ran.events, ran.runner.steps)["r1-0"]
    assert any(t % 2 == 0 for t in sampled[:-1])
    _ends_once(ran.outputs["r1-0"])
    assert ran.text("r1-0") == "".join(f" t{t}" for t in sampled if t % 2)


class _CountingQueue(SimpleQueue[Ticket]):
    blocking_gets = 0

    def get(self, block: bool = True, timeout: float | None = None) -> Ticket:
        if block:
            self.blocking_gets += 1
        return super().get(block, timeout)


def test_the_loop_waits_on_the_queue_only_when_idle() -> None:
    inbound = _CountingQueue()
    _run([_submit(max_tokens=5)], inbound=inbound)
    # Once at the start, once after the reply ended: never while it ran, and never a spin.
    assert inbound.blocking_gets == 2


def test_the_trace_starts_with_hello() -> None:
    ran = _run([_submit()])
    hello = ran.events[0]
    assert isinstance(hello, ev.Hello)
    assert (hello.replica, hello.source, hello.model) == ("r1", "engine", "fake")
    assert hello.limits == ev.Limits(max_num_seqs=16, max_prefill_tokens=256, max_model_len=256)
    assert hello.kv == ev.KvInfo(capacity_tokens=40_960)


def test_an_abort_of_a_waiting_request_fails_it_without_touching_the_runner() -> None:
    ran = _run([_submit(rid="r1-0"), _submit(rid="r1-1"), Abort("r1-1")])
    assert ran.outputs["r1-1"] == [Failed("client_disconnect")]
    assert isinstance(_ends_once(ran.outputs["r1-0"]), Finished)
    assert ran.runner.adds == ran.runner.releases == 1
    assert ran.report().problems == []


def test_an_abort_of_a_running_request_releases_it_and_ends_its_stream() -> None:
    def abort_after_first_delta(
        batch: Sequence[tuple[str, Output]], inbound: "SimpleQueue[Ticket]"
    ) -> None:
        for rid, out in batch:
            if isinstance(out, Delta):
                inbound.put(Abort(rid))
            if isinstance(out, Failed):
                inbound.put(Shutdown())

    ran = _run([_submit(max_tokens=50)], on_deliver=abort_after_first_delta)
    assert _ends_once(ran.outputs["r1-0"]) == Failed("client_disconnect")
    assert ran.runner.adds == ran.runner.releases == 1
    assert ran.report().problems == []


def test_an_abort_for_an_ended_request_is_ignored() -> None:
    def abort_after_end(
        batch: Sequence[tuple[str, Output]], inbound: "SimpleQueue[Ticket]"
    ) -> None:
        if any(isinstance(o, Finished) for _, o in batch):
            inbound.put(Abort("r1-0"))
            inbound.put(Shutdown())

    ran = _run([_submit()], on_deliver=abort_after_end)
    assert isinstance(_ends_once(ran.outputs["r1-0"]), Finished)
    assert ran.report().problems == []


def test_shutdown_fails_every_live_request() -> None:
    ran = _run([_submit(rid="r1-0"), _submit(rid="r1-1"), Shutdown()], on_deliver=lambda b, q: None)
    assert ran.outputs == {
        "r1-0": [Failed("shutdown")],
        "r1-1": [Failed("shutdown")],
    }
    assert ran.runner.adds == ran.runner.releases
    assert ran.report().problems == []


def test_a_runner_failure_fails_only_the_requests_in_that_step() -> None:
    calls = 0

    def fail_third_step(seconds: float) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("the GPU fell over")

    runner = FakeRunner(INFO, FAST, sleep=fail_third_step)
    submits: list[Ticket] = [_submit(rid="r1-0", max_tokens=5), _submit(rid="r1-1")]
    # At batch size 1 the third step is r1-0's second decode while r1-1 waits.
    ran = _run(submits, runner=runner, config=EngineConfig(max_num_seqs=1))
    assert _ends_once(ran.outputs["r1-0"]) == Failed("error")
    assert isinstance(_ends_once(ran.outputs["r1-1"]), Finished)
    assert ran.runner.adds == ran.runner.releases == 2
    assert ran.report().problems == []


class _AddFails(RecordingRunner):
    def add(self, seq: int, sampling: SamplingParams) -> None:
        raise RuntimeError("a bug")


def test_a_bug_in_the_loop_aborts_every_live_request_in_the_trace_and_raises() -> None:
    events: list[ev.Event] = []
    inbound: SimpleQueue[Ticket] = SimpleQueue()
    inbound.put(_submit(rid="r1-0"))
    inbound.put(_submit(rid="r1-1"))
    engine = Engine(
        _AddFails(FakeRunner(INFO, FAST, sleep=lambda s: None)),
        FakeDetokenizer,
        EngineConfig(),
        inbound,
        lambda batch, load: None,
        events.append,
    )
    with pytest.raises(RuntimeError, match="a bug"):
        engine.run()
    aborted = [e for e in events if isinstance(e, ev.ReqAborted)]
    assert sorted((e.rid, e.reason) for e in aborted) == [("r1-0", "error"), ("r1-1", "error")]
    assert check_trace(frames_from(events, "r1")).problems == []


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"max_num_seqs": 0}, "max_num_seqs"),
        ({"max_num_seqs": True}, "max_num_seqs"),
        ({"max_num_seqs": 2.0}, "max_num_seqs"),
        ({"max_kv_slots": 0}, "max_kv_slots"),
        ({"max_kv_slots": True}, "max_kv_slots"),
    ],
    ids=repr,
)
def test_the_config_refuses_a_malformed_limit(changes: dict[str, Any], field: str) -> None:
    with pytest.raises(ValueError, match=field):
        EngineConfig(**changes)


def test_the_engine_refuses_a_slot_budget_that_a_legal_request_could_never_fit() -> None:
    with pytest.raises(ValueError, match="max_kv_slots"):
        Engine(
            FakeRunner(INFO, FAST, sleep=lambda s: None),
            FakeDetokenizer,
            EngineConfig(max_kv_slots=INFO.max_model_len - 1),
            SimpleQueue(),
            lambda batch, load: None,
            lambda event: None,
        )


def _decode_rows(ran: Ran) -> list[int]:
    return [len(s.rids) for s in ran.events if isinstance(s, ev.Step) and s.kind == "decode"]


def test_concurrent_requests_decode_in_one_batch_and_reply_as_they_would_alone() -> None:
    submits = [_submit(rid=f"r1-{i}", prompt=(i + 1, 2), max_tokens=6) for i in range(4)]
    batched = _run(list(submits), config=EngineConfig(max_num_seqs=4))
    alone = _run(list(submits), config=EngineConfig(max_num_seqs=1))
    assert max(_decode_rows(batched)) == 4
    assert max(_decode_rows(alone)) == 1
    for i in range(4):
        assert isinstance(_ends_once(batched.outputs[f"r1-{i}"]), Finished)
        assert batched.text(f"r1-{i}") == alone.text(f"r1-{i}")
    assert batched.runner.adds == batched.runner.releases == 4
    assert batched.report().problems == []


def test_a_batch_never_grows_past_max_num_seqs() -> None:
    submits: list[Ticket] = [_submit(rid=f"r1-{i}", max_tokens=8) for i in range(5)]
    ran = _run(submits, config=EngineConfig(max_num_seqs=2))
    assert max(_decode_rows(ran)) == 2
    assert all(isinstance(_ends_once(ran.outputs[f"r1-{i}"]), Finished) for i in range(5))
    assert ran.report().problems == []


def test_a_failed_batched_decode_fails_every_row_and_the_waiting_request_still_finishes() -> None:
    calls = 0

    def fail_the_first_decode(seconds: float) -> None:
        nonlocal calls
        calls += 1
        if calls == 4:  # three prefills, one per step, then the decode of all three
            raise RuntimeError("the GPU fell over")

    runner = FakeRunner(INFO, FAST, sleep=fail_the_first_decode)
    submits: list[Ticket] = [_submit(rid=f"r1-{i}", max_tokens=8) for i in range(4)]
    ran = _run(submits, runner=runner, config=EngineConfig(max_num_seqs=3))
    for i in range(3):
        assert ran.outputs[f"r1-{i}"][-1] == Failed("error")
    assert isinstance(_ends_once(ran.outputs["r1-3"]), Finished)
    assert ran.runner.adds == ran.runner.releases == 4
    assert ran.report().problems == []


def test_the_config_refuses_a_bound_below_one() -> None:
    with pytest.raises(ValueError, match="max_in_flight"):
        EngineConfig(max_in_flight=0)


def _seeded(seed: int) -> SamplingParams:
    return SamplingParams(temperature=0.0, top_p=1.0, top_k=0, seed=seed)


def test_a_failed_prefill_releases_its_sequence_and_fails_only_that_request() -> None:
    calls = 0

    def fail_first_step(seconds: float) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("the GPU fell over")

    runner = FakeRunner(INFO, FAST, sleep=fail_first_step)
    ran = _run([_submit(rid="r1-0"), _submit(rid="r1-1")], runner=runner)
    assert ran.outputs["r1-0"] == [Failed("error")]
    assert isinstance(_ends_once(ran.outputs["r1-1"]), Finished)
    assert ran.runner.adds == ran.runner.releases == 2
    assert ran.report().problems == []


class _ReleaseFails(RecordingRunner):
    def release(self, seq: int) -> None:
        raise RuntimeError("a bug in release")


def test_a_bug_after_a_reply_finishes_still_raises_the_bug_and_ends_the_others() -> None:
    events: list[ev.Event] = []
    inbound: SimpleQueue[Ticket] = SimpleQueue()
    inbound.put(_submit(rid="r1-0", max_tokens=1))
    inbound.put(_submit(rid="r1-1"))
    engine = Engine(
        _ReleaseFails(FakeRunner(INFO, FAST, sleep=lambda s: None)),
        FakeDetokenizer,
        EngineConfig(),
        inbound,
        lambda batch, load: None,
        events.append,
    )
    with pytest.raises(RuntimeError, match="a bug in release"):
        engine.run()
    aborted = [(e.rid, e.reason) for e in events if isinstance(e, ev.ReqAborted)]
    assert aborted == [("r1-1", "error")]
    assert check_trace(frames_from(events, "r1")).problems == []


def test_a_duplicate_rid_is_a_bug_that_stops_the_engine_instead_of_hanging_it() -> None:
    ran_events: list[ev.Event] = []
    inbound: SimpleQueue[Ticket] = SimpleQueue()
    inbound.put(_submit(rid="r1-0"))
    inbound.put(_submit(rid="r1-0"))
    engine = Engine(
        RecordingRunner(FakeRunner(INFO, FAST, sleep=lambda s: None)),
        FakeDetokenizer,
        EngineConfig(),
        inbound,
        lambda batch, load: None,
        ran_events.append,
    )
    with pytest.raises(ValueError, match="r1-0"):
        engine.run()
    assert check_trace(frames_from(ran_events, "r1")).problems == []
