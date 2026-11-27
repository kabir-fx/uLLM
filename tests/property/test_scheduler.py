"""Q2 for E6 (decision 0020): over random arrivals, aborts and limits, the engine on
``FakeRunner`` ends every request, admits first come first served within ``max_num_seqs``
and the padded KV slot budget, and leaks nothing."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from queue import SimpleQueue

from hypothesis import given, settings
from hypothesis import strategies as st

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
)
from ullm.runner.base import (
    BlockId,
    DecodePlan,
    PrefillPlan,
    RunnerInfo,
    SamplingParams,
    SeqId,
    StepPlan,
    StepResult,
)
from ullm.runner.fake_runner import FakeDetokenizer, FakeRunner, Latency
from ullm.telemetry import events as ev
from ullm.telemetry.check import check_trace
from ullm.telemetry.trace import frames_from

INFO = RunnerInfo(model_id="fake", vocab_size=50, eos_token_ids=frozenset({0}), max_model_len=256)
FAST = Latency(
    prefill_base_ms=1.0, prefill_per_token_ms=0.1, decode_base_ms=2.0, decode_per_row_ms=0.5
)
GREEDY = SamplingParams(temperature=0.0, top_p=1.0, top_k=0, seed=None)


@dataclass(frozen=True)
class New:
    prompt_len: int
    max_tokens: int


@dataclass(frozen=True)
class Cancel:
    pick: int  # of the requests submitted so far, modulo their number


Action = New | Cancel


@st.composite
def _new(draw: st.DrawFn) -> New:
    prompt_len = draw(st.integers(1, 192))
    return New(prompt_len, draw(st.integers(1, min(64, INFO.max_model_len - prompt_len))))


ACTIONS = st.one_of(_new(), _new(), st.builds(Cancel, st.integers(0, 1 << 16)))
TURNS = st.lists(st.lists(ACTIONS, max_size=4), min_size=1, max_size=24)


@dataclass
class Checked:
    """``FakeRunner``, checking the limits on every step. Seq ``i`` is the ``i``-th arrival,
    because the engine numbers requests as they arrive."""

    config: EngineConfig
    arrivals: list[ev.ReqArrived]
    inner: FakeRunner = field(default_factory=lambda: FakeRunner(INFO, FAST, lambda s: None))
    written: dict[SeqId, int] = field(default_factory=dict)  # KV slots per held sequence
    tables: dict[SeqId, tuple[BlockId, ...]] = field(default_factory=dict)  # the last planned
    adds: int = 0
    releases: int = 0
    violations: list[str] = field(default_factory=list)

    def info(self) -> RunnerInfo:
        return INFO

    def add(self, seq: SeqId, sampling: SamplingParams) -> None:
        self.inner.add(seq, sampling)
        self.adds += 1
        self.written[seq] = 0

    def release(self, seq: SeqId, freed: tuple[BlockId, ...]) -> None:
        self.inner.release(seq, freed)
        self.releases += 1
        del self.written[seq]
        table = self.tables.pop(seq, ())
        if sorted(freed) != sorted(table):
            self.violations.append(f"seq {seq} freed {freed}, its table was {table}")

    def execute(self, plan: StepPlan) -> StepResult:
        """Records a broken limit instead of raising: the engine fails a step that raises."""
        rows = len(self.written)
        reservation = max(
            self.arrivals[s].prompt_tokens + self.arrivals[s].max_tokens for s in self.written
        )
        if rows > self.config.max_num_seqs:
            self.violations.append(f"{rows} rows held, max_num_seqs {self.config.max_num_seqs}")
        if rows * reservation > self.config.max_kv_slots:
            self.violations.append(f"{rows} rows x reservation {reservation} reserved")
        match plan:
            case PrefillPlan(seq=seq, token_ids=token_ids, blocks=blocks):
                self.written[seq] = len(token_ids)
                self.tables[seq] = blocks
            case DecodePlan(seqs=seqs, blocks=blocks):
                self.tables.update(zip(seqs, blocks, strict=True))
                if sorted(seqs) != sorted(self.written):
                    self.violations.append(f"decode of {seqs}, held {sorted(self.written)}")
                for s in seqs:
                    self.written[s] += 1
        held = [b for table in self.tables.values() for b in table]
        if len(held) != len(set(held)):
            self.violations.append(f"two held sequences share a block: {self.tables}")
        if rows * max(self.written.values()) > self.config.max_kv_slots:
            self.violations.append(f"{rows} rows x {max(self.written.values())} slots written")
        return self.inner.execute(plan)


@dataclass
class Script:
    """Feeds one turn of tickets per engine loop turn, then shuts down once all have ended."""

    turns: list[list[Action]]
    submitted: list[Submit] = field(default_factory=list)
    cancelled: set[str] = field(default_factory=set)
    outputs: dict[str, list[Output]] = field(default_factory=dict)

    def feed(self, inbound: "SimpleQueue[Ticket]") -> None:
        while self.turns:
            tickets = [t for a in self.turns.pop(0) if (t := self._ticket(a)) is not None]
            for ticket in tickets:
                inbound.put(ticket)
            if tickets or not self._idle():
                return
        if self._idle():
            inbound.put(Shutdown())

    def deliver(
        self, batch: Sequence[tuple[str, Output]], load: Load, inbound: "SimpleQueue[Ticket]"
    ) -> None:
        for rid, out in batch:
            self.outputs.setdefault(rid, []).append(out)
        self.feed(inbound)

    def _ticket(self, action: Action) -> Ticket | None:
        match action:
            case New(prompt_len=n, max_tokens=m):
                submit = Submit(
                    rid=f"r1-{len(self.submitted)}",
                    prompt=tuple(1 + i % (INFO.vocab_size - 1) for i in range(n)),
                    sampling=GREEDY,
                    max_tokens=m,
                    stop=(),
                )
                self.submitted.append(submit)
                return submit
            case Cancel(pick=pick):
                if not self.submitted:
                    return None
                rid = self.submitted[pick % len(self.submitted)].rid
                self.cancelled.add(rid)
                return Abort(rid)

    def _idle(self) -> bool:
        ended = sum(isinstance(o[-1], Finished | Failed) for o in self.outputs.values())
        return ended == len(self.submitted)


def _admitted_first_come_first_served(events: list[ev.Event]) -> None:
    """At each admission, every earlier arrival is already admitted or has ended."""
    order: list[str] = []
    settled: set[str] = set()
    for e in events:
        match e:
            case ev.ReqArrived(rid=rid):
                order.append(rid)
            case ev.ReqAdmitted(rid=rid):
                earlier = order[: order.index(rid)]
                assert settled.issuperset(earlier), f"{rid} skipped ahead"
                settled.add(rid)
            case ev.ReqAborted(rid=rid) | ev.ReqFinished(rid=rid):
                settled.add(rid)


@settings(deadline=None)
@given(
    max_num_seqs=st.integers(1, 8),
    max_kv_slots=st.integers(INFO.max_model_len, 4 * INFO.max_model_len),
    turns=TURNS,
)
def test_every_request_ends_first_come_first_served_within_the_limits(
    max_num_seqs: int, max_kv_slots: int, turns: list[list[Action]]
) -> None:
    config = EngineConfig(max_num_seqs=max_num_seqs, max_kv_slots=max_kv_slots)
    events: list[ev.Event] = []
    arrivals: list[ev.ReqArrived] = []

    def emit(e: ev.Event) -> None:
        events.append(e)
        if isinstance(e, ev.ReqArrived):
            arrivals.append(e)

    runner = Checked(config, arrivals)
    script = Script(turns)
    inbound: SimpleQueue[Ticket] = SimpleQueue()
    script.feed(inbound)
    loads: list[Load] = []

    def deliver(batch: Sequence[tuple[str, Output]], load: Load) -> None:
        loads.append(load)
        script.deliver(batch, load, inbound)

    Engine(runner, FakeDetokenizer, config, inbound, deliver, emit).run()

    assert runner.violations == []
    for submit in script.submitted:
        *deltas, last = script.outputs[submit.rid]
        assert not any(isinstance(o, Finished | Failed) for o in deltas)
        if submit.rid in script.cancelled:
            assert isinstance(last, Finished) or last == Failed("client_disconnect")
        else:
            assert isinstance(last, Finished)
            assert last.completion_tokens == submit.max_tokens  # FakeRunner never samples EOS
    _admitted_first_come_first_served(events)
    assert runner.adds == runner.releases
    assert runner.written == {}
    assert loads[-1] == Load(waiting=0, running=0)
    assert check_trace(frames_from(events, "r1")).problems == []
