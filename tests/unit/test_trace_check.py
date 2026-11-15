"""The trace checker must catch broken stories, or the fixtures prove nothing."""

from ullm.telemetry import events as ev
from ullm.telemetry.check import check_trace

HELLO = ev.Hello(
    t=0,
    replica="r1",
    model="m",
    source="fixture",
    version="0",
    commit=None,
    epoch_unix_ms=0,
    limits=ev.Limits(max_num_seqs=4, max_prefill_tokens=64, max_model_len=128),
    kv=ev.KvInfo(),
    machine=None,
)


def arrived(t: float, rid: str = "a") -> ev.ReqArrived:
    return ev.ReqArrived(t=t, rid=rid, prompt_tokens=8, max_tokens=4)


def admitted(t: float, rid: str = "a") -> ev.ReqAdmitted:
    return ev.ReqAdmitted(t=t, rid=rid, queue_ms=0)


def step(t: float, kind: str, *rids: str) -> ev.Step:
    return ev.Step(
        t=t,
        n=0,
        kind=kind,  # type: ignore[arg-type]
        rids=list(rids),
        tokens=8 if kind == "prefill" else len(rids),
        ms=1,
        waiting=0,
        running=len(rids),
        kv_tokens=8,
    )


# Every request in these stories arrives at t=1.
def first(t: float, rid: str = "a") -> ev.ReqFirstToken:
    return ev.ReqFirstToken(t=t, rid=rid, ttft_ms=t - 1)


def finished(t: float, tokens: int, rid: str = "a") -> ev.ReqFinished:
    return ev.ReqFinished(
        t=t,
        rid=rid,
        reason="stop",
        usage=ev.TokenUsage(prompt_tokens=8, completion_tokens=tokens, cached_tokens=0),
        timings=ev.Timings(queue_ms=0, ttft_ms=1, tpot_ms=1, e2e_ms=t - 1),
    )


def frames(*events: ev.Event, seqs: list[int] | None = None) -> list[ev.Frame]:
    seqs = seqs or list(range(len(events) + 1))
    return [
        ev.Frame(replica="r1", seq=seqs[i], t=e.t, events=[e])
        for i, e in enumerate([HELLO, *events])
    ]


GOOD = (
    arrived(1),
    admitted(1),
    step(2, "prefill", "a"),
    first(2),
    step(3, "decode", "a"),
    finished(3, tokens=2),
)


def problems(*events: ev.Event, **kw: object) -> list[str]:
    return check_trace(frames(*events), **kw).problems  # type: ignore[arg-type]


def test_a_good_trace_passes() -> None:
    assert problems(*GOOD) == []


def test_missing_hello() -> None:
    bad = [ev.Frame(replica="r1", seq=0, t=1, events=[arrived(1)])]
    assert any("hello" in p for p in check_trace(bad, complete=False).problems)


def test_sequence_gap() -> None:
    report = check_trace(frames(*GOOD, seqs=[0, 1, 2, 4, 5, 6, 7]))
    assert any("expected 3" in p for p in report.problems)


def test_time_going_backwards() -> None:
    assert any("backwards" in p for p in problems(arrived(5), admitted(1), complete=False))


def test_first_token_before_prefill() -> None:
    assert any(
        "out of order" in p for p in problems(arrived(1), admitted(1), first(1), complete=False)
    )


def test_decode_before_prefill() -> None:
    assert any(
        "decode of a" in p
        for p in problems(arrived(1), admitted(1), step(2, "decode", "a"), complete=False)
    )


def test_prefill_with_two_rows() -> None:
    out = problems(
        arrived(1), arrived(1, "b"), admitted(1), step(2, "prefill", "a", "b"), complete=False
    )
    assert any("2 rows" in p for p in out)


def test_token_count_mismatch() -> None:
    out = problems(*GOOD[:-1], finished(3, tokens=5))
    assert any("reports 5 tokens, steps sampled 2" in p for p in out)


def test_unknown_request() -> None:
    assert any("never announced" in p for p in problems(admitted(1, "ghost"), complete=False))


def test_unfinished_request_in_a_complete_trace() -> None:
    assert any("never ended" in p for p in problems(*GOOD[:-1]))
    assert problems(*GOOD[:-1], complete=False) == []


def test_preempted_request_resumes_without_a_second_first_token() -> None:
    story = (
        arrived(1),
        admitted(1),
        step(2, "prefill", "a"),
        first(2),
        ev.ReqPreempted(t=3, rid="a", completion_tokens=1, blocks_freed=1),
        admitted(4),
        step(5, "prefill", "a"),
        step(6, "decode", "a"),
        finished(6, tokens=3),
    )
    assert problems(*story) == []


def test_wrong_ttft() -> None:
    bad_first = ev.ReqFirstToken(t=2, rid="a", ttft_ms=1.5)
    out = problems(arrived(1), admitted(1), step(2, "prefill", "a"), bad_first, complete=False)
    assert any("ttft_ms" in p for p in out)
