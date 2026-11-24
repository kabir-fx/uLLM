"""``bench.steps`` splits one chunk's recorded steps into the figures that explain a gap
(decision 0026). Every expected value below is worked out by hand from the events."""

from typing import Literal

import pytest
from bench.regimes import ChunkSpan
from bench.steps import Split, chunk, split

from ullm.telemetry import events as ev

EPOCH_MS = 1_000_000  # t = 0 is unix 1000 s


def _hello() -> ev.Hello:
    return ev.Hello(
        t=0,
        replica="r",
        model="m",
        source="engine",
        version="0",
        commit=None,
        epoch_unix_ms=EPOCH_MS,
        limits=ev.Limits(max_num_seqs=16, max_prefill_tokens=4096, max_model_len=4096),
        kv=ev.KvInfo(),
        machine=None,
    )


def _step(
    t: float, ms: float, kind: Literal["prefill", "decode"], rids: list[str], tokens: int
) -> ev.Step:
    return ev.Step(
        t=t,
        n=0,
        kind=kind,
        rids=rids,
        tokens=tokens,
        ms=ms,
        waiting=0,
        running=len(rids),
        kv_tokens=0,
    )


def _finished(t: float, rid: str) -> ev.ReqFinished:
    usage = ev.TokenUsage(prompt_tokens=1, completion_tokens=1, cached_tokens=0)
    timings = ev.Timings(queue_ms=0, ttft_ms=0, tpot_ms=None, e2e_ms=0)
    return ev.ReqFinished(t=t, rid=rid, reason="stop", usage=usage, timings=timings)


def _arrived(t: float, rid: str) -> ev.ReqArrived:
    return ev.ReqArrived(t=t, rid=rid, prompt_tokens=1, max_tokens=1)


EVENTS: list[ev.Event] = [
    _hello(),
    _arrived(0, "a"),
    _step(100, 100, "prefill", ["a"], 500),
    ev.ReqFirstToken(t=100, rid="a", ttft_ms=100),
    _step(120, 15, "decode", ["a"], 1),
    _arrived(125, "b"),
    _step(225, 100, "prefill", ["b"], 1000),
    ev.ReqFirstToken(t=225, rid="b", ttft_ms=100),
    _step(250, 20, "decode", ["a", "b"], 2),
    _step(275, 20, "decode", ["a", "b"], 2),
    _finished(275, "a"),
    _step(300, 15, "decode", ["b"], 1),
    _finished(300, "b"),
    # After the client's span: the next chunk's warm-up, never counted.
    _arrived(1400, "z"),
    _step(1500, 10, "prefill", ["z"], 100),
]


def _split() -> Split:
    frames = [ev.Frame(replica="r", seq=0, t=1500, events=EVENTS)]
    return split([chunk(frames, ChunkSpan("ullm", 1000.0, 1001.0))])


def test_decode_steps_are_grouped_by_rows() -> None:
    assert _split().decode_ms_by_rows == {1: [15, 15], 2: [20, 20]}


def test_prefill_cost_is_per_thousand_prompt_tokens() -> None:
    assert _split().prefill_ms_per_1k == [200, 100]


def test_time_inside_the_span_is_split_by_step_kind() -> None:
    s = _split()
    assert s.prefill_ms == 200
    assert s.decode_ms == 70
    assert s.span_ms == pytest.approx(1000)


def test_a_gap_counts_only_while_the_engine_has_work() -> None:
    # The gap before a's prefill and after b's last decode are idle, not overhead.
    assert _split().gaps_ms == [5, 5, 5, 5, 10]


def test_a_requests_stall_is_others_prefill_while_it_decodes() -> None:
    # a decodes from 100 to 275 and waits through b's prefill, 125 to 225. b waits for none.
    assert _split().stall_shares == [pytest.approx(100 / 175), 0]


def test_dropped_events_are_counted() -> None:
    frames = [ev.Frame(replica="r", seq=0, t=1500, dropped=3, events=EVENTS)]
    assert split([chunk(frames, ChunkSpan("ullm", 1000.0, 1001.0))]).dropped == 3


def test_a_recording_without_hello_is_refused_by_name() -> None:
    # A server that dies before its first frame leaves a recording with no epoch to align.
    with pytest.raises(ValueError, match="no hello"):
        chunk([], ChunkSpan("ullm", 1000.0, 1001.0))
