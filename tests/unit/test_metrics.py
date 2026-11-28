"""O2: ``Metrics`` turns C2 events into Prometheus metrics (decision 0024)."""

from typing import Literal

import pytest
from prometheus_client.parser import text_string_to_metric_families

from ullm.core.block_manager import KvUsage
from ullm.telemetry import events as ev
from ullm.telemetry.metrics import Metrics

Samples = dict[tuple[str, frozenset[tuple[str, str]]], float]
IDLE = KvUsage.idle(num_blocks=8, block_size=16)


def _samples(metrics: Metrics, waiting: int = 0, running: int = 0, kv: KvUsage = IDLE) -> Samples:
    text = metrics.render(waiting=waiting, running=running, kv=kv).decode()
    return {
        (s.name, frozenset(s.labels.items())): s.value
        for family in text_string_to_metric_families(text)
        for s in family.samples
    }


def _value(samples: Samples, name: str, **labels: str) -> float:
    return samples[(name, frozenset(labels.items()))]


def _finished(
    rid: str, reason: Literal["stop", "length"], completion: int, tpot_ms: float | None
) -> ev.ReqFinished:
    return ev.ReqFinished(
        t=100.0,
        rid=rid,
        reason=reason,
        usage=ev.TokenUsage(prompt_tokens=7, completion_tokens=completion, cached_tokens=0),
        timings=ev.Timings(queue_ms=5.0, ttft_ms=40.0, tpot_ms=tpot_ms, e2e_ms=1500.0),
    )


def _step(kind: Literal["prefill", "decode"], rids: list[str], tokens: int, ms: float) -> ev.Step:
    return ev.Step(
        t=10.0,
        n=1,
        kind=kind,
        rids=rids,
        tokens=tokens,
        ms=ms,
        waiting=0,
        running=len(rids),
        kv_tokens=tokens,
    )


def test_a_fresh_registry_shows_every_series_at_zero() -> None:
    samples = _samples(Metrics())
    for reason in ("stop", "length", "tool_calls"):
        assert _value(samples, "ullm_requests_total", finish_reason=reason) == 0
    for reason in ("client_disconnect", "shutdown", "error"):
        assert _value(samples, "ullm_aborts_total", reason=reason) == 0
    for kind in ("prefill", "decode"):
        assert _value(samples, "ullm_step_seconds_count", kind=kind) == 0
    for name in (
        "ullm_prompt_tokens_total",
        "ullm_generation_tokens_total",
        "ullm_prefix_cache_hit_tokens_total",
        "ullm_preemptions_total",
        "ullm_telemetry_dropped_total",
        "ullm_ttft_seconds_count",
        "ullm_tpot_seconds_count",
        "ullm_e2e_seconds_count",
        "ullm_queue_wait_seconds_count",
        "ullm_queue_depth",
        "ullm_batch_size",
    ):
        assert _value(samples, name) == 0, name


def test_no_series_is_labelled_by_request() -> None:
    m = Metrics()
    m.observe(
        [
            ev.ReqAdmitted(t=1.0, rid="r-0", queue_ms=1.0),
            _step("prefill", ["r-0"], 7, 3.0),
            ev.ReqFirstToken(t=2.0, rid="r-0", ttft_ms=2.0),
            _finished("r-0", "stop", 1, None),
        ],
        0,
    )
    labels = {key for _, labels in _samples(m) for key, _ in labels}
    assert labels <= {"finish_reason", "reason", "kind", "state", "le"}


def test_tokens_count_the_work_of_each_step() -> None:
    m = Metrics()
    m.observe(
        [
            _step("prefill", ["a"], 7, 3.0),
            ev.ReqFirstToken(t=1.0, rid="a", ttft_ms=1.0),
            _step("prefill", ["b"], 5, 2.0),
            ev.ReqFirstToken(t=2.0, rid="b", ttft_ms=2.0),
            _step("decode", ["a", "b"], 2, 1.0),
            _step("decode", ["a", "b"], 2, 1.0),
            _step("decode", ["b"], 1, 1.0),
        ],
        0,
    )
    samples = _samples(m)
    assert _value(samples, "ullm_prompt_tokens_total") == 12
    # Each request's first token, then one a row per decode step.
    assert _value(samples, "ullm_generation_tokens_total") == 2 + 5


def test_a_finished_request_counts_its_reason_and_its_timings_in_seconds() -> None:
    m = Metrics()
    m.observe([_finished("a", "stop", 3, 20.0), _finished("b", "length", 1, None)], 0)
    samples = _samples(m)
    assert _value(samples, "ullm_requests_total", finish_reason="stop") == 1
    assert _value(samples, "ullm_requests_total", finish_reason="length") == 1
    assert _value(samples, "ullm_e2e_seconds_count") == 2
    assert _value(samples, "ullm_e2e_seconds_sum") == pytest.approx(3.0)
    # One output token has no TPOT.
    assert _value(samples, "ullm_tpot_seconds_count") == 1
    assert _value(samples, "ullm_tpot_seconds_sum") == pytest.approx(0.020)
    assert _value(samples, "ullm_tpot_seconds_bucket", le="0.015") == 0
    assert _value(samples, "ullm_tpot_seconds_bucket", le="0.02") == 1


def test_ttft_and_queue_wait_are_observed_when_they_end() -> None:
    m = Metrics()
    m.observe(
        [
            ev.ReqAdmitted(t=1.0, rid="a", queue_ms=250.0, cached_tokens=4),
            ev.ReqFirstToken(t=2.0, rid="a", ttft_ms=400.0),
        ],
        0,
    )
    samples = _samples(m)
    assert _value(samples, "ullm_queue_wait_seconds_count") == 1
    assert _value(samples, "ullm_queue_wait_seconds_sum") == pytest.approx(0.25)
    assert _value(samples, "ullm_ttft_seconds_count") == 1
    assert _value(samples, "ullm_ttft_seconds_sum") == pytest.approx(0.4)
    assert _value(samples, "ullm_prefix_cache_hit_tokens_total") == 4


def test_steps_are_timed_by_kind() -> None:
    m = Metrics()
    m.observe([_step("prefill", ["a"], 7, 30.0), _step("decode", ["a"], 1, 15.0)], 0)
    samples = _samples(m)
    assert _value(samples, "ullm_step_seconds_sum", kind="prefill") == pytest.approx(0.030)
    assert _value(samples, "ullm_step_seconds_sum", kind="decode") == pytest.approx(0.015)


def test_aborts_preemptions_and_drops_are_counted() -> None:
    m = Metrics()
    m.observe(
        [
            ev.ReqAborted(t=1.0, rid="a", reason="client_disconnect", completion_tokens=2),
            ev.ReqAborted(t=1.0, rid="b", reason="error", completion_tokens=0),
            ev.ReqPreempted(t=1.0, rid="c", completion_tokens=3, blocks_freed=2),
        ],
        5,
    )
    m.observe([], 2)
    samples = _samples(m)
    assert _value(samples, "ullm_aborts_total", reason="client_disconnect") == 1
    assert _value(samples, "ullm_aborts_total", reason="error") == 1
    assert _value(samples, "ullm_aborts_total", reason="shutdown") == 0
    assert _value(samples, "ullm_preemptions_total") == 1
    assert _value(samples, "ullm_telemetry_dropped_total") == 7


def test_the_gauges_are_the_load_given_at_render() -> None:
    m = Metrics()
    samples = _samples(m, waiting=3, running=2)
    assert _value(samples, "ullm_queue_depth") == 3
    assert _value(samples, "ullm_batch_size") == 2
    samples = _samples(m)
    assert (_value(samples, "ullm_queue_depth"), _value(samples, "ullm_batch_size")) == (0, 0)


def test_the_kv_gauges_are_the_pool_given_at_render() -> None:
    kv = KvUsage(num_blocks=10, block_size=4, free=5, in_use=4, cached=1, empty_slots=3)
    samples = _samples(Metrics(), kv=kv)
    assert _value(samples, "ullm_kv_blocks", state="free") == 5
    assert _value(samples, "ullm_kv_blocks", state="in_use") == 4
    assert _value(samples, "ullm_kv_blocks", state="cached") == 1
    assert _value(samples, "ullm_kv_waste_ratio") == 3 / 16
    idle = _samples(Metrics())
    assert _value(idle, "ullm_kv_blocks", state="free") == 8
    assert _value(idle, "ullm_kv_waste_ratio") == 0


def _hello(max_num_seqs: int, num_blocks: int) -> ev.Hello:
    return ev.Hello(
        t=0.0,
        replica="r1",
        model="fake",
        source="engine",
        version="0",
        commit=None,
        epoch_unix_ms=0,
        limits=ev.Limits(max_num_seqs=max_num_seqs, max_prefill_tokens=256, max_model_len=256),
        kv=ev.KvInfo(block_size=16, num_blocks=num_blocks, capacity_tokens=num_blocks * 16),
        machine=None,
    )


@pytest.mark.parametrize(
    ("max_num_seqs", "num_blocks", "want"),
    [(16, 40, 2), (16, 47, 2), (16, 2560, 16), (1, 16, 1)],
)
def test_max_concurrent_seqs_is_how_many_max_model_len_requests_the_pool_holds(
    max_num_seqs: int, num_blocks: int, want: int
) -> None:
    m = Metrics()
    assert _value(_samples(m), "ullm_kv_max_concurrent_seqs") == 0  # before hello
    m.observe([_hello(max_num_seqs, num_blocks)], 0)  # max_model_len 256: 16 blocks
    assert _value(_samples(m), "ullm_kv_max_concurrent_seqs") == want


def test_two_instances_keep_separate_counts() -> None:
    a, b = Metrics(), Metrics()
    a.observe([_finished("a", "stop", 3, 20.0)], 0)
    assert _value(_samples(b), "ullm_requests_total", finish_reason="stop") == 0
