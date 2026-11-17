"""B1 and B3: the metrics bench.analysis computes from raw.jsonl (decision 0013)."""

import pytest
from bench.analysis import FIGURES, Percentiles, headline, nearest_rank, repeat_stats
from bench.schema import RequestRecord, Usage
from hypothesis import given
from hypothesis import strategies as st


def _record(
    id: str,
    sent_s: float,
    chunk_ms: tuple[float, ...],
    end_ms: float,
    completion: int,
    *,
    repeat: int = 0,
    phase: str = "measure",
    prompt: int = 10,
    error: str | None = None,
) -> RequestRecord:
    return RequestRecord.model_validate(
        {
            "phase": phase,
            "repeat": repeat,
            "id": id,
            "sent_s": sent_s,
            "chunk_ms": chunk_ms,
            "end_ms": end_ms,
            "usage": None if error else Usage(prompt_tokens=prompt, completion_tokens=completion),
            "finish_reason": None if error else "length",
            "error": error,
        }
    )


def test_tpot_divides_by_usage_tokens_not_by_chunks() -> None:
    # 5 tokens in 3 chunks: TPOT is (40 - 10) / (5 - 1), not (40 - 10) / (3 - 1).
    [stats] = repeat_stats([_record("a", 0.0, (10.0, 25.0, 40.0), 45.0, 5)], {"a": 10})
    assert stats.ttft_ms == Percentiles(10.0, 10.0)
    assert stats.tpot_ms == Percentiles(7.5, 7.5)
    assert stats.e2e_ms == Percentiles(45.0, 45.0)


def test_throughput_is_output_tokens_over_the_repeats_wall_time() -> None:
    records = [
        _record("a", 0.0, (100.0, 900.0), 1000.0, 30),
        _record("b", 0.5, (100.0, 1400.0), 1500.0, 50),  # ends at 2.0 s
    ]
    [stats] = repeat_stats(records, {"a": 10, "b": 10})
    assert stats.output_tokens == 80
    assert stats.wall_s == pytest.approx(2.0)
    assert stats.tokens_per_s == pytest.approx(40.0)


def test_warmup_failures_and_mismatches_are_kept_apart() -> None:
    records = [
        _record("a", 0.0, (5.0,), 9.0, 1, phase="warmup", repeat=-1),
        _record("a", 1.0, (10.0, 20.0), 30.0, 2, repeat=0),
        _record("b", 1.1, (), 3.0, 0, repeat=0, error="HTTP 400: too long"),
        _record("a", 5.0, (12.0, 22.0), 31.0, 2, repeat=1, prompt=11),
    ]
    first, second = repeat_stats(records, {"a": 10, "b": 10})
    assert (first.repeat, first.requests, first.failed, first.prompt_mismatches) == (0, 2, 1, 0)
    assert first.ttft_ms == Percentiles(10.0, 10.0)  # the warm-up's 5.0 ms is not counted
    assert (second.repeat, second.prompt_mismatches) == (1, 1)


def test_a_one_token_reply_has_no_tpot() -> None:
    [stats] = repeat_stats([_record("a", 0.0, (10.0,), 12.0, 1)], {"a": 10})
    assert stats.tpot_ms is None


def test_nearest_rank_matches_its_definition() -> None:
    values = [float(v) for v in range(1, 101)]
    assert nearest_rank(values, 50) == 50.0
    assert nearest_rank(values, 99) == 99.0
    assert nearest_rank([3.0], 99) == 3.0


@given(st.lists(st.floats(0, 1e6, allow_nan=False), min_size=1, max_size=200), st.integers(1, 100))
def test_nearest_rank_is_a_sample_with_at_least_q_percent_at_or_below(
    values: list[float], q: int
) -> None:
    v = nearest_rank(values, q)
    assert v in values
    assert sum(x <= v for x in values) * 100 >= q * len(values)
    assert sum(x < v for x in values) * 100 < q * len(values)


def test_a_reply_with_no_text_is_not_a_failure() -> None:
    [stats] = repeat_stats([_record("a", 0.0, (), 12.0, 1)], {"a": 10})
    assert stats.failed == 0
    assert stats.ttft_ms is None
    assert stats.e2e_ms == Percentiles(12.0, 12.0)


def test_the_headline_is_the_median_over_repeats() -> None:
    # TTFT 10, 30 and 20 ms in repeats 0, 1 and 2: the headline is 20, not the first, last,
    # best or worst repeat.
    records = [
        _record(f"r{repeat}", 0.0, (ttft, ttft + 10.0), 100.0, 2, repeat=repeat)
        for repeat, ttft in enumerate((10.0, 30.0, 20.0))
    ]
    stats = repeat_stats(records, {f"r{repeat}": 10 for repeat in range(3)})
    figures = dict(zip(FIGURES, headline(stats), strict=True))
    assert figures["TTFT p50"] == 20.0
    assert figures["E2E p50"] == 100.0
