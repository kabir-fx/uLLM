"""C3: step plans reject shapes and types the runner must never see (decision 0008)."""

from typing import Any

import pytest

from ullm.runner.base import DecodePlan, PrefillPlan, Runner


def test_prefill_needs_tokens_and_a_valid_start() -> None:
    assert PrefillPlan(seq=1, token_ids=(1, 2, 3), start=0).sample
    with pytest.raises(ValueError, match="at least one token"):
        PrefillPlan(seq=1, token_ids=(), start=0)
    with pytest.raises(ValueError, match="non-negative"):
        PrefillPlan(seq=1, token_ids=(1,), start=-1)


def test_decode_needs_one_token_per_distinct_sequence() -> None:
    DecodePlan(seqs=(1, 2), token_ids=(5, 6))
    with pytest.raises(ValueError, match="at least one sequence"):
        DecodePlan(seqs=(), token_ids=())
    with pytest.raises(ValueError, match="one input token per sequence"):
        DecodePlan(seqs=(1, 2), token_ids=(5,))
    with pytest.raises(ValueError, match="appears twice"):
        DecodePlan(seqs=(1, 1), token_ids=(5, 6))


def test_plans_are_immutable() -> None:
    plan = DecodePlan(seqs=(1,), token_ids=(5,))
    with pytest.raises(AttributeError):
        plan.seqs = (2,)  # type: ignore[misc]


def test_runner_protocol_is_structural() -> None:
    class NotARunner:
        def info(self) -> None: ...

    assert not isinstance(NotARunner(), Runner)


@pytest.mark.parametrize("bad", [True, 6.5, "5", None])
def test_plans_refuse_token_ids_that_are_not_plain_ints(bad: Any) -> None:
    with pytest.raises(ValueError, match="int"):
        PrefillPlan(seq=1, token_ids=(5, bad), start=0)
    with pytest.raises(ValueError, match="int"):
        DecodePlan(seqs=(1,), token_ids=(bad,))


@pytest.mark.parametrize("bad", [True, 0.0, "1"])
def test_plans_refuse_positions_and_ids_that_are_not_plain_ints(bad: Any) -> None:
    with pytest.raises(ValueError, match="int"):
        PrefillPlan(seq=1, token_ids=(5,), start=bad)
    with pytest.raises(ValueError, match="int"):
        PrefillPlan(seq=bad, token_ids=(5,), start=0)
    with pytest.raises(ValueError, match="int"):
        DecodePlan(seqs=(bad,), token_ids=(5,))


@pytest.mark.parametrize("bad", [1, 0, None, "True"])
def test_prefill_refuses_a_sample_flag_that_is_not_a_bool(bad: Any) -> None:
    with pytest.raises(ValueError, match="bool"):
        PrefillPlan(seq=1, token_ids=(5, 6), start=0, sample=bad)


@pytest.mark.parametrize(
    "make", [list, range, lambda ids: (i for i in ids)], ids=["list", "range", "generator"]
)
def test_plans_take_ids_and_seqs_only_as_tuples(make: Any) -> None:
    with pytest.raises(ValueError, match="tuple"):
        PrefillPlan(seq=1, token_ids=make((5, 6)), start=0)
    with pytest.raises(ValueError, match="tuple"):
        DecodePlan(seqs=make((1,)), token_ids=(5,))
    with pytest.raises(ValueError, match="tuple"):
        DecodePlan(seqs=(1,), token_ids=make((5,)))
