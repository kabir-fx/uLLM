"""C3 v0: step plans reject shapes the runner must never see."""

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
