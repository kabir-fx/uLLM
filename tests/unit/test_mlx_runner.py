"""What MLXRunner refuses in M1, and the mask its cache gives the model (decision 0008)."""

from typing import TYPE_CHECKING

import pytest

from ullm.runner.base import DecodePlan, PrefillPlan, SamplingParams

if TYPE_CHECKING:
    from ullm.runner.mlx_runner import MLXRunner, MLXTokenizer

pytestmark = pytest.mark.mlx

contiguous = pytest.importorskip("ullm.runner.kv.contiguous")

GREEDY = SamplingParams(temperature=0.0, top_p=1.0, top_k=0, seed=None)


def test_sampling_above_temperature_zero_is_refused_and_adds_nothing(
    mlx_loaded: "tuple[MLXRunner, MLXTokenizer]",
) -> None:
    runner, _ = mlx_loaded
    with pytest.raises(ValueError, match="M1-T2"):
        runner.add(100, SamplingParams(temperature=0.7, top_p=1.0, top_k=0, seed=1))
    runner.add(100, GREEDY)
    runner.release(100)


def test_batched_decode_is_refused_without_failing_the_sequences(
    mlx_loaded: "tuple[MLXRunner, MLXTokenizer]",
) -> None:
    runner, _ = mlx_loaded
    for seq in (101, 102):
        runner.add(seq, GREEDY)
        runner.execute(PrefillPlan(seq=seq, token_ids=(9707, 11), start=0))
    with pytest.raises(ValueError, match="M2-T1"):
        runner.execute(DecodePlan(seqs=(101, 102), token_ids=(11, 11)))
    for seq in (101, 102):
        runner.execute(DecodePlan(seqs=(seq,), token_ids=(11,)))
        runner.release(seq)


def test_the_cache_asks_for_no_mask_on_one_query_and_causal_otherwise() -> None:
    kv = contiguous.ContiguousKV()
    assert kv.make_mask(1) is None
    assert kv.make_mask(5) == "causal"
    with pytest.raises(ValueError, match="causal"):
        kv.make_mask(5, window_size=4)
