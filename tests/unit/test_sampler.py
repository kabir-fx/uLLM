"""E2: the sampler, one token per row of ``[B, vocab]`` logits (decision 0009)."""

import math
from typing import TYPE_CHECKING, cast

import pytest

from ullm.runner.base import SamplingParams

if TYPE_CHECKING:
    import mlx.core as mx

    from ullm.runner.sampler import SampleRow

pytestmark = pytest.mark.mlx

GREEDY = SamplingParams(temperature=0.0, top_p=1.0, top_k=0, seed=None)


def _logits(*rows: list[float]) -> "mx.array":
    import mlx.core as mx

    return mx.array(rows, dtype=mx.float32)


def _row(params: SamplingParams, seed: int = 1, position: int = 0) -> "SampleRow":
    from ullm.runner.sampler import SampleRow

    return SampleRow(params=params, seed=seed, position=position)


def _draws(logits: "mx.array", params: SamplingParams, n: int = 400, seed: int = 1) -> list[int]:
    from ullm.runner.sampler import sample

    return [cast(int, sample(logits, [_row(params, seed, p)]).item()) for p in range(n)]


def _from_probs(*probs: float) -> list[float]:
    return [math.log(p) for p in probs]


def test_greedy_is_the_argmax_of_each_row() -> None:
    import mlx.core as mx

    from ullm.runner.sampler import sample

    logits = mx.random.normal((4, 1000), key=mx.random.key(0))
    tokens = sample(logits, [_row(GREEDY, position=p) for p in range(4)])
    assert tokens.tolist() == mx.argmax(logits, axis=-1).tolist()


def test_top_k_samples_only_from_the_k_most_likely() -> None:
    logits = _logits([0.0, 3.0, 1.0, 2.5, -1.0, 2.0])
    params = SamplingParams(temperature=5.0, top_p=1.0, top_k=3, seed=None)
    assert set(_draws(logits, params)) == {1, 3, 5}


@pytest.mark.parametrize(("top_p", "kept"), [(0.75, {0, 1}), (0.85, {0, 1, 2}), (0.3, {0})])
def test_top_p_samples_from_the_smallest_set_reaching_p(top_p: float, kept: set[int]) -> None:
    logits = _logits(_from_probs(0.5, 0.3, 0.15, 0.05))
    params = SamplingParams(temperature=5.0, top_p=top_p, top_k=0, seed=None)
    assert set(_draws(logits, params)) == kept


def test_top_k_and_top_p_together_keep_the_smaller_set() -> None:
    logits = _logits(_from_probs(0.5, 0.3, 0.15, 0.05))
    params = SamplingParams(temperature=5.0, top_p=0.85, top_k=2, seed=None)
    assert set(_draws(logits, params)) == {0, 1}


def test_masks_apply_before_temperature() -> None:
    # As in mlx-lm: top-p sees the untempered distribution. At temperature 5, a
    # tempered nucleus at 0.75 would hold three tokens.
    logits = _logits(_from_probs(0.5, 0.3, 0.15, 0.05))
    params = SamplingParams(temperature=5.0, top_p=0.75, top_k=0, seed=None)
    assert set(_draws(logits, params)) == {0, 1}


def test_a_low_temperature_concentrates_on_the_most_likely() -> None:
    logits = _logits([2.0, 0.0, 1.9])
    params = SamplingParams(temperature=0.01, top_p=1.0, top_k=0, seed=None)
    assert set(_draws(logits, params, n=100)) == {0}


def test_temperature_one_follows_the_distribution() -> None:
    probs = (0.6, 0.3, 0.1)
    params = SamplingParams(temperature=1.0, top_p=1.0, top_k=0, seed=None)
    draws = _draws(_logits(_from_probs(*probs)), params, n=3000)
    for token, p in enumerate(probs):
        share = draws.count(token) / len(draws)
        # 4 standard errors: about one seed in 5,000 fails one of the three by chance.
        assert abs(share - p) < 4 * math.sqrt(p * (1 - p) / len(draws))


def test_a_seed_and_position_reproduce_the_token() -> None:
    params = SamplingParams(temperature=1.0, top_p=1.0, top_k=0, seed=None)
    logits = _logits([0.0] * 1000)
    assert _draws(logits, params, n=50, seed=42) == _draws(logits, params, n=50, seed=42)
    assert _draws(logits, params, n=50, seed=42) != _draws(logits, params, n=50, seed=43)


def test_a_row_draws_the_same_token_in_any_batch() -> None:
    import mlx.core as mx

    from ullm.runner.sampler import sample

    params = SamplingParams(temperature=1.0, top_p=0.95, top_k=50, seed=None)
    logits = mx.random.normal((3, 1000), key=mx.random.key(1))
    rows = [_row(params, seed=s, position=9) for s in (5, 6, 7)]
    together = sample(logits, rows).tolist()
    alone = [sample(logits[i : i + 1], [rows[i]]).item() for i in range(3)]
    reversed_ = cast(list[int], sample(logits[::-1], rows[::-1]).tolist())
    assert together == alone == reversed_[::-1]


def test_greedy_and_sampled_rows_share_a_batch() -> None:
    import mlx.core as mx

    from ullm.runner.sampler import sample

    logits = _logits([0.0, 9.0, 0.0], [0.0, 0.0, 0.0])
    hot = SamplingParams(temperature=1.0, top_p=1.0, top_k=0, seed=None)
    tokens = sample(logits, [_row(GREEDY), _row(hot)])
    assert tokens.dtype == mx.uint32
    assert tokens.shape == (2,)
    assert tokens[0].item() == 1


def test_any_int_seed_is_usable() -> None:
    params = SamplingParams(temperature=1.0, top_p=1.0, top_k=0, seed=None)
    logits = _logits([0.0] * 10)
    for seed in (-1, 0, 2**64, 2**100):
        _draws(logits, params, n=1, seed=seed)


def test_one_row_per_logits_row() -> None:
    from ullm.runner.sampler import sample

    with pytest.raises(ValueError, match="one sampling row per logits row"):
        sample(_logits([0.0, 1.0], [1.0, 0.0]), [_row(GREEDY)])


@pytest.mark.parametrize("temperature", [1e38, 1e300])
def test_a_huge_temperature_still_respects_the_masks(temperature: float) -> None:
    logits = _logits([5.0, 4.0, 1.0, 0.0, -1.0])
    for params in (
        SamplingParams(temperature=temperature, top_p=1.0, top_k=2, seed=None),
        SamplingParams(temperature=temperature, top_p=0.5, top_k=0, seed=None),
    ):
        allowed = {0, 1} if params.top_k else {0}
        assert set(_draws(logits, params, n=200)) == allowed


@pytest.mark.parametrize("top_p", [1e-50, 1e-38, 5e-324])
def test_a_tiny_top_p_keeps_the_most_likely_token(top_p: float) -> None:
    logits = _logits([5.0, 4.0, 1.0, 0.0])
    params = SamplingParams(temperature=1.0, top_p=top_p, top_k=0, seed=None)
    assert set(_draws(logits, params, n=100)) == {0}


def test_a_tiny_temperature_is_greedy_even_without_a_certain_token() -> None:
    logits = _logits([1.0, 0.0, 1.0 - 1e-3])
    params = SamplingParams(temperature=1e-30, top_p=1.0, top_k=0, seed=None)
    assert set(_draws(logits, params, n=50)) == {0}


def test_top_k_at_or_past_the_vocabulary_is_off() -> None:
    logits = _logits([0.0, 0.0, 0.0])
    for top_k in (3, 10**9):
        params = SamplingParams(temperature=1.0, top_p=1.0, top_k=top_k, seed=None)
        assert set(_draws(logits, params, n=100)) == {0, 1, 2}


def test_ties_at_the_edge_of_a_mask_stay() -> None:
    logits = _logits([2.0, 1.0, 1.0, 0.0])
    top_k = SamplingParams(temperature=1.0, top_p=1.0, top_k=2, seed=None)
    assert set(_draws(logits, top_k)) == {0, 1, 2}
    probs = _logits(_from_probs(0.4, 0.3, 0.3))
    top_p = SamplingParams(temperature=1.0, top_p=0.6, top_k=0, seed=None)
    assert set(_draws(probs, top_p)) == {0, 1, 2}
