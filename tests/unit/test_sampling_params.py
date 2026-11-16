"""C3: sampling settings are checked when built, like step plans (decision 0009)."""

from typing import Any

import pytest

from ullm.runner.base import SamplingParams


def _params(**changes: Any) -> SamplingParams:
    fields: dict[str, Any] = {"temperature": 0.7, "top_p": 0.9, "top_k": 40, "seed": 7}
    return SamplingParams(**(fields | changes))


@pytest.mark.parametrize(
    "fields",
    [
        {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "seed": None},
        {"temperature": 2, "top_p": 1e-6, "top_k": 1, "seed": -3},
        {"temperature": 1.2e-38, "top_p": 1.0, "top_k": 0, "seed": None},
        {"temperature": 3.4e38, "top_p": 1.0, "top_k": 0, "seed": None},
        {"temperature": 0.7, "top_p": 0.9, "top_k": 10**9, "seed": 2**80},
    ],
)
def test_valid_settings_are_accepted(fields: dict[str, Any]) -> None:
    SamplingParams(**fields)


@pytest.mark.parametrize(
    "bad",
    [-0.1, float("nan"), float("inf"), 1e300, 3.5e38, 1e-38, 5e-324, True, "0.7", None],
    ids=repr,
)
def test_temperature_is_zero_or_a_normal_float32(bad: Any) -> None:
    with pytest.raises(ValueError, match="temperature"):
        _params(temperature=bad)


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.01, float("nan"), True, None], ids=repr)
def test_top_p_is_in_zero_exclusive_to_one(bad: Any) -> None:
    with pytest.raises(ValueError, match="top_p"):
        _params(top_p=bad)


@pytest.mark.parametrize("bad", [-1, 2.0, True, None], ids=repr)
def test_top_k_is_a_non_negative_int(bad: Any) -> None:
    with pytest.raises(ValueError, match="top_k"):
        _params(top_k=bad)


@pytest.mark.parametrize("bad", [1.0, True, "7"], ids=repr)
def test_seed_is_an_int_or_none(bad: Any) -> None:
    with pytest.raises(ValueError, match="seed"):
        _params(seed=bad)


@pytest.mark.parametrize("temperature", [1, 10**20, 2**63])
def test_temperature_and_top_p_are_stored_as_floats(temperature: int) -> None:
    params = _params(temperature=temperature, top_p=1)
    assert type(params.temperature) is float and params.temperature == temperature
    assert type(params.top_p) is float and params.top_p == 1.0
