"""E5: the engine loop, run on the test thread with tickets queued in advance."""

from typing import Any

import pytest

from ullm.core.engine import Submit
from ullm.runner.base import SamplingParams

GREEDY = SamplingParams(temperature=0.0, top_p=1.0, top_k=0, seed=None)


def _submit(**changes: Any) -> Submit:
    fields: dict[str, Any] = {
        "rid": "r1-0",
        "prompt": (1, 2),
        "sampling": GREEDY,
        "max_tokens": 4,
        "stop": (),
    }
    return Submit(**(fields | changes))


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"rid": ""}, "rid"),
        ({"rid": 3}, "rid"),
        ({"prompt": ()}, "prompt"),
        ({"prompt": [1, 2]}, "prompt"),
        ({"prompt": (1, True)}, "prompt"),
        ({"prompt": (1, -1)}, "prompt"),
        ({"sampling": {"temperature": 0.0}}, "sampling"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"max_tokens": 2.0}, "max_tokens"),
        ({"stop": ["x"]}, "stop"),
        ({"stop": ("",)}, "stop"),
        ({"trace": 1}, "trace"),
        ({"tenant": b"acme"}, "tenant"),
    ],
    ids=repr,
)
def test_a_malformed_submit_is_refused(changes: dict[str, Any], field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _submit(**changes)


def test_a_well_formed_submit_is_built() -> None:
    assert _submit(trace="00-a-b-01", tenant="acme").prompt == (1, 2)
