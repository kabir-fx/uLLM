"""The group file refuses what a session would only find out halfway (decision 0016)."""

import pytest
from bench.schema import Group
from pydantic import ValidationError


def test_a_group_whose_engine_replaces_a_common_request_field_is_refused() -> None:
    spec = {"launch": "x", "model": None, "footprint_gb": 1.0}
    group = {
        "name": "g",
        "concurrency": 1,
        "servers": "resident",
        "prompts_per_chunk": 2,
        "warmup_per_chunk": 1,
        "seed": 0,
        "baseline": {"name": "base", **spec},
        "engines": [{"name": "other", **spec, "extra_body": {"temperature": 1}}],
        "gates": {"twin_tpot_peak": 0.1, "twin_tpot_sustained": 0.1},
    }
    with pytest.raises(ValidationError, match="temperature"):
        Group.model_validate(group)
