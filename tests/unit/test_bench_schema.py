"""The group file refuses what a session would only find out halfway (decision 0016)."""

from pathlib import Path

import pytest
from bench.schema import Group
from pydantic import ValidationError


def _group(servers: str, launch: str, extra_body: dict[str, object]) -> dict[str, object]:
    spec = {"launch": "x", "model": None, "footprint_gb": 1.0}
    return {
        "name": "g",
        "concurrency": 1,
        "servers": servers,
        "prompts_per_chunk": 2,
        "warmup_per_chunk": 1,
        "seed": 0,
        "baseline": {"name": "base", **spec},
        "engines": [{"name": "other", **spec, "launch": launch, "extra_body": extra_body}],
        "gates": {"twin_tpot_peak": 0.1, "twin_tpot_sustained": 0.1},
    }


def test_a_group_whose_engine_replaces_a_common_request_field_is_refused() -> None:
    with pytest.raises(ValidationError, match="temperature"):
        Group.model_validate(_group("resident", "x", {"temperature": 1}))


def test_a_resident_server_cannot_name_a_block() -> None:
    # A resident server starts once, before any block (decision 0026).
    launch = "x --record {out}/r-{block}.jsonl.gz"
    with pytest.raises(ValidationError, match="block"):
        Group.model_validate(_group("resident", launch, {}))
    Group.model_validate(_group("per_chunk", launch, {}))


@pytest.mark.parametrize("name", ["m1-c1", "m1-c8"])
def test_the_m1_groups_rerun_ullm_at_batch_size_1_as_committed(name: str) -> None:
    # M1-T7 measured uLLM sequential; the engine batches by default since M2-T2 (0020).
    path = Path(__file__).parents[2] / "bench" / "groups" / f"{name}.json"
    group = Group.model_validate_json(path.read_text())
    [ullm] = [e for e in group.engines if e.name == "ullm"]
    assert ullm.launch.split()[-2:] == ["--max-num-seqs", "1"]


@pytest.mark.parametrize(
    "path", sorted((Path(__file__).parents[2] / "bench/groups").glob("*.json")), ids=str
)
def test_every_committed_group_is_valid(path: Path) -> None:
    Group.model_validate_json(path.read_text())
