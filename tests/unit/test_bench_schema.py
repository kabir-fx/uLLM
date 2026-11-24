"""The group file refuses what a session would only find out halfway (decision 0016)."""

from pathlib import Path

import pytest
from bench.plan import plan
from bench.schema import Group, load_workload
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
    # A resident server starts once, before any block (decision 0028).
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


SWEEP = {1: 21, 2: 21, 4: 21, 8: 10, 16: 8}


@pytest.mark.parametrize(("concurrency", "blocks"), SWEEP.items())
def test_each_sweep_group_runs_the_same_engines_and_plans_its_blocks(
    concurrency: int, blocks: int
) -> None:
    # Decision 0028: one baseline and one uLLM configuration at every concurrency.
    root = Path(__file__).parents[2]
    group = Group.model_validate_json((root / f"bench/groups/m2-c{concurrency}.json").read_text())
    assert group.concurrency == concurrency
    assert group.servers == "per_chunk"
    assert group.baseline.name == "mlx-lm-batched-no-cache"
    names = [e.name for e in group.engines]
    # At c16 batched mlx-lm with its prompt cache ran out of GPU memory: m2-c16-cache shows it.
    expected = {1: ["ullm", "mlx-lm-batched", "mlx-lm-sequential-no-cache"], 16: ["ullm"]}
    assert names == expected.get(concurrency, ["ullm", "mlx-lm-batched"])
    [ullm] = [e for e in group.engines if e.name == "ullm"]
    assert "--max-num-seqs 16" in ullm.launch
    assert "{block}" in ullm.launch
    workload = load_workload(root / "bench/workloads/a.json")
    p = plan(workload.requests, names, group.prompts_per_chunk, group.warmup_per_chunk, 0)
    assert len(p.blocks) == blocks


def test_the_c16_cache_group_runs_cached_mlx_lm_against_the_c16_baseline() -> None:
    root = Path(__file__).parents[2] / "bench/groups"
    sweep = Group.model_validate_json((root / "m2-c16.json").read_text())
    cache = Group.model_validate_json((root / "m2-c16-cache.json").read_text())
    assert cache.baseline == sweep.baseline
    assert [e.name for e in cache.engines] == ["mlx-lm-batched"]
    assert "--prompt-cache-size" not in cache.engines[0].launch
