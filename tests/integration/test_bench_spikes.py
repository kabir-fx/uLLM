"""The capped rule gives the Final Approach's block counts on both committed spikes, outside Low
Power Mode (decision 0016). The spikes' engines: ``ullm`` and two identical ``mlx_lm.server``
processes, ``mlxA`` (the baseline) and ``mlxB`` (its twin)."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest
from bench.regimes import ChunkSpan, Regime, classify, read_powermetrics

SPIKES = Path(__file__).resolve().parents[2] / "bench/results/m1-spikes"


def _blocks(run: Path) -> dict[int, list[ChunkSpan]]:
    """Each measured block's chunk spans, Low Power Mode blocks left out."""
    edges: dict[tuple[int, str], list[float]] = defaultdict(list)
    lpm: set[int] = set()
    for line in (run / "raw.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["phase"] == "warmup":
            continue
        if row["phase"] == "lpm":
            lpm.add(row["block"])
        edges[row["block"], row["engine"]] += [row["t_send"], row["t_send"] + row["end_ms"] / 1000]
    blocks: dict[int, list[ChunkSpan]] = defaultdict(list)
    for (block, engine), times in edges.items():
        if block not in lpm:
            blocks[block].append(ChunkSpan(engine, min(times), max(times)))
    return blocks


@pytest.mark.parametrize(
    ("run", "counts"),
    [
        ("run-20261118-141243", {Regime.PEAK: 3, Regime.TRANSITION: 5, Regime.SUSTAINED: 4}),
        ("run-20261118-163540", {Regime.PEAK: 7, Regime.TRANSITION: 2, Regime.SUSTAINED: 7}),
    ],
)
def test_the_capped_rule_reproduces_the_approachs_counts(
    run: str, counts: dict[Regime, int]
) -> None:
    samples = read_powermetrics(SPIKES / run / "powermetrics.plist.gz")
    regimes = Counter(
        classify(spans, samples, ("mlxA", "mlxB")) for spans in _blocks(SPIKES / run).values()
    )
    assert regimes == counts
