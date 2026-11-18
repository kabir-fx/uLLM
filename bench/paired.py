"""Paired ratios between engines that ran in the same blocks, and the twin's noise floor
(decision 0016)."""

import random
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

MIN_BLOCKS = 3
RESAMPLES = 10_000


@dataclass(frozen=True)
class Estimate:
    point: float
    low: float | None
    high: float | None
    prompts: int
    blocks: int


def bootstrap(
    per_block: Mapping[int, Sequence[float]],
    seed: int,
    stat: Callable[[Sequence[float]], float] = statistics.median,
) -> Estimate:
    """``stat`` of every value pooled, with a 95% bootstrap interval that resamples whole
    blocks: prompts in one block shared the GPU's state, so they are not independent."""
    groups = [list(per_block[b]) for b in sorted(per_block)]
    pooled = [v for g in groups for v in g]
    point = stat(pooled)
    if len(groups) < MIN_BLOCKS:
        return Estimate(point, None, None, len(pooled), len(groups))
    rng = random.Random(seed)
    stats = sorted(
        stat([v for g in rng.choices(groups, k=len(groups)) for v in g]) for _ in range(RESAMPLES)
    )
    low = stats[int(0.025 * RESAMPLES)]
    high = stats[int(0.975 * RESAMPLES) - 1]
    return Estimate(point, low, high, len(pooled), len(groups))


def ratio(per_block: Mapping[int, Sequence[float]], seed: int) -> Estimate:
    """The median paired ratio and its interval."""
    return bootstrap(per_block, seed)


def floor(twin: Estimate) -> float | None:
    if twin.low is None or twin.high is None:
        return None
    return max(abs(twin.low - 1), abs(twin.high - 1))


def claimed(engine: Estimate, floor: float | None) -> bool:
    """A gap is claimed only outside the twin's noise."""
    return floor is not None and abs(engine.point - 1) > floor
