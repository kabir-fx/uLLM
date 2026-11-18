"""Which prompts each block holds and the order engines take their turns in (decision 0016)."""

import random
from collections.abc import Sequence
from dataclasses import dataclass

from bench.schema import WorkloadRequest


@dataclass(frozen=True)
class Plan:
    blocks: tuple[tuple[WorkloadRequest, ...], ...]
    warmup: tuple[WorkloadRequest, ...]
    orders: tuple[tuple[str, ...], ...]

    def order(self, block: int) -> tuple[str, ...]:
        return self.orders[block % len(self.orders)]


def williams(names: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    """A Williams design: each name in each place, and right after each other name, equally
    often. ``n`` orders for an even ``n``; for an odd ``n`` the reversed orders join them."""
    n = len(names)
    first = [0]
    for i in range(1, n):
        first.append((i + 1) // 2 if i % 2 else n - i // 2)
    orders = [tuple(names[(x + shift) % n] for x in first) for shift in range(n)]
    if n % 2:
        orders += [order[::-1] for order in orders]
    return tuple(orders)


def plan(
    requests: Sequence[WorkloadRequest],
    engines: Sequence[str],
    k: int,
    warmup_per_chunk: int,
    seed: int,
) -> Plan:
    """Blocks of ``k`` prompts, one from each of ``k`` prompt-length strata. The longest
    requests left over form the warm-up pool, never measured."""
    ordered = sorted(requests, key=lambda r: (r.prompt_tokens, r.id))
    n = len(ordered) // k
    pool = tuple(ordered[k * n :])
    if len(pool) < warmup_per_chunk:
        raise ValueError(
            f"{len(pool)} requests are left for warm-up after {k} strata of {n}; "
            f"each chunk needs {warmup_per_chunk}"
        )
    rng = random.Random(seed)
    blocks: list[list[WorkloadRequest]] = [[] for _ in range(n)]
    for s in range(k):
        stratum = ordered[s * n : (s + 1) * n]
        rng.shuffle(stratum)
        for b, request in enumerate(stratum):
            blocks[b].append(request)
    for block in blocks:
        rng.shuffle(block)
    return Plan(tuple(map(tuple, blocks)), pool, williams(engines))
