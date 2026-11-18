"""A session's plan: blocks with one prompt per prompt-length stratum, a warm-up pool no block
holds, and balanced engine orders (decision 0016)."""

from collections import Counter
from itertools import pairwise

import pytest
from bench.plan import plan, williams
from bench.schema import Message, WorkloadRequest


def _requests(n: int) -> list[WorkloadRequest]:
    # Lengths repeat so that ties are broken by id.
    return [
        WorkloadRequest(
            id=f"r-{i:03d}",
            messages=(Message(role="user", content="x"),),
            max_tokens=8,
            prompt_tokens=100 + (i * 37) % (n // 2 + 1),
        )
        for i in range(n)
    ]


def test_each_block_holds_one_prompt_from_each_stratum_and_no_warmup_prompt() -> None:
    requests = _requests(256)
    p = plan(requests, ["a", "b", "c", "d"], k=12, warmup_per_chunk=2, seed=0)
    ordered = sorted(requests, key=lambda r: (r.prompt_tokens, r.id))
    stratum = {r.id: i // 21 for i, r in enumerate(ordered[: 12 * 21])}
    assert len(p.blocks) == 21
    for block in p.blocks:
        assert sorted(stratum[r.id] for r in block) == list(range(12))
    measured = [r.id for block in p.blocks for r in block]
    assert len(set(measured)) == len(measured) == 252
    assert [r.id for r in p.warmup] == [r.id for r in ordered[252:]]
    assert not set(measured) & {r.id for r in p.warmup}


def test_the_plan_depends_only_on_the_seed() -> None:
    requests = _requests(256)
    first = plan(requests, ["a", "b"], k=12, warmup_per_chunk=2, seed=0)
    assert plan(list(reversed(requests)), ["a", "b"], k=12, warmup_per_chunk=2, seed=0) == first
    assert plan(requests, ["a", "b"], k=12, warmup_per_chunk=2, seed=1).blocks != first.blocks


def test_the_c8_group_gets_ten_blocks_and_sixteen_warmup_prompts() -> None:
    p = plan(_requests(256), list("abcde"), k=24, warmup_per_chunk=8, seed=0)
    assert len(p.blocks) == 10
    assert len(p.warmup) == 16


def test_a_warmup_pool_smaller_than_one_chunks_warmup_is_refused() -> None:
    with pytest.raises(ValueError, match="warm-up"):
        plan(_requests(252), ["a", "b"], k=12, warmup_per_chunk=2, seed=0)


def test_blocks_cycle_through_the_orders() -> None:
    p = plan(_requests(256), ["a", "b", "c", "d"], k=12, warmup_per_chunk=2, seed=0)
    assert [p.order(b) for b in range(8)] == [*p.orders, *p.orders]


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_williams_puts_each_engine_in_each_place_and_after_each_other_equally_often(
    n: int,
) -> None:
    names = [f"e{i}" for i in range(n)]
    orders = williams(names)
    assert len(orders) == (n if n % 2 == 0 else 2 * n)
    assert all(sorted(order) == names for order in orders)
    places = Counter((place, name) for order in orders for place, name in enumerate(order))
    assert set(places.values()) == {len(orders) // n}
    after = Counter(pair for order in orders for pair in pairwise(order))
    assert len(after) == n * (n - 1)
    assert len(set(after.values())) == 1


def test_williams_for_three_engines_is_every_ordering() -> None:
    assert len(set(williams(["a", "b", "c"]))) == 6
