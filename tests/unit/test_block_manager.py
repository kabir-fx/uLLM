"""E8 (decision 0032): a fixed pool of 16-token blocks, one block table per sequence."""

import pytest

from ullm.core.block_manager import BlockManager, OutOfBlocks


def test_a_prompt_takes_one_block_per_16_tokens_rounded_up() -> None:
    blocks = BlockManager(num_blocks=8)
    blocks.allocate(seq=0, num_tokens=33)
    assert len(blocks.table(0)) == 3
    assert (blocks.num_free, blocks.num_in_use, blocks.num_cached) == (5, 3, 0)


def test_two_sequences_get_different_blocks() -> None:
    blocks = BlockManager(num_blocks=4)
    blocks.allocate(seq=0, num_tokens=32)
    blocks.allocate(seq=1, num_tokens=32)
    assert not set(blocks.table(0)) & set(blocks.table(1))


def test_can_allocate_counts_the_free_blocks() -> None:
    blocks = BlockManager(num_blocks=2)
    assert blocks.can_allocate(32)
    assert not blocks.can_allocate(33)


def test_allocating_past_the_pool_raises_and_changes_nothing() -> None:
    blocks = BlockManager(num_blocks=2)
    blocks.allocate(seq=0, num_tokens=16)
    with pytest.raises(OutOfBlocks):
        blocks.allocate(seq=1, num_tokens=17)
    assert blocks.num_free == 1
    with pytest.raises(KeyError):
        blocks.table(1)


def test_freeing_returns_every_block_and_they_are_reusable() -> None:
    blocks = BlockManager(num_blocks=3)
    blocks.allocate(seq=0, num_tokens=48)
    blocks.free(0)
    assert blocks.num_free == 3
    blocks.allocate(seq=1, num_tokens=48)
    assert blocks.num_in_use == 3


def test_a_sequence_is_allocated_once() -> None:
    blocks = BlockManager(num_blocks=4)
    blocks.allocate(seq=0, num_tokens=1)
    with pytest.raises(ValueError, match="already"):
        blocks.allocate(seq=0, num_tokens=1)


def test_freeing_an_unknown_sequence_raises() -> None:
    with pytest.raises(KeyError):
        BlockManager(num_blocks=1).free(0)


@pytest.mark.parametrize("num_tokens", [0, -1])
def test_a_sequence_holds_at_least_one_token(num_tokens: int) -> None:
    with pytest.raises(ValueError, match="at least one token"):
        BlockManager(num_blocks=1).allocate(seq=0, num_tokens=num_tokens)


def test_the_block_size_is_configurable() -> None:
    blocks = BlockManager(num_blocks=8, block_size=4)
    blocks.allocate(seq=0, num_tokens=9)
    assert len(blocks.table(0)) == 3


@pytest.mark.parametrize(("num_blocks", "block_size"), [(0, 16), (1, 0)])
def test_the_pool_and_the_block_are_not_empty(num_blocks: int, block_size: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        BlockManager(num_blocks=num_blocks, block_size=block_size)


def test_decode_fills_the_last_block_before_taking_another() -> None:
    blocks = BlockManager(num_blocks=4, block_size=4)
    blocks.allocate(seq=0, num_tokens=3)
    blocks.append_slot(0)
    assert len(blocks.table(0)) == 1
    blocks.append_slot(0)
    assert len(blocks.table(0)) == 2


def test_appending_needs_a_block_only_when_the_last_is_full() -> None:
    blocks = BlockManager(num_blocks=2, block_size=4)
    blocks.allocate(seq=0, num_tokens=4)
    blocks.allocate(seq=1, num_tokens=3)
    blocks.append_slot(1)
    assert not blocks.can_append(0)
    assert not blocks.can_append(1)
    with pytest.raises(OutOfBlocks):
        blocks.append_slot(0)
    assert len(blocks.table(0)) == 1
    blocks.free(1)
    assert blocks.can_append(0)


def _cached(blocks: BlockManager, seq: int, keys: tuple[int, ...]) -> tuple[int, ...]:
    """Allocates ``seq`` with one full block per key, registers them and frees it."""
    blocks.allocate(seq, num_tokens=len(keys) * blocks.block_size)
    for index, key in enumerate(keys):
        assert blocks.register(seq, index, key)
    table = blocks.table(seq)
    blocks.free(seq)
    return table


def test_a_freed_block_with_a_key_is_cached_and_one_without_is_free() -> None:
    blocks = BlockManager(num_blocks=4, block_size=4)
    blocks.allocate(seq=0, num_tokens=6)
    assert blocks.register(0, 0, key=10)
    first, _ = blocks.table(0)
    blocks.free(0)
    assert (blocks.num_free, blocks.num_cached, blocks.num_in_use) == (3, 1, 0)
    assert blocks.lookup(10) == first


def test_only_a_full_block_takes_a_key() -> None:
    blocks = BlockManager(num_blocks=4, block_size=4)
    blocks.allocate(seq=0, num_tokens=6)
    with pytest.raises(ValueError, match="not full"):
        blocks.register(0, 1, key=11)
    blocks.append_slot(0)
    blocks.append_slot(0)
    assert blocks.register(0, 1, key=11)


def test_a_block_takes_one_key() -> None:
    blocks = BlockManager(num_blocks=4, block_size=4)
    blocks.allocate(seq=0, num_tokens=4)
    blocks.register(0, 0, key=10)
    with pytest.raises(ValueError, match="already has a key"):
        blocks.register(0, 0, key=11)


def test_a_key_held_by_another_block_is_not_taken_over() -> None:
    blocks = BlockManager(num_blocks=4, block_size=4)
    blocks.allocate(seq=0, num_tokens=4)
    blocks.allocate(seq=1, num_tokens=4)
    assert blocks.register(0, 0, key=10)
    assert not blocks.register(1, 0, key=10)
    assert blocks.lookup(10) == blocks.table(0)[0]
    blocks.free(1)
    assert blocks.num_cached == 0


def test_sharing_a_cached_block_revives_it() -> None:
    blocks = BlockManager(num_blocks=4, block_size=4)
    (block,) = _cached(blocks, seq=0, keys=(10,))
    blocks.allocate(seq=1, num_tokens=6, shared=(block,))
    assert blocks.table(1)[0] == block
    assert blocks.ref_count(block) == 1
    assert (blocks.num_free, blocks.num_cached, blocks.num_in_use) == (2, 0, 2)


def test_sharing_a_block_in_use_counts_both_owners() -> None:
    blocks = BlockManager(num_blocks=4, block_size=4)
    blocks.allocate(seq=0, num_tokens=4)
    blocks.register(0, 0, key=10)
    (block,) = blocks.table(0)
    blocks.allocate(seq=1, num_tokens=4, shared=(block,))
    assert blocks.ref_count(block) == 2
    blocks.free(0)
    assert blocks.ref_count(block) == 1
    assert blocks.num_cached == 0


def test_only_a_block_with_a_key_is_shared() -> None:
    blocks = BlockManager(num_blocks=4, block_size=4)
    blocks.allocate(seq=0, num_tokens=4)
    (block,) = blocks.table(0)
    with pytest.raises(ValueError, match="no key"):
        blocks.allocate(seq=1, num_tokens=4, shared=(block,))


def test_shared_blocks_are_distinct_and_cover_at_most_the_tokens() -> None:
    blocks = BlockManager(num_blocks=4, block_size=4)
    a, b = _cached(blocks, seq=0, keys=(10, 11))
    with pytest.raises(ValueError, match="twice"):
        blocks.allocate(seq=1, num_tokens=8, shared=(a, a))
    with pytest.raises(ValueError, match="more tokens"):
        blocks.allocate(seq=1, num_tokens=7, shared=(a, b))
    assert blocks.num_cached == 2


def test_free_blocks_are_taken_before_cached_ones() -> None:
    blocks = BlockManager(num_blocks=2, block_size=4)
    _cached(blocks, seq=0, keys=(10,))
    blocks.allocate(seq=1, num_tokens=4)
    assert blocks.lookup(10) is not None
    assert blocks.evictions == 0


def test_an_empty_pool_evicts_the_least_recently_cached_block() -> None:
    blocks = BlockManager(num_blocks=2, block_size=4)
    (old,) = _cached(blocks, seq=0, keys=(10,))
    blocks.allocate(seq=1, num_tokens=4)
    blocks.register(1, 0, key=11)
    blocks.free(1)
    assert blocks.can_allocate(8)
    blocks.allocate(seq=2, num_tokens=4)
    assert blocks.table(2) == (old,)
    assert blocks.lookup(10) is None
    assert blocks.lookup(11) is not None
    assert blocks.evictions == 1


def test_a_freed_table_is_evicted_tail_first() -> None:
    """A later block is useful only after every block before it, so it goes first."""
    blocks = BlockManager(num_blocks=3, block_size=4)
    head, _, tail = _cached(blocks, seq=0, keys=(10, 11, 12))
    blocks.allocate(seq=1, num_tokens=4)
    assert blocks.table(1) == (tail,)
    assert blocks.lookup(10) == head


def test_reviving_a_cached_block_does_not_count_it_as_room() -> None:
    blocks = BlockManager(num_blocks=2, block_size=4)
    a, b = _cached(blocks, seq=0, keys=(10, 11))
    assert blocks.can_allocate(8, shared=(a,))
    assert not blocks.can_allocate(9, shared=(a,))
    with pytest.raises(OutOfBlocks):
        blocks.allocate(seq=1, num_tokens=9, shared=(a,))
    assert blocks.num_cached == 2
    blocks.allocate(seq=1, num_tokens=8, shared=(a,))
    assert blocks.table(1) == (a, b)
    assert blocks.lookup(11) is None


def test_a_hit_moves_a_block_to_the_back_of_the_eviction_order() -> None:
    blocks = BlockManager(num_blocks=3, block_size=4)
    (a,) = _cached(blocks, seq=0, keys=(10,))
    (b,) = _cached(blocks, seq=1, keys=(11,))
    blocks.allocate(seq=2, num_tokens=4, shared=(a,))
    blocks.free(2)
    blocks.allocate(seq=3, num_tokens=8)
    assert blocks.lookup(11) is None
    assert blocks.lookup(10) == a
    assert b in blocks.table(3)


def test_a_shared_block_is_not_evicted_to_make_room_for_its_own_sequence() -> None:
    blocks = BlockManager(num_blocks=2, block_size=4)
    (oldest,) = _cached(blocks, seq=0, keys=(10,))
    (newer,) = _cached(blocks, seq=1, keys=(11,))
    blocks.allocate(seq=2, num_tokens=8, shared=(oldest,))
    assert blocks.table(2) == (oldest, newer)
    assert blocks.lookup(10) == oldest
