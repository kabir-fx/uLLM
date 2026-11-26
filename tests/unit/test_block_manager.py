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
