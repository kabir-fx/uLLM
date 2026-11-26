"""Q2 for E8 (decision 0032): random allocate, free and preempt sequences, with every
invariant checked after each operation."""

from collections import Counter

from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from ullm.core.block_manager import BlockManager, OutOfBlocks

NUM_BLOCKS = 12
BLOCK_SIZE = 4


class BlockMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = BlockManager(NUM_BLOCKS, BLOCK_SIZE)
        self.tokens: dict[int, int] = {}  # live seq -> its token count, in allocation order
        self.next_seq = 0

    def _blocks_for(self, num_tokens: int) -> int:
        return -(-num_tokens // BLOCK_SIZE)

    @rule(num_tokens=st.integers(1, NUM_BLOCKS * BLOCK_SIZE + 1))
    def allocate(self, num_tokens: int) -> None:
        seq, self.next_seq = self.next_seq, self.next_seq + 1
        room = self.blocks.num_free + self.blocks.num_cached
        fits = self._blocks_for(num_tokens) <= room
        assert self.blocks.can_allocate(num_tokens) == fits
        if not fits:
            try:
                self.blocks.allocate(seq, num_tokens)
            except OutOfBlocks:
                return
            raise AssertionError("allocate past the pool did not raise")
        self.blocks.allocate(seq, num_tokens)
        self.tokens[seq] = num_tokens

    @precondition(lambda self: self.tokens)
    @rule(pick=st.integers(0, 1 << 16))
    def free(self, pick: int) -> None:
        seq = list(self.tokens)[pick % len(self.tokens)]
        self.blocks.free(seq)
        del self.tokens[seq]

    @precondition(lambda self: self.tokens)
    @rule()
    def preempt_the_newest(self) -> None:
        seq = max(self.tokens)
        self.blocks.free(seq)
        del self.tokens[seq]

    @rule()
    def fill_the_pool(self) -> None:
        """Freed blocks are reusable: everything not in use can be allocated at once."""
        room = self.blocks.num_free + self.blocks.num_cached
        if room == 0:
            assert not self.blocks.can_allocate(1)
            return
        seq, self.next_seq = self.next_seq, self.next_seq + 1
        self.blocks.allocate(seq, room * BLOCK_SIZE)
        self.tokens[seq] = room * BLOCK_SIZE
        assert not self.blocks.can_allocate(1)

    @invariant()
    def each_block_is_owned_once_per_table_and_counted_once_per_owner(self) -> None:
        owners: Counter[int] = Counter()
        for seq, num_tokens in self.tokens.items():
            table = self.blocks.table(seq)
            assert len(table) == self._blocks_for(num_tokens)
            assert len(set(table)) == len(table)
            owners.update(table)
        for block in range(NUM_BLOCKS):
            assert self.blocks.ref_count(block) == owners[block]

    @invariant()
    def free_used_and_cached_add_up_to_the_pool(self) -> None:
        used = len({b for seq in self.tokens for b in self.blocks.table(seq)})
        assert self.blocks.num_in_use == used
        assert self.blocks.num_free + self.blocks.num_in_use + self.blocks.num_cached == NUM_BLOCKS


TestBlockManager = BlockMachine.TestCase
