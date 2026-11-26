"""Q2 for E8 (decision 0032): random allocate, share, append, register, preempt, free and
evict sequences, with every invariant checked after each operation.

The model knows each sequence's token count and which key each of its blocks was
given, so it can tell when the manager hands out, finds or forgets the wrong block.
Eviction has no rule of its own: it happens whenever the free list runs out.
"""

from collections import Counter

from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from ullm.core.block_manager import BlockManager, OutOfBlocks

NUM_BLOCKS = 12
BLOCK_SIZE = 4
KEYS = range(10)  # few, so keys collide and blocks are shared often

PICK = st.integers(0, 1 << 16)


def _blocks_for(num_tokens: int) -> int:
    return -(-num_tokens // BLOCK_SIZE)


class BlockMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = BlockManager(NUM_BLOCKS, BLOCK_SIZE)
        self.tokens: dict[int, int] = {}  # live seq -> its token count, in allocation order
        self.keys: dict[tuple[int, int], int] = {}  # (live seq, block index) -> its key
        self.next_seq = 0

    @property
    def room(self) -> int:
        return self.blocks.num_free + self.blocks.num_cached

    def _pick(self, pick: int) -> int:
        return list(self.tokens)[pick % len(self.tokens)]

    def _allocate(self, num_tokens: int, shared: tuple[int, ...], keys: tuple[int, ...]) -> None:
        seq, self.next_seq = self.next_seq, self.next_seq + 1
        revived = sum(self.blocks.ref_count(b) == 0 for b in shared)
        fits = _blocks_for(num_tokens) - len(shared) <= self.room - revived
        assert self.blocks.can_allocate(num_tokens, shared) == fits
        if not fits:
            before = (self.blocks.num_free, self.blocks.num_cached, self.blocks.evictions)
            try:
                self.blocks.allocate(seq, num_tokens, shared)
            except OutOfBlocks:
                after = (self.blocks.num_free, self.blocks.num_cached, self.blocks.evictions)
                assert after == before
                return
            raise AssertionError("allocate past the pool did not raise")
        self.blocks.allocate(seq, num_tokens, shared)
        assert self.blocks.table(seq)[: len(shared)] == shared
        self.tokens[seq] = num_tokens
        for index, key in enumerate(keys):
            self.keys[seq, index] = key

    def _release(self, seq: int) -> None:
        self.blocks.free(seq)
        del self.tokens[seq]
        self.keys = {k: v for k, v in self.keys.items() if k[0] != seq}

    @rule(num_tokens=st.integers(1, NUM_BLOCKS * BLOCK_SIZE + 1))
    def allocate(self, num_tokens: int) -> None:
        self._allocate(num_tokens, (), ())

    @rule(keys=st.lists(st.sampled_from(KEYS), min_size=1, max_size=4), extra=st.integers(0, 9))
    def share(self, keys: list[int], extra: int) -> None:
        """A prefix hit: the blocks found for the leading keys, then new blocks."""
        hits: list[int] = []
        for key in keys:
            block = self.blocks.lookup(key)
            if block is None or block in hits:
                break
            hits.append(block)
        if hits:
            found = keys[: len(hits)]
            self._allocate(len(hits) * BLOCK_SIZE + extra, tuple(hits), tuple(found))

    @precondition(lambda self: self.tokens)
    @rule(pick=PICK)
    def append(self, pick: int) -> None:
        seq = self._pick(pick)
        fits = self.tokens[seq] % BLOCK_SIZE != 0 or self.room > 0
        assert self.blocks.can_append(seq) == fits
        if not fits:
            try:
                self.blocks.append_slot(seq)
            except OutOfBlocks:
                return
            raise AssertionError("append past the pool did not raise")
        self.blocks.append_slot(seq)
        self.tokens[seq] += 1

    @precondition(lambda self: self.tokens)
    @rule(pick=PICK, index=PICK, key=st.sampled_from(KEYS))
    def register(self, pick: int, index: int, key: int) -> None:
        seq = self._pick(pick)
        full = self.tokens[seq] // BLOCK_SIZE
        if full == 0:
            return
        index %= full
        if (seq, index) in self.keys:
            return  # registered here, or shared from a block with a key
        owner = self.blocks.lookup(key)
        took = self.blocks.register(seq, index, key)
        assert took == (owner is None)
        if took:
            self.keys[seq, index] = key

    @precondition(lambda self: self.tokens)
    @rule(pick=PICK)
    def free(self, pick: int) -> None:
        self._release(self._pick(pick))

    @precondition(lambda self: self.tokens)
    @rule()
    def preempt_the_newest(self) -> None:
        self._release(max(self.tokens))

    @rule()
    def fill_the_pool(self) -> None:
        """Freed blocks are reusable: everything not in use can be allocated at once."""
        if self.room == 0:
            assert not self.blocks.can_allocate(1)
            return
        self._allocate(self.room * BLOCK_SIZE, (), ())
        assert self.blocks.num_free == self.blocks.num_cached == 0

    @invariant()
    def each_block_is_owned_once_per_table_and_counted_once_per_owner(self) -> None:
        owners: Counter[int] = Counter()
        for seq, num_tokens in self.tokens.items():
            table = self.blocks.table(seq)
            assert len(table) == _blocks_for(num_tokens)
            assert len(set(table)) == len(table)
            owners.update(table)
        for block in range(NUM_BLOCKS):
            assert self.blocks.ref_count(block) == owners[block] >= 0

    @invariant()
    def free_used_and_cached_add_up_to_the_pool(self) -> None:
        used = len({b for seq in self.tokens for b in self.blocks.table(seq)})
        assert self.blocks.num_in_use == used
        assert self.blocks.num_free + self.blocks.num_in_use + self.blocks.num_cached == NUM_BLOCKS

    @invariant()
    def a_block_in_use_keeps_its_key(self) -> None:
        for (seq, index), key in self.keys.items():
            assert self.blocks.lookup(key) == self.blocks.table(seq)[index]

    @invariant()
    def a_key_finds_its_own_contents_or_a_cached_block(self) -> None:
        cached = 0
        for key in KEYS:
            block = self.blocks.lookup(key)
            if block is None:
                continue
            if self.blocks.ref_count(block) == 0:
                cached += 1
            else:
                holders = {k for (s, i), k in self.keys.items() if self.blocks.table(s)[i] == block}
                assert holders == {key}
        assert self.blocks.num_cached == cached


TestBlockManager = BlockMachine.TestCase
