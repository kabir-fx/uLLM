"""Logical KV blocks (E8, decision 0032). MLX-free, engine thread only.

The manager owns every block table. A block is Free (no contents), In use
(ref count >= 1, never evicted) or Cached (ref count 0, contents and key kept
for a later prefix hit, evicted least recently cached first). Keys are E10's
chained prefix hashes; this module only stores them.
"""

from collections import OrderedDict, deque
from collections.abc import Sequence
from dataclasses import dataclass

from ullm.runner.base import SeqId

BlockId = int
BlockKey = int


class OutOfBlocks(Exception):
    """The pool cannot give the blocks asked for. Nothing was changed."""


@dataclass(slots=True)
class _Table:
    blocks: list[BlockId]
    num_tokens: int


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int = 16) -> None:
        if num_blocks < 1 or block_size < 1:
            raise ValueError("num_blocks and block_size must be at least 1")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.evictions = 0
        self._refs = [0] * num_blocks
        self._free: deque[BlockId] = deque(range(num_blocks))
        self._cached: OrderedDict[BlockId, None] = OrderedDict()  # oldest first
        self._key_of: dict[BlockId, BlockKey] = {}
        self._block_of: dict[BlockKey, BlockId] = {}
        self._tables: dict[SeqId, _Table] = {}

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_in_use(self) -> int:
        return sum(r > 0 for r in self._refs)

    @property
    def num_cached(self) -> int:
        return len(self._cached)

    def ref_count(self, block: BlockId) -> int:
        return self._refs[block]

    def table(self, seq: SeqId) -> tuple[BlockId, ...]:
        return tuple(self._tables[seq].blocks)

    def lookup(self, key: BlockKey) -> BlockId | None:
        return self._block_of.get(key)

    def can_allocate(self, num_tokens: int, shared: Sequence[BlockId] = ()) -> bool:
        """A shared block that is Cached leaves the cache, so it is not room for new blocks."""
        self._check_allocation(num_tokens, shared)
        revived = sum(self._refs[b] == 0 for b in shared)
        new = self._blocks_for(num_tokens) - len(shared)
        return new <= len(self._free) + len(self._cached) - revived

    def allocate(self, seq: SeqId, num_tokens: int, shared: Sequence[BlockId] = ()) -> None:
        """A table for ``num_tokens``: the ``shared`` prefix blocks first, then new blocks."""
        if seq in self._tables:
            raise ValueError(f"seq {seq} is already allocated")
        if not self.can_allocate(num_tokens, shared):
            raise OutOfBlocks(f"seq {seq} needs {self._blocks_for(num_tokens)} blocks")
        # Revive the shared blocks before taking new ones, so none of them is evicted.
        for block in shared:
            self._cached.pop(block, None)
            self._refs[block] += 1
        new = [self._take() for _ in range(self._blocks_for(num_tokens) - len(shared))]
        self._tables[seq] = _Table([*shared, *new], num_tokens)

    def can_append(self, seq: SeqId) -> bool:
        has_slot = self._tables[seq].num_tokens % self.block_size != 0
        return has_slot or bool(self._free) or bool(self._cached)

    def append_slot(self, seq: SeqId) -> None:
        """Room for one more token: decode writes into the last block, or a new one when full."""
        table = self._tables[seq]
        if not self.can_append(seq):
            raise OutOfBlocks(f"seq {seq} needs a block")
        if table.num_tokens % self.block_size == 0:
            table.blocks.append(self._take())
        table.num_tokens += 1

    def register(self, seq: SeqId, index: int, key: BlockKey) -> bool:
        """Gives the full block at ``index`` of the table its prefix key.

        False when another block already holds ``key``: this block stays without
        a key and becomes Free when released.
        """
        table = self._tables[seq]
        if index < 0 or (index + 1) * self.block_size > table.num_tokens:
            raise ValueError(f"block {index} of seq {seq} is not full")
        block = table.blocks[index]
        if block in self._key_of:
            raise ValueError(f"block {block} already has a key")
        if key in self._block_of:
            return False
        self._key_of[block] = key
        self._block_of[key] = block
        return True

    def free(self, seq: SeqId) -> tuple[BlockId, ...]:
        """The blocks that became Free: their contents are no longer needed (decision 0034)."""
        freed = []
        # Tail first: a later block is useful only after every block before it,
        # so it should be evicted first.
        for block in reversed(self._tables.pop(seq).blocks):
            self._refs[block] -= 1
            if self._refs[block] > 0:
                continue
            if block in self._key_of:
                self._cached[block] = None
            else:
                self._free.append(block)
                freed.append(block)
        return tuple(freed)

    def _check_allocation(self, num_tokens: int, shared: Sequence[BlockId]) -> None:
        if num_tokens < 1:
            raise ValueError("a sequence holds at least one token")
        if len(set(shared)) != len(shared):
            raise ValueError("a block is shared twice")
        if len(shared) * self.block_size > num_tokens:
            raise ValueError("the shared blocks hold more tokens than the sequence")
        if any(b not in self._key_of for b in shared):
            raise ValueError("a shared block has no key, so it is not full")

    def _blocks_for(self, num_tokens: int) -> int:
        return -(-num_tokens // self.block_size)

    def _take(self) -> BlockId:
        if self._free:
            block = self._free.popleft()
        else:
            block, _ = self._cached.popitem(last=False)
            del self._block_of[self._key_of.pop(block)]
            self.evictions += 1
        self._refs[block] = 1
        return block
