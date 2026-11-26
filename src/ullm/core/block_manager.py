"""Logical KV blocks (E8, decision 0032). MLX-free, engine thread only.

The manager owns every block table. A block is Free (no owner) or In use
(ref count >= 1).
"""

from collections import deque
from dataclasses import dataclass

from ullm.runner.base import SeqId

BlockId = int


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
        self._refs = [0] * num_blocks
        self._free: deque[BlockId] = deque(range(num_blocks))
        self._tables: dict[SeqId, _Table] = {}

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_in_use(self) -> int:
        return sum(r > 0 for r in self._refs)

    @property
    def num_cached(self) -> int:
        return 0

    def ref_count(self, block: BlockId) -> int:
        return self._refs[block]

    def table(self, seq: SeqId) -> tuple[BlockId, ...]:
        return tuple(self._tables[seq].blocks)

    def can_allocate(self, num_tokens: int) -> bool:
        return self._blocks_for(num_tokens) <= len(self._free)

    def allocate(self, seq: SeqId, num_tokens: int) -> None:
        if seq in self._tables:
            raise ValueError(f"seq {seq} is already allocated")
        if num_tokens < 1:
            raise ValueError("a sequence holds at least one token")
        if not self.can_allocate(num_tokens):
            raise OutOfBlocks(f"seq {seq} needs {self._blocks_for(num_tokens)} blocks")
        table = _Table([self._take() for _ in range(self._blocks_for(num_tokens))], num_tokens)
        self._tables[seq] = table

    def can_append(self, seq: SeqId) -> bool:
        return self._tables[seq].num_tokens % self.block_size != 0 or bool(self._free)

    def append_slot(self, seq: SeqId) -> None:
        """Room for one more token: decode writes into the last block, or a new one when full."""
        table = self._tables[seq]
        if not self.can_append(seq):
            raise OutOfBlocks(f"seq {seq} needs a block")
        if table.num_tokens % self.block_size == 0:
            table.blocks.append(self._take())
        table.num_tokens += 1

    def free(self, seq: SeqId) -> None:
        for block in self._tables.pop(seq).blocks:
            self._refs[block] -= 1
            if self._refs[block] == 0:
                self._free.append(block)

    def _blocks_for(self, num_tokens: int) -> int:
        return -(-num_tokens // self.block_size)

    def _take(self) -> BlockId:
        block = self._free.popleft()
        self._refs[block] = 1
        return block
