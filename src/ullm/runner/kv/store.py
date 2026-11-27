"""Every KV tensor of a runner, behind one interface (E9, decisions 0019, 0033 and 0034).

A step asks the store for its cache (``prefill`` or ``decode``), runs the model on
it, then takes ``written()`` into the step's one ``mx.eval``. A step that raises
calls ``fail()``. Only the engine thread calls a store (rule 2).

``BlockKV`` is the layout the runner uses: blocks are the store of record, and
the running batch is E7's contiguous working tensor. ``ContiguousKV`` is E7
alone, the reference that rule 1 compares it against.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

import mlx.core as mx

from ullm.runner.base import BlockId, SeqId
from ullm.runner.kv.batch import BatchKV, LayerKV


class KVStore(Protocol):
    def prefill(self, seq: SeqId, start: int, blocks: tuple[BlockId, ...]) -> list[LayerKV]:
        """The cache a prefill of ``seq`` runs on, holding its first ``start`` tokens."""
        ...

    def decode(self, tables: Mapping[SeqId, tuple[BlockId, ...]]) -> BatchKV:
        """The working batch of exactly these sequences; one not in it yet joins."""
        ...

    def written(self) -> list[mx.array]:
        """What the last prefill or decode wrote. Call after the forward, before its eval."""
        ...

    def release(self, seq: SeqId, freed: tuple[BlockId, ...]) -> None: ...

    def fail(self) -> None:
        """The last step raised: drop what it touched that no released sequence will free."""
        ...


class ContiguousKV:
    """E7: each sequence's KV is its own one-row ``BatchKV`` until its first decode, then a
    row of the working batch. Block ids are ignored."""

    def __init__(self, n_layers: int) -> None:
        self._n_layers = n_layers
        self._prefilling: dict[SeqId, BatchKV] = {}
        self._batch: BatchKV | None = None
        self._decoding = False  # the last step's kind, set before anything in it can raise
        self._prefilled: BatchKV | None = None  # the last prefill's cache

    def prefill(self, seq: SeqId, start: int, blocks: tuple[BlockId, ...]) -> list[LayerKV]:
        self._decoding = False
        self._prefilled = self._prefilling.setdefault(seq, BatchKV.empty(seq, self._n_layers))
        return self._prefilled.layers

    def decode(self, tables: Mapping[SeqId, tuple[BlockId, ...]]) -> BatchKV:
        self._decoding, self._prefilled = True, None
        self._batch = _joined(
            self._batch,
            tables,
            lambda seq: self._prefilling.pop(seq, None) or BatchKV.empty(seq, self._n_layers),
        )
        return self._batch

    def written(self) -> list[mx.array]:
        if self._decoding or self._prefilled is None:
            return []
        return [a for kv in self._prefilled.layers for a in (kv.keys, kv.values) if a is not None]

    def release(self, seq: SeqId, freed: tuple[BlockId, ...]) -> None:
        if self._prefilling.pop(seq, None) is None:
            self._batch = _without(self._batch, seq)

    def fail(self) -> None:
        # A failed prefill leaves its own row, which release drops. A failed decode failed
        # every row of the batch.
        if self._decoding:
            self._batch = None


class BlockKV:
    """B: each block id maps to one ``[n_kv_heads, n, head_dim]`` K and V per layer, ``n`` up to
    ``block_size``. A prefill runs on a one-row cache gathered from the sequence's blocks,
    and is cut back into them. A joining row is gathered from its blocks into the working
    batch, and a decode copies each block that it fills out of the working batch.

    Every full block of a live sequence is in the store. So is a prefilling sequence's
    partial last block. A decoding sequence's partial last block is only in the batch.
    """

    def __init__(self, n_layers: int, block_size: int) -> None:
        self._n_layers = n_layers
        self._size = block_size
        self._blocks: dict[BlockId, list[tuple[mx.array, mx.array]]] = {}
        self._held: dict[SeqId, int] = {}  # tokens of each prefilling sequence
        self._batch: BatchKV | None = None
        self._decoding = False  # the last step's kind, set before anything in it can raise
        self._step: _Prefill | _Decode | None = None  # what written() copies

    def prefill(self, seq: SeqId, start: int, blocks: tuple[BlockId, ...]) -> list[LayerKV]:
        self._decoding, self._step = False, None
        row = self._gathered(seq, blocks, start)
        self._step = _Prefill(seq, start, blocks, row)
        return row.layers

    def decode(self, tables: Mapping[SeqId, tuple[BlockId, ...]]) -> BatchKV:
        self._decoding, self._step = True, None
        self._batch = _joined(self._batch, tables, lambda seq: self._joining(seq, tables[seq]))
        self._step = _Decode(self._batch, dict(tables))
        return self._batch

    def written(self) -> list[mx.array]:
        step, self._step = self._step, None
        match step:
            case _Prefill():
                return self._cut(step)
            case _Decode():
                return self._copy_filled(step)
        return []

    def release(self, seq: SeqId, freed: tuple[BlockId, ...]) -> None:
        if self._held.pop(seq, None) is None:
            self._batch = _without(self._batch, seq)
        for block in freed:
            # A block the sequence never filled during decode was never stored.
            self._blocks.pop(block, None)

    def fail(self) -> None:
        # A failed decode failed every row of the batch. A failed prefill wrote no block.
        if self._decoding:
            self._batch = None
        self._step = None

    def _joining(self, seq: SeqId, table: tuple[BlockId, ...]) -> BatchKV:
        """The row gathered from its blocks. Its partial last block leaves the store: from
        here on that block grows in the batch, and the store's copy would go stale."""
        n = self._held.pop(seq, 0)
        row = self._gathered(seq, table, n)
        if n % self._size:
            self._blocks.pop(table[n // self._size])
        return row

    def _gathered(self, seq: SeqId, table: tuple[BlockId, ...], n: int) -> BatchKV:
        """A one-row batch of the first ``n`` tokens in ``table``'s blocks."""
        row = BatchKV.empty(seq, self._n_layers)
        if n == 0:
            return row
        used = [self._blocks[block] for block in table[: -(-n // self._size)]]
        for i, layer in enumerate(row.layers):
            layer.keys = mx.concatenate([b[i][0] for b in used], axis=1)[None, :, :n, :]
            layer.values = mx.concatenate([b[i][1] for b in used], axis=1)[None, :, :n, :]
            layer.length = n
        return row

    def _cut(self, step: "_Prefill") -> list[mx.array]:
        """Every block the prefill wrote, its partial last block included, out of its cache."""
        end = step.row.layers[0].length
        self._held[step.seq] = end
        written = []
        for index in range(step.start // self._size, -(-end // self._size)):
            begin = index * self._size
            block = [
                _copied(layer, 0, begin, min(begin + self._size, end)) for layer in step.row.layers
            ]
            self._blocks[step.blocks[index]] = block
            written += [a for kv in block for a in kv]
        return written

    def _copy_filled(self, step: "_Decode") -> list[mx.array]:
        batch = step.batch
        end = batch.layers[0].length
        written = []
        for row, seq in enumerate(batch.seqs):
            if (end - batch.pads[row]) % self._size:
                continue
            block = [_copied(layer, row, end - self._size, end) for layer in batch.layers]
            self._blocks[step.tables[seq][-1]] = block
            written += [a for kv in block for a in kv]
        return written


@dataclass(frozen=True, slots=True)
class _Prefill:
    seq: SeqId
    start: int
    blocks: tuple[BlockId, ...]
    row: BatchKV


@dataclass(frozen=True, slots=True)
class _Decode:
    batch: BatchKV
    tables: dict[SeqId, tuple[BlockId, ...]]


def _copied(layer: LayerKV, row: int, begin: int, end: int) -> tuple[mx.array, mx.array]:
    if layer.keys is None or layer.values is None:
        raise AssertionError("a layer that wrote tokens has K and V")
    keys = mx.contiguous(layer.keys[row, :, begin:end, :])
    values = mx.contiguous(layer.values[row, :, begin:end, :])
    return keys, values


def _joined(
    batch: BatchKV | None,
    tables: Mapping[SeqId, tuple[BlockId, ...]],
    row_of: Callable[[SeqId], BatchKV],
) -> BatchKV:
    """The batch with a row for each sequence in ``tables`` it lacks, from ``row_of``."""
    held = batch.seqs if batch is not None else ()
    joiners = [seq for seq in tables if seq not in held]
    if joiners:
        rows = [(batch, s) for s in held] if batch is not None else []
        rows += [(row_of(seq), seq) for seq in joiners]
        batch = BatchKV.pack(rows)
    if batch is None:
        raise AssertionError("a valid decode names at least one sequence")
    return batch


def _without(batch: BatchKV | None, seq: SeqId) -> BatchKV | None:
    """The batch repacked without ``seq``'s row, or unchanged when it has none."""
    if batch is None or seq not in batch.seqs:
        return batch
    rest = [(batch, s) for s in batch.seqs if s != seq]
    return BatchKV.pack(rest) if rest else None
