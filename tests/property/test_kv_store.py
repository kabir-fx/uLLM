"""E9: both ``KVStore`` layouts keep each sequence's K and V intact through any mix of
prefill chunks, joins, decodes, failed steps and releases (decision 0034).

Both stores run in lockstep with the same random K and V, against one reference per
sequence, as ``MLXRunner`` drives them: the step's cache from the store, the model's
writes into it, then ``written()`` into one eval. Block tables come from a real
``BlockManager`` with 4-token blocks, so rows cross block boundaries often.
"""

import importlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ullm.core.block_manager import BlockManager

pytestmark = pytest.mark.mlx

mx = pytest.importorskip("mlx.core")
store = importlib.import_module("ullm.runner.kv.store")  # after the skip: it imports MLX
batch_test = importlib.import_module("tests.property.test_batch_kv")

if TYPE_CHECKING:
    from mlx.core import array as Array

    from ullm.runner.kv.batch import LayerKV

LAYERS, HEADS, DIM = batch_test.LAYERS, batch_test.HEADS, batch_test.DIM
BLOCK = 4
Reference = dict[int, list[tuple["Array", "Array"]]]


@dataclass
class Model:
    blocks: BlockManager = field(default_factory=lambda: BlockManager(512, BLOCK))
    stores: dict[str, Any] = field(
        default_factory=lambda: {
            "blocks": store.BlockKV(LAYERS, BLOCK),
            "contiguous": store.ContiguousKV(LAYERS),
        }
    )
    reference: Reference = field(default_factory=dict)
    prefilling: set[int] = field(default_factory=set)
    allocated: set[int] = field(default_factory=set)
    next_seq: int = 0

    def add(self) -> None:
        self.prefilling.add(self.next_seq)
        self.reference[self.next_seq] = [
            (mx.zeros((1, HEADS, 0, DIM), mx.bfloat16), mx.zeros((1, HEADS, 0, DIM), mx.bfloat16))
            for _ in range(LAYERS)
        ]
        self.next_seq += 1

    def prefill(self, seq: int, n: int, fails: bool) -> None:
        start = self._held(seq)
        if start == 0:
            self.blocks.allocate(seq, n)
            self.allocated.add(seq)
        else:
            for _ in range(n):
                self.blocks.append_slot(seq)
        new = batch_test._tokens(1, n)
        for kv in self.stores.values():
            layers = kv.prefill(seq, start, self.blocks.table(seq))
            _write(layers, new)
            if fails:
                kv.fail()
                continue
            mx.eval(kv.written())
            _check_row(layers, 0, self.reference[seq], new)
        if fails:
            self.release(seq)
        else:
            self._extend({seq: 0}, new)

    def decode(self, joiners: list[int], fails: bool) -> None:
        seqs = [s for s in self.reference if s not in self.prefilling] + joiners
        for seq in seqs:
            self.blocks.append_slot(seq)
        tables = {seq: self.blocks.table(seq) for seq in seqs}
        batches = {name: kv.decode(tables) for name, kv in self.stores.items()}
        order = batches["blocks"].seqs
        assert batches["contiguous"].seqs == order
        new = batch_test._tokens(len(order), 1)
        for name, kv in self.stores.items():
            _write(batches[name].layers, new)
            if fails:
                kv.fail()
            else:
                mx.eval(kv.written())
        self.prefilling -= set(joiners)
        if fails:
            for seq in seqs:
                self.release(seq)
            return
        rows = {seq: order.index(seq) for seq in seqs}
        self._extend(rows, new)
        for b in batches.values():
            batch_test._check(b, self.reference)

    def release(self, seq: int) -> None:
        freed = self.blocks.free(seq) if seq in self.allocated else ()
        self.allocated.discard(seq)
        for kv in self.stores.values():
            kv.release(seq, freed)
        del self.reference[seq]
        self.prefilling.discard(seq)

    def check_held(self) -> None:
        """``BlockKV`` holds a block only while a live sequence's table names it, and holds
        each full block of a live sequence as it is. So is a prefilling one's partial block."""
        held = self.stores["blocks"]._blocks
        assert set(held) <= {b for seq in self.allocated for b in self.blocks.table(seq)}
        for seq in self.allocated:
            n = self._held(seq)
            stored = -(-n // BLOCK) if seq in self.prefilling else n // BLOCK
            for index, block in enumerate(self.blocks.table(seq)[:stored]):
                span = slice(index * BLOCK, min((index + 1) * BLOCK, n))
                for (keys, values), (ref_keys, ref_values) in zip(
                    held[block], self.reference[seq], strict=True
                ):
                    assert mx.array_equal(keys, ref_keys[0, :, span]).item()
                    assert mx.array_equal(values, ref_values[0, :, span]).item()

    def _held(self, seq: int) -> int:
        return int(self.reference[seq][0][0].shape[2])

    def _extend(self, rows: dict[int, int], new: list[tuple["Array", "Array"]]) -> None:
        for seq, row in rows.items():
            self.reference[seq] = [
                (
                    mx.concatenate([rk, keys[row : row + 1]], axis=2),
                    mx.concatenate([rv, values[row : row + 1]], axis=2),
                )
                for (rk, rv), (keys, values) in zip(self.reference[seq], new, strict=True)
            ]


def _write(layers: list["LayerKV"], new: list[tuple["Array", "Array"]]) -> None:
    """What the model's attention does to the cache in a forward."""
    for layer, (keys, values) in zip(layers, new, strict=True):
        layer.update_and_fetch(keys, values)


def _check_row(
    layers: list["LayerKV"],
    row: int,
    before: list[tuple["Array", "Array"]],
    new: list[tuple["Array", "Array"]],
) -> None:
    for layer, (ref_keys, ref_values), (keys, values) in zip(layers, before, new, strict=True):
        n = layer.length
        assert layer.keys is not None
        assert layer.values is not None
        want_keys = mx.concatenate([ref_keys, keys], axis=2)
        want_values = mx.concatenate([ref_values, values], axis=2)
        assert mx.array_equal(layer.keys[row : row + 1, :, :n], want_keys).item()
        assert mx.array_equal(layer.values[row : row + 1, :, :n], want_values).item()


Step = tuple[str, int, int, bool]


@settings(deadline=None)
@given(
    st.lists(
        st.tuples(
            st.sampled_from(["add", "prefill", "decode", "release"]),
            st.integers(0, 2**16),
            st.integers(1, 3 * BLOCK + 1),
            st.integers(0, 9).map(lambda n: n == 0),  # one step in ten fails
        ),
        min_size=1,
        max_size=30,
    )
)
def test_both_layouts_keep_each_sequence_intact_and_blocks_free_with_it(steps: list[Step]) -> None:
    model = Model()
    for op, pick, size, fails in steps:
        live = sorted(model.reference)
        prefilling = sorted(model.prefilling)
        decoding = [s for s in live if s not in model.prefilling]
        match op:
            case "add":
                model.add()
            case "prefill" if prefilling:
                model.prefill(prefilling[pick % len(prefilling)], size, fails)
            case "decode" if live:
                started = [s for s in prefilling if model._held(s)]
                joiners = [s for i, s in enumerate(started) if pick >> i & 1]
                if decoding or joiners:
                    model.decode(joiners, fails)
            case "release" if live:
                model.release(live[pick % len(live)])
        model.check_held()
    for seq in sorted(model.reference):
        model.release(seq)
    assert model.stores["blocks"]._blocks == {}
