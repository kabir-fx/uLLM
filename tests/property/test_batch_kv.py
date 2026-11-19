"""E7: ``BatchKV`` keeps each sequence's K and V intact through any mix of prefills,
decode appends, joins and releases (decision 0019).

The reference is each sequence's own K and V, concatenated token by token. The
operations follow ``MLXRunner``: a sequence prefills into its own one-row
``BatchKV``, joins the batch at a decode, and leaves it at release.
"""

import importlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytestmark = pytest.mark.mlx

mx = pytest.importorskip("mlx.core")
kv_batch = importlib.import_module("ullm.runner.kv.batch")  # after the skip: it imports MLX

if TYPE_CHECKING:
    from mlx.core import array as Array

    from ullm.runner.kv.batch import BatchKV

LAYERS, HEADS, DIM = 2, 2, 4


def _tokens(n_rows: int, n: int) -> list[tuple["Array", "Array"]]:
    """Random K and V for each layer: ``[n_rows, HEADS, n, DIM]``, bf16 like the model's."""
    shape = (n_rows, HEADS, n, DIM)
    return [
        (mx.random.normal(shape).astype(mx.bfloat16), mx.random.normal(shape).astype(mx.bfloat16))
        for _ in range(LAYERS)
    ]


@dataclass
class Model:
    """The runner's KV placement, beside the reference K and V of each sequence."""

    prefilling: dict[int, "BatchKV"] = field(default_factory=dict)
    batch: "BatchKV | None" = None
    reference: dict[int, list[tuple["Array", "Array"]]] = field(default_factory=dict)
    next_seq: int = 0

    def add(self) -> None:
        seq = self.next_seq
        self.next_seq += 1
        self.prefilling[seq] = kv_batch.BatchKV.empty(seq, LAYERS)
        self.reference[seq] = [
            (mx.zeros((1, HEADS, 0, DIM), mx.bfloat16), mx.zeros((1, HEADS, 0, DIM), mx.bfloat16))
            for _ in range(LAYERS)
        ]

    def prefill(self, seq: int, n: int) -> None:
        self._append(self.prefilling[seq], {seq: 0}, _tokens(1, n))

    def decode(self, joiners: list[int]) -> None:
        if joiners:
            rows = [(self.batch, s) for s in self.batch.seqs] if self.batch else []
            rows += [(self.prefilling[s], s) for s in joiners]
            self.batch = kv_batch.BatchKV.pack(rows)
            for s in joiners:
                del self.prefilling[s]
        assert self.batch is not None
        rows_of = {s: self.batch.row(s) for s in self.batch.seqs}
        self._append(self.batch, rows_of, _tokens(len(self.batch.seqs), 1))

    def release(self, seq: int) -> None:
        del self.reference[seq]
        if seq in self.prefilling:
            del self.prefilling[seq]
            return
        assert self.batch is not None
        rest = [(self.batch, s) for s in self.batch.seqs if s != seq]
        self.batch = kv_batch.BatchKV.pack(rest) if rest else None

    def _append(
        self, kv: "BatchKV", rows: dict[int, int], new: list[tuple["Array", "Array"]]
    ) -> None:
        for layer, (keys, values) in zip(kv.layers, new, strict=True):
            fetched = layer.update_and_fetch(keys, values)
            assert fetched[0].shape[2] == fetched[1].shape[2] == layer.length
        for seq, row in rows.items():
            self.reference[seq] = [
                (
                    mx.concatenate([rk, keys[row : row + 1]], axis=2),
                    mx.concatenate([rv, values[row : row + 1]], axis=2),
                )
                for (rk, rv), (keys, values) in zip(self.reference[seq], new, strict=True)
            ]

    def check(self) -> None:
        kvs = [*self.prefilling.values(), *([self.batch] if self.batch else [])]
        assert sorted(s for kv in kvs for s in kv.seqs) == sorted(self.reference)
        for kv in kvs:
            _check(kv, self.reference)


def _check(kv: "BatchKV", reference: dict[int, list[tuple["Array", "Array"]]]) -> None:
    lengths = [reference[s][0][0].shape[2] for s in kv.seqs]
    length = max(lengths)
    pads = [length - n for n in lengths]
    assert min(pads) == 0  # no padding common to every row
    for layer, layer_index in zip(kv.layers, range(LAYERS), strict=True):
        assert layer.length == length
        assert layer.offset.tolist() == lengths
        assert layer.pads.tolist() == pads
        if length:
            assert layer.keys is not None
            assert layer.values is not None
            fetched_keys, fetched_values = (
                layer.keys[..., :length, :],
                layer.values[..., :length, :],
            )
            for seq in kv.seqs:
                row = kv.row(seq)
                ref_keys, ref_values = reference[seq][layer_index]
                assert mx.array_equal(fetched_keys[row : row + 1, :, pads[row] :], ref_keys).item()
                assert mx.array_equal(
                    fetched_values[row : row + 1, :, pads[row] :], ref_values
                ).item()
        for n in (1, 2):
            _check_mask(layer.make_mask(n), pads, length, n)


def _check_mask(mask: object, pads: list[int], length: int, n: int) -> None:
    """Query ``q`` of row ``i`` sees key ``j`` exactly when ``pads[i] <= j <= length + q``."""
    if not any(pads):
        assert mask == (None if n == 1 else "causal")
        return
    assert isinstance(mask, mx.array)
    want = [
        [[[pad <= j <= length + q for j in range(length + n)] for q in range(n)]] for pad in pads
    ]
    assert mx.broadcast_to(mask, (len(pads), 1, n, length + n)).tolist() == want


Step = tuple[str, int, int]


@st.composite
def _steps(draw: st.DrawFn) -> list[Step]:
    # Each step is (operation, a draw that picks a sequence or a subset, a size).
    ops = st.sampled_from(["add", "prefill", "decode", "release"])
    return draw(
        st.lists(
            st.tuples(ops, st.integers(0, 2**16), st.integers(1, kv_batch.STEP + 8)),
            min_size=1,
            max_size=30,
        )
    )


@settings(deadline=None)
@given(_steps())
def test_each_row_holds_exactly_its_own_sequence(steps: list[Step]) -> None:
    model = Model()
    for op, pick, size in steps:
        live = sorted(model.reference)
        prefilling = sorted(model.prefilling)
        match op:
            case "add":
                model.add()
            case "prefill" if prefilling:
                model.prefill(prefilling[pick % len(prefilling)], size)
            case "decode" if live:
                joiners = [s for i, s in enumerate(prefilling) if pick >> i & 1]
                if model.batch is None and not joiners:
                    joiners = prefilling[:1]
                if joiners or model.batch is not None:
                    model.decode(joiners)
            case "release" if live:
                model.release(live[pick % len(live)])
        model.check()


def test_pack_copies_rows_and_leaves_its_sources_unchanged() -> None:
    a, b = kv_batch.BatchKV.empty(1, LAYERS), kv_batch.BatchKV.empty(2, LAYERS)
    for kv, n in ((a, 3), (b, 5)):
        for layer, (keys, values) in zip(kv.layers, _tokens(1, n), strict=True):
            layer.update_and_fetch(keys, values)
    before = [mx.array(layer.keys) for layer in a.layers]
    packed = kv_batch.BatchKV.pack([(a, 1), (b, 2)])
    for layer in packed.layers:
        layer.update_and_fetch(*_tokens(2, 1)[0])
    assert packed.seqs == (1, 2)
    assert all(
        mx.array_equal(layer.keys, kept).item()
        for layer, kept in zip(a.layers, before, strict=True)
    )
    assert a.layers[0].length == 3


def test_pack_refuses_an_empty_repeated_or_missing_row() -> None:
    kv = kv_batch.BatchKV.empty(1, LAYERS)
    with pytest.raises(ValueError, match="at least one row"):
        kv_batch.BatchKV.pack([])
    with pytest.raises(ValueError, match="twice"):
        kv_batch.BatchKV.pack([(kv, 1), (kv, 1)])
    with pytest.raises(ValueError, match="not in"):
        kv_batch.BatchKV.pack([(kv, 2)])


def test_the_mask_supports_only_full_causal_attention() -> None:
    layer = kv_batch.BatchKV.empty(1, LAYERS).layers[0]
    with pytest.raises(ValueError, match="causal"):
        layer.make_mask(5, return_array=True)
    with pytest.raises(ValueError, match="causal"):
        layer.make_mask(5, window_size=4)
