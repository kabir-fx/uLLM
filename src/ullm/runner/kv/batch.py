"""The KV of the running batch: one row per sequence, left-padded (E7, decision 0019).

Rows end at the same slot, so a decode writes every row's next token at one
index, in place. Changing which sequences have rows copies them all
(``BatchKV.pack``). ``LayerKV`` plugs into mlx-lm's model modules through
``offset``, ``update_and_fetch`` and ``make_mask`` (roadmap section 4).
"""

from collections.abc import Sequence

import mlx.core as mx
from mlx_lm.models.base import create_causal_mask

from ullm.runner.base import SeqId

STEP = 256  # capacity grows in steps of this many tokens, as mlx-lm's KVCache does


class LayerKV:
    def __init__(
        self,
        keys: mx.array | None,
        values: mx.array | None,
        length: int,
        pads: mx.array,
        padded: bool,
    ) -> None:
        # [B, n_kv_heads, capacity, head_dim]: row i holds pads[i] zeros, then its tokens.
        self.keys = keys
        self.values = values
        self.length = length  # slots written, the same for every row
        self.pads = pads  # [B] int32, shared by every layer of one BatchKV
        self._padded = padded

    @property
    def offset(self) -> mx.array:
        """``[B]``: each row's own position, for RoPE."""
        return self.length - self.pads

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        start, n = self.length, keys.shape[2]
        if self.keys is None or self.values is None or start + n > self.keys.shape[2]:
            self.keys = _grown(self.keys, keys, start + n)
            self.values = _grown(self.values, values, start + n)
        self.length += n
        self.keys[..., start : self.length, :] = keys
        self.values[..., start : self.length, :] = values
        return self.keys[..., : self.length, :], self.values[..., : self.length, :]

    def make_mask(
        self, N: int, return_array: bool = False, window_size: int | None = None
    ) -> mx.array | str | None:
        """mlx-lm's mask for a plain KV cache when no row is padded, so one sequence attends
        exactly as in mlx-lm; else a causal mask that also hides each row's padding."""
        if return_array or window_size is not None:
            raise ValueError("LayerKV supports only full causal attention")
        if self._padded:
            mask: mx.array = create_causal_mask(N, offset=self.length, left_padding=self.pads)
            return mask
        return None if N == 1 else "causal"


def _capacity(length: int) -> int:
    """The next multiple of STEP above ``length``, so the next token has a free slot."""
    return (length // STEP + 1) * STEP


def _grown(held: mx.array | None, new: mx.array, length: int) -> mx.array:
    rows, heads, _, dim = new.shape
    grown = mx.zeros((rows, heads, _capacity(length), dim), new.dtype)
    if held is not None:
        written = length - new.shape[2]
        grown[..., :written, :] = held[..., :written, :]
    return grown


class BatchKV:
    """The KV of a fixed set of sequences, one row each, left-padded to one length."""

    def __init__(
        self, seqs: tuple[SeqId, ...], pads: tuple[int, ...], layers: list[LayerKV]
    ) -> None:
        self.seqs = seqs  # row order: the order rows were packed, not plan order
        self.pads = pads  # the host's copy of every layer's pads, so packing never syncs
        self.layers = layers  # model(x, cache=batch.layers)

    @classmethod
    def empty(cls, seq: SeqId, n_layers: int) -> "BatchKV":
        """One row with no tokens: where a sequence prefills."""
        pads = mx.zeros((1,), mx.int32)
        return cls((seq,), (0,), [LayerKV(None, None, 0, pads, False) for _ in range(n_layers)])

    @classmethod
    def pack(cls, rows: Sequence[tuple["BatchKV", SeqId]]) -> "BatchKV":
        """A new batch of the named rows, in the order given; the sources do not change.

        Each row is copied right-aligned to the longest, so no padding is common to every
        row. The only way to change which sequences share a batch.
        """
        if not rows:
            raise ValueError("a batch needs at least one row")
        seqs = tuple(seq for _, seq in rows)
        if len(set(seqs)) != len(seqs):
            raise ValueError(f"a sequence is named twice in {seqs}")
        for source, seq in rows:
            if seq not in source.seqs:
                raise ValueError(f"sequence {seq} is not in the batch named for it")
        spans = [(source.row(seq), source.layers[0].length, source) for source, seq in rows]
        lengths = [end - source.pads[row] for row, end, source in spans]
        length = max(lengths)
        pads = tuple(length - n for n in lengths)
        pads_array = mx.array(pads, mx.int32)
        layers = []
        for i in range(len(rows[0][0].layers)):
            keys, values = _packed(
                [
                    (source.layers[i], row, end, n)
                    for (row, end, source), n in zip(spans, lengths, strict=True)
                ],
                length,
            )
            layers.append(LayerKV(keys, values, length, pads_array, any(pads)))
        return cls(seqs, pads, layers)

    def row(self, seq: SeqId) -> int:
        return self.seqs.index(seq)


def _packed(
    rows: list[tuple[LayerKV, int, int, int]], length: int
) -> tuple[mx.array | None, mx.array | None]:
    """One layer's K and V for rows given as (layer, row, end, tokens), right-aligned to
    ``length`` in one fresh allocation; (None, None) when no row holds a token."""
    held = [(layer.keys, layer.values) for layer, _, _, n in rows if n]
    if not held:
        return None, None
    like_keys, like_values = held[0]
    if like_keys is None or like_values is None:
        raise AssertionError("a row that holds tokens has K and V")
    _, heads, _, key_dim = like_keys.shape
    keys = mx.zeros((len(rows), heads, _capacity(length), key_dim), like_keys.dtype)
    values = mx.zeros(
        (len(rows), heads, _capacity(length), like_values.shape[3]), like_values.dtype
    )
    for i, (layer, row, end, n) in enumerate(rows):
        if n and layer.keys is not None and layer.values is not None:
            keys[i, :, length - n : length, :] = layer.keys[row, :, end - n : end, :]
            values[i, :, length - n : length, :] = layer.values[row, :, end - n : end, :]
    return keys, values
