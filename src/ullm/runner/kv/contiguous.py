"""One layer's K and V for one sequence, grown by concatenation (E1, decision 0008).

It plugs into mlx-lm's model modules through ``offset``, ``update_and_fetch``
and ``make_mask`` (roadmap section 4). Every append copies the sequence's
whole K and V, so decode cost grows with context; E7 replaces it in M2-T1.
"""

import mlx.core as mx


class ContiguousKV:
    def __init__(self) -> None:
        self.offset = 0
        self.keys: mx.array | None = None  # [1, n_kv_heads, offset, head_dim]
        self.values: mx.array | None = None

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        if self.keys is None or self.values is None:
            self.keys, self.values = keys, values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
        self.offset += keys.shape[2]
        return self.keys, self.values

    def make_mask(
        self, N: int, return_array: bool = False, window_size: int | None = None
    ) -> str | None:
        """mlx-lm's mask for a plain KV cache: none for one query, else causal."""
        if return_array or window_size is not None:
            raise ValueError("ContiguousKV supports only full causal attention")
        return None if N == 1 else "causal"
