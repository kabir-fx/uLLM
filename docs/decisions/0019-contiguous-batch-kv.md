# 0019: Contiguous batch KV, and a runner protocol where a batch decodes together

**Status:** accepted (approved by the author before implementation, with the recommended answer to each open question) · **Task:** M2-T1 `[design]` (E7)

## Context

`MLXRunner` refuses a `DecodePlan` of more than one sequence (decision 0008). Each sequence owns 28 `ContiguousKV` objects that concatenate on every append. M2-T2 lets the scheduler run up to `max_num_seqs` sequences at once, so the runner needs one KV cache for the whole running batch (E7). E7 is also the reference that E9's layouts are measured against, so it must be fast enough to be a fair reference, and simple enough to read.

Constraints:

- Rule 1: greedy token IDs equal mlx-lm's. At batch size 1 there is no near-tie allowance (decision 0012). Above 1, M2-T3 sets the threshold from a measurement.
- Rule 2: only the engine thread calls the runner. No locks.
- Rule 11: our own cache, not `BatchKVCache`. Reading it is allowed, and it is the baseline.
- C3 and decision 0008: the core owns every decision, the runner owns every tensor. `SeqLedger` enforces the protocol's invariants once, for `MLXRunner` and `FakeRunner` alike.
- DEC-7 and E6: one prefill of one sequence, `[1, L]`, or one decode of every running sequence, `[B, 1]`. A step never mixes them.

**The plug point** (mlx-lm 0.32.2, read in `.venv`): `qwen3.py` calls `rope(q, offset=cache.offset)` and `rope(k, offset=cache.offset)`, then `cache.update_and_fetch(keys, values)`. `mx.fast.rope` takes `offset` as an int or a vector of `B` offsets. The mask is built once, before the layers, from `cache[0].make_mask(N, ...)`. `create_causal_mask(N, offset, left_padding=...)` masks each row's padding.

**Measured** on the author's M4 (throwaway scripts in the session scratchpad, not committed; reproduce with the prototype described in Decision). E6-like scheduling: admit one prompt per step and prefill it alone, then decode every running row; a row leaves at EOS or 128 tokens. All 10 golden prompts, greedy, against `reference()` on the same machine.

| Variant | B=1 | B=2 | B=4 | B=8 |
|---|---|---|---|---|
| Preallocated, in-place writes; no mask when no row is padded | 10/10 | 9/10 | 9/10 | 8/10 |
| Concatenate on every append | 10/10 | 9/10 | 9/10 | 8/10 |
| Array mask on every step, padded or not | 10/10 | 9/10 | 9/10 | 8/10 |

Every divergence is at a near-tie: `palindrome` at token 20 (mlx-lm's top-2 margin 0.0) for B ≥ 2, and `kv-cache` at token 32 (margin 0.25) at B=8. The three layouts give the same tokens, so the divergence comes from the batched forward, not from how K and V are stored (inferred: the matmuls run at a different shape).

Decode step time, Qwen3-0.6B, every row at context T, 40 steps, median:

| | B=1, T=1024 | B=8, T=512 | B=8, T=1024 |
|---|---|---|---|
| Preallocated, in-place writes | 21.8 ms | 34.0 ms | 33.0 ms |
| Concatenate on every append | 46.4 ms | 87.2 ms | 114.4 ms |
| Preallocated, array mask always | 24.7 ms | 37.1 ms | 50.1 ms |

The cost of one membership change, copying every row into one fresh allocation per layer for K and V (28 layers), median of 10:

| Batch before | Add one row | Remove one row |
|---|---|---|
| B=1, T=512 | 6.8 ms | 3.3 ms |
| B=7, T=512 | 23.4 ms | 14.6 ms |
| B=7, T=1024 | 73.2 ms | 25.3 ms |
| B=15, T=1024 | 86.8 ms | 57.2 ms |

The "remove" column took rows with `mx.take` and sliced off the common padding, which leaves a strided view. Writing into a view in place is not measured; the design avoids it by repacking.

## Decision

**Usage.** The scheduler and engine do not change. They already send what E7 needs: one prefill per admitted sequence, then decodes that name every running sequence (`Scheduler.schedule`, `src/ullm/core/scheduler.py`). The runner, as the engine drives it:

```python
runner.add(1, sampling)
runner.execute(PrefillPlan(seq=1, token_ids=p1, start=0))
runner.execute(DecodePlan(seqs=(1,), token_ids=(t1,)))  # batch: [1]
runner.add(2, sampling)
runner.execute(PrefillPlan(seq=2, token_ids=p2, start=0))
runner.execute(DecodePlan(seqs=(1, 2), token_ids=(t1, t2)))  # 2 joins: one repack
runner.execute(DecodePlan(seqs=(2, 1), token_ids=(t2, t1)))  # any order; no copy
runner.release(1)  # 1 leaves: one repack

runner.execute(DecodePlan(seqs=(1,), token_ids=(t1,)))  # with 1 and 2 decoding:
# ValueError: decode plan omits decoding sequences [2]; a batch decodes together (0019)
runner.execute(PrefillPlan(seq=2, token_ids=more, start=held))
# ValueError: sequence 2 is decoding; a prefill comes before its first decode (0019)
```

**Protocol change (C3, invariant 2 of `runner.base`).** A sequence has two phases: *prefilling* from `add` to its first decode, then *decoding* until `release`. Two new rules:

1. A `DecodePlan` names every decoding sequence that has not failed, in any order. It may also name prefilling sequences; they join the batch.
2. A `PrefillPlan` is legal only while its sequence is prefilling.

`SeqLedger` keeps the phase and checks both rules before any forward pass, so `FakeRunner` refuses exactly what `MLXRunner` refuses, and a violation changes nothing. Its per-sequence entry becomes `_Live(held: int, decoding: bool)`, or `None` once failed. After a successful decode, every sequence in the plan is decoding. This amends decision 0008's invariant 2; the rest of 0008 stands.

**Shape** (`src/ullm/runner/kv/batch.py`, replaces `kv/contiguous.py`):

```python
STEP = 256  # capacity grows in steps of this many tokens, as mlx-lm's caches do


class LayerKV:
    """One layer's K and V for every row of a ``BatchKV``: mlx-lm's cache plug point."""

    keys: mx.array  # [B, n_kv_heads, capacity, head_dim]; row i holds pad_i zeros, then its tokens
    values: mx.array
    length: int  # slots written, the same for every row; capacity >= length
    pads: mx.array  # [B] int32, shared by every layer of one BatchKV; never written

    @property
    def offset(self) -> mx.array: ...

    # [B]: length - pads, each row's position for RoPE.

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]: ...

    # Writes [B, H, L, D] at length in place, growing capacity by STEP first when full,
    # advances length by L, and returns keys[..., :length, :] and values[..., :length, :].

    def make_mask(
        self, N: int, return_array: bool = False, window_size: int | None = None
    ) -> mx.array | str | None: ...

    # No row padded: None for N == 1, else "causal", as mlx-lm's plain KVCache (bit-identical
    # at batch size 1). Any row padded: create_causal_mask(N, offset=length, left_padding=pads).
    # return_array and window_size raise ValueError, as ContiguousKV does today.


class BatchKV:
    """The KV of a fixed set of sequences, one row each, left-padded to one length."""

    seqs: tuple[SeqId, ...]  # row order: the order rows were packed, not plan order
    layers: list[LayerKV]  # model(x, cache=batch.layers)

    @classmethod
    def empty(cls, seq: SeqId, n_layers: int) -> "BatchKV": ...

    # One row with no tokens and pad 0: where a sequence prefills.

    @classmethod
    def pack(cls, rows: Sequence[tuple["BatchKV", SeqId]]) -> "BatchKV": ...

    # The only membership operation. Copies each named row, right-aligned to the longest,
    # into one fresh [B, H, capacity, D] allocation per layer for K and V; the new pads are
    # each row's distance from the longest, so common padding is gone. The sources are not
    # changed. Raises ValueError on an empty list, a repeated seq, or a seq not in its source.

    def row(self, seq: SeqId) -> int: ...
```

**`MLXRunner`** keeps each sequence's KV in exactly one place:

- `_prefilling: dict[SeqId, BatchKV]`: one single-row `BatchKV` per added sequence that has not decoded.
- `_batch: BatchKV | None`: every decoding sequence, one row each, or `None` when none decodes.
- A prefill runs on `_prefilling[seq].layers`, with the `[L-1]` then `[1]` split of decision 0008.
- A decode first joins: if the plan names prefilling sequences, `_batch = BatchKV.pack(rows of _batch, then the joiners)`, and only then are the joiners removed from `_prefilling`. It builds the input `[B, 1]` in `_batch.seqs` order, samples each row with its own `SampleRow`, and returns the tokens in plan order.
- `release` drops the sequence from `_prefilling`, or repacks `_batch` without its row (`None` if it was the last). A failed sequence may have no KV left; `release` then frees only its ledger entry.
- A decode that raises after validation sets `_batch = None`. Rule 1 above means every row in it was in the plan, and the ledger has failed them all.
- One `mx.eval` per step, then the timings (rule 4). Join cost lands inside the decode step that joins, so `forward_ms` and the `step` event show it.
- `CACHE_LIMIT` stays (decision 0015): each repack frees buffers of a size that does not recur.

**`FakeRunner`** changes only through `SeqLedger`.

**Tests** (written first, in `/build`):

- `tests/property/test_batch_kv.py` (MLX, no model; `mlx` mark): hypothesis drives random sequences of prefill, decode-append, join and release over small random K and V (2 layers, 2 heads, head dim 4). After every operation, each row's unpadded K and V equal a per-sequence concatenation bit for bit; each row's `offset` is its own length; the mask hides exactly the padding and the future; no padding is common to every row.
- `tests/unit/test_runner_conformance.py`, both runners: a decode that omits a decoding sequence raises and changes nothing; a prefill after the first decode raises; a decode may name a prefilling sequence with decoding ones; after a failed batch, a sequence that never joined decodes alone (the existing test stays).
- `tests/unit/test_mlx_runner.py`, real model: the "M2-T1" refusal test goes. Instead: a batch of 4 returns the same tokens per sequence whatever the plan order; the tokens come back in plan order.
- The existing equivalence tests stay exact and now run through `BatchKV` at batch size 1. Batch sizes 2, 4 and 8 against mlx-lm are M2-T3's, with its threshold (decision 0012).
- `make bench-smoke` on the base and after; both summaries in the pull request.

## Alternatives considered

- **The runner reconciles any decode set.** A decode that omits a decoding sequence would park its row in a single-row `BatchKV`; a prefill after decode would extract the row first. This keeps C3's literal "any set of sequences", but no caller needs it: E6 decodes every running sequence, E11 releases a preempted sequence, and E13 prefills before decode. It would hide a copy behind an innocent plan, and a failed step would change sequences that were not in the plan, which breaks invariant 4.
- **The core plans membership** (`DecodePlan` with `joins` and `leaves`, or a `KVStore` the core calls). It shows the core a fact about one physical layout. E9's option A gathers every step and has no membership at all, so the contract would change with the layout.
- **Concatenate on append, as `ContiguousKV` does.** Same tokens, but 3.5 times slower at B=8 and T=1024 (114.4 ms against 33.0 ms). As the reference for E9, it would make every candidate look good.
- **Per-row views after a removal** (`mx.take` and a slice, as `BatchKVCache.filter` does). Cheaper to remove a row (25.3 ms against a full repack at the same size), but every later in-place write lands on a strided view, which is not measured. One `pack` for join and leave keeps one code path.
- **A `KVStore` interface now.** One adapter makes the seam hypothetical. E9 adds it when it adds the second.
- **Join at the end of the prefill step.** Same copy, earlier. A request that ends on its prefill token (EOS or `max_tokens: 1`) would pay a join and a leave for nothing.

## Consequences

- Every membership change copies the whole batch: 23 to 87 ms measured at B=7 to 15. With E6 admitting one request per step, each admission costs one prefill and one repack, and each finish one repack. The step chart will show decode spikes where rows join or leave. E9 is measured against this.
- During a repack the old and new batch both exist, so peak KV memory is about twice the batch until the step's eval.
- Rows are padded to the longest row, and the padding stays until a repack. A batch of one 8k-token prompt and seven short ones holds 8 × 8k slots. E8 and E9 exist for that waste.
- Once any row is padded, every decode uses an array mask. It cost 50.1 ms against 33.0 ms at B=8 and T=1024 in one run; the mask is always on in real batching, where lengths differ.
- At batch size above 1, greedy output diverges from mlx-lm's single-request output at near-ties, margins 0.0 and 0.25 measured. M2-T3 must choose its threshold knowing that 0.25 occurs, which equals the cross-machine allowance of decision 0012. Another target M2-T3 may weigh: mlx-lm's own batched output for the same batch composition.
- The decode step time at batch size 1 should fall: 21.8 ms against 46.4 ms at T=1024 in this spike, the same direction decision 0015 measured for preallocation. `make bench-smoke` before and after will give the committed figure.
- `ContiguousKV` and `kv/contiguous.py` are deleted, so one cache class exists.
