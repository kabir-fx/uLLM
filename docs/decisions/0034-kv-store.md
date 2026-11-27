# 0034: The KV store behind `KVStore`; plans carry block tables (C3 v2)

**Status:** accepted · **Task:** M3-T3 (E9, Q3)

## Context

Decision 0033 chose layout B: E7's contiguous working tensor for the running batch, plus a block store of record. A block is copied out when it fills, and a joining row is gathered from its blocks. Its point 4: the block manager (decision 0032) owns the tables, and the store maps each block id to its K and V.

The manager lives in `ullm.core` and is MLX-free (rule 3). The store holds MLX arrays, so it lives in the runner. C3 (decisions 0003, 0008) is the only channel between them, and today it carries no block ids. Q3 and rule 1 require an equivalence variant "block-based against contiguous", so E7 stays as a second, selectable KV.

Decision 0032 point 8 said M3-T4 is where the engine first builds a manager. This record moves that to M3-T3, because the runner cannot place KV in blocks that nobody allocated. Admission by free blocks and sizing from a memory budget stay in M3-T4.

## Usage

The core, on the engine thread:

```python
# admission (Scheduler.schedule)
self._blocks.allocate(req.seq, len(req.prompt))
plan = req.prefill_plan(blocks=self._blocks.table(req.seq))
# PrefillPlan(seq=3, token_ids=(...40 ids...), start=0, blocks=(7, 8, 9))

# a decode step
for r in rows:
    self._blocks.append_slot(r.seq)
plan = DecodePlan(seqs=..., token_ids=..., blocks=tuple(self._blocks.table(r.seq) for r in rows))

# an ended request (Engine._forget)
freed = self._scheduler.remove(req)  # the blocks that became Free
self._runner.release(req.seq, freed)
```

The runner, inside `MLXRunner.execute`:

```python
cache = self._kv.prefill(plan.seq, plan.start, plan.blocks)  # list[LayerKV]
logits = self.model(x, cache=cache)
mx.eval(sampled, *self._kv.written())

batch = self._kv.decode(dict(zip(plan.seqs, plan.blocks)))  # BatchKV, joiners gathered in
logits = self.model(x, cache=batch.layers)
mx.eval(sampled, *self._kv.written())
```

A test that compares the two layouts:

```python
blocks, _ = mlx_loaded  # load_mlx gives the block store
contiguous = MLXRunner(blocks.model, blocks.info(), ContiguousKV(len(blocks.model.layers)))
```

## Shape

**C3 v2** (`ullm.runner.base`, MLX-free):

- `BlockId = int` moves here from `ullm.core.block_manager`, which imports it. The runner must not import the core.
- `RunnerInfo.block_size: int`. The runner states its block size, and the core builds its manager from it, so the two cannot disagree.
- `PrefillPlan.blocks: tuple[BlockId, ...]`: the sequence's whole table after this step.
- `DecodePlan.blocks: tuple[tuple[BlockId, ...], ...]`: each row's whole table after this step, in plan order. Whole tables keep the runner free of table state: a joining row is gathered from its table, and a filled block goes to `table[-1]`.
- `Runner.release(seq, freed: tuple[BlockId, ...])`: the blocks that became Free when the core freed `seq`. A runner drops their contents. A Cached block (M3-T5) keeps them, because it is not in `freed`.
- `SeqLedger` checks, before any forward: every block id is an int, a table has `ceil(tokens after the step / block_size)` entries, and one `blocks` entry per row. A violation raises `ValueError` and changes nothing (invariant 2).

**Core:**

- `BlockManager.free(seq)` returns the blocks that became Free.
- `Scheduler` owns a `BlockManager(num_blocks, info.block_size)`. It allocates at admission, appends a slot for each row before a decode, and puts the tables in the plans. `Scheduler.remove(req)` returns the freed blocks, `()` for a waiting request.
- `num_blocks = ceil(max_kv_slots / block_size) + max_num_seqs`. Admission still holds rows × the longest reservation to `max_kv_slots` (decision 0020), and each row wastes under one block, so allocation cannot fail. If it does, `OutOfBlocks` is a core bug and the engine thread dies, as for any core bug. M3-T4 replaces this with admission by free blocks and a pool sized from memory.

**Runner** (`ullm.runner.kv.store`, MLX):

```python
class KVStore(Protocol):
    """Every KV tensor of a runner. Engine thread only (rule 2)."""

    def prefill(self, seq: SeqId, start: int, blocks: tuple[BlockId, ...]) -> list[LayerKV]:
        """The cache a prefill of ``seq`` runs on, holding its first ``start`` tokens."""

    def decode(self, tables: Mapping[SeqId, tuple[BlockId, ...]]) -> BatchKV:
        """The working batch of exactly these sequences. One not in it yet joins."""

    def written(self) -> list[mx.array]:
        """What the last prefill or decode wrote. Call after the forward, before its eval."""

    def release(self, seq: SeqId, freed: tuple[BlockId, ...]) -> None: ...

    def fail(self) -> None:
        """Drop the working batch and anything not yet written: a step raised."""
```

Two adapters:

- **`ContiguousKV`** (E7). Today's code moved out of `MLXRunner`: a one-row `BatchKV` per prefilling sequence, packed into the working batch at its first decode. It ignores block ids. It is the reference for Q3.
- **`BlockKV`** (B, the default from `load_mlx`). A dict `BlockId → per-layer (K, V)`, each `[n_kv_heads, block_size, head_dim]`, one array each, the shape measured in M3-T2.
  - `prefill`: a fresh one-row cache. With `start > 0`, it is gathered from the first `ceil(start / block_size)` blocks, cut to `start` tokens. `written` then cuts every block the prefill touched, `start // block_size` onward, out of the cache with `mx.contiguous`, the partial last block included, and the cache is dropped. Between steps, a prefilling sequence's KV is only in its blocks.
  - `decode`: each joiner is gathered from its table into a one-row `BatchKV`, then `BatchKV.pack` builds the working batch as E7 does. `written` copies `table[-1]` out of the working tensor for each row whose length reached a multiple of `block_size`.
  - `release`: the row leaves the working batch by a repack, as in E7, and the `freed` blocks are dropped.
  - Invariant: every full block of a live sequence is in the store and equals its KV. A decoding sequence's partial last block is only in the working tensor.

`MLXRunner` keeps sampling, the ledger, timings and the one `mx.eval` per step (rule 4). It holds no KV of its own. `prefill` with `start > 0` is the path M3-T5's prefix hits and M3-T6's re-prefill take, with shared blocks in the table. In M3-T3 only a chunked prefill (`sample=False`) takes it, and only tests send one.

What the store does not do: decide which blocks a sequence gets, count free blocks, or know Cached from Free. The core decides all three.

## Failure and operation

- **Twice.** A second `add`, `release` or prefill of a decoding sequence raises in the ledger, before the store sees it.
- **A raise halfway.** The ledger fails every sequence in the plan (invariant 4). `MLXRunner` calls `fail()`: the working batch and the pending block copies go. A failed sequence's blocks may be stale. Its only legal call is `release`, which drops the `freed` blocks.
- **Concurrent writers.** None: one engine thread (rule 2). A shared block (M3-T5) is full, so no decode writes into it (decision 0032 point 3).
- **Trust boundary.** Plans come from the core, not from clients. The ledger checks types and table lengths. It does not check that ids are in range or unshared: the manager's property tests own that.
- **In production.** Equivalence (Q3) at batch 1, 2, 4 and 8 for both adapters, and B's tokens equal to E7's. `make bench-smoke` before and after. MLX's memory after a long run (the existing `test_a_long_request_leaves_little_freed_memory_held_for_reuse`) must hold with the store.

## Trade-offs accepted

- We accept about 1.78 times E7's KV memory while decoding (decision 0033) in exchange for a store of record that M3-T5 and M3-T6 can share and re-gather.
- We accept a copy at every prefill (the cut into blocks) and a gather at every join in exchange for one place per prefilling sequence's KV. The cost shows in TTFT in `make bench-smoke`.
- We accept whole tables in every decode plan, O(total blocks) small ints per step, in exchange for a runner with no table state to drift from the manager's.
- We accept that the engine builds a manager one task before admission uses it, with a pool sized from slots, in exchange for real block ids from the first commit.

## Alternatives considered

- **The runner owns its block ids (no C3 change).** `BlockKV` allocates ids from its own free list, keyed by sequence and index. It lost because M3-T5 must then move the ids to the core anyway, and until then two pools would count the same memory (split ownership).
- **Only the last block per row in `DecodePlan`.** A smaller plan, but the runner must then remember each prefilled sequence's table to gather it at the join: a copy of the manager's state.
- **Keep the prefill cache until the join, as E7 does, and only copy blocks out.** It skips the gather, so the equivalence would never read the store. It also holds two copies of every waiting prompt's KV.
- **A write-through layer wrapper (the spike's `WriteThroughLayer`) instead of `written()`.** It records copies inside `update_and_fetch`, 28 times a step. `written()` does the same once per step, after the forward, with `LayerKV` unchanged.

## Consequences

- C3 is v2. `FakeRunner`, `RecordingRunner` and the `sim` runner take `freed` and report `block_size`.
- `MLXRunner(model, info, kv)` takes its store. `load_mlx` passes `BlockKV`. No CLI flag selects the layout: only the equivalence test needs E7.
- Decision 0032 point 8 changes in this one respect: the engine builds its manager in M3-T3. Sizing from a memory budget and the metrics stay in M3-T4.
