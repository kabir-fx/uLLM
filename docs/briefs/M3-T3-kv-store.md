# M3-T3: KV store behind its interface

**State:** in build. You approved the design and its three recommended answers.
**Decision record:** [0034](../decisions/0034-kv-store.md) (the full design, for agents) · **Spec:** roadmap E9 and Q3, task M3-T3

## The goal

The engine will keep each request's KV in 16-token blocks, as decision 0033 chose. Replies stay token for token the same as with the contiguous KV of M2.

## The problem

- The `BlockManager` from M3-T1 counts blocks, but no tensor sits behind a block yet.
- M3-T5 (prefix caching) and M3-T6 (preemption) need KV in blocks. Requests can then share it and gather it again.
- The block tables live in `ullm.core`, which never imports MLX (rule 3). The tensors live in the runner. The runner protocol C3 carries no block ids now.
- Rule 1 asks for equivalence "block-based against contiguous". Thus the E7 layout must stay as a reference.
- Rule 2: only the engine thread touches the store. Rule 4: one `mx.eval` at the end of each step.

## How it works

```
 core (MLX-free)                          runner (MLX)
 Scheduler ── BlockManager                MLXRunner ── KVStore
    │  allocate, append_slot                  │          ├─ BlockKV      (B, default)
    │                                         │          └─ ContiguousKV (E7, reference)
    └── PrefillPlan(blocks=table) ──────────► │
        DecodePlan(blocks=tables) ──────────► │
        release(seq, freed) ────────────────► │
```

1. `RunnerInfo.block_size`, in `src/ullm/runner/base.py`, tells the core the block size. The core builds its `BlockManager` from it.
2. `Scheduler`, in `src/ullm/core/scheduler.py`, allocates blocks at admission. Before each decode it gives each row room for one token. It puts each table in the plan.
3. `BlockManager.free` returns the blocks that became Free. The engine passes them to `runner.release`.
4. `KVStore`, in `src/ullm/runner/kv/store.py`, holds every KV tensor of the runner. `MLXRunner` asks it for the cache of each step.
5. `BlockKV` cuts a prefill's KV into blocks and drops the prefill cache. A row that joins the batch is gathered from its blocks. When a block fills during decode, `BlockKV` copies it out of the working tensor.
6. `ContiguousKV` is the M2 code, moved out of `MLXRunner`. Only the equivalence test uses it.

The pool has `ceil(max_kv_slots / 16) + max_num_seqs` blocks. Admission still uses slots (decision 0020), so the pool cannot run out. M3-T4 changes admission to free blocks.

## What you get at the end

- `make equivalence` runs the engine at batch sizes 1, 2, 4 and 8 with each layout. Both must match mlx-lm. The block layout must also give exactly the tokens of the contiguous layout.
- A new property test, `tests/property/test_kv_store.py`, runs random prefills, chunks, joins, decodes and releases on both layouts. It checks that each sequence keeps exactly its own KV. It also checks that `BlockKV` drops every freed block.
- New ledger tests prove that a plan with a wrong table length fails before any forward.
- `make bench-smoke` before and after, on AC power. The difference in TTFT shows the cost of the block cut and the gather.

What you do not get yet:

- Admission by free blocks, a pool sized from memory, and the block metrics: M3-T4.
- Shared blocks and `cached_tokens`: M3-T5. Preemption: M3-T6.

## Where this puts the project

```
M3  ██▓░░░░░░
```

This slice opens M3-T4 (admission by free blocks) and M3-T5 (prefix caching).

## What you approve

1. **The core sends block tables to the runner, in a C3 v2.** This option beat a runner that picks its own block ids. With runner ids, M3-T5 must move the ids to the core anyway. Until then, two pools count the same memory.
2. **The engine builds its `BlockManager` now, one task before M3-T4.** This changes decision 0032 point 8. The runner cannot put KV into blocks that nobody allocated.
3. **A prefill's KV goes into blocks at once, and a joining row is gathered from them.** This option beat a prefill cache that stays until the join. That cache skips the store, so the equivalence test never reads it.
4. **Each decode plan carries the whole table of each row.** This option beat "only the last block". With only the last block, the runner must keep a copy of each table.
5. **`ContiguousKV` stays as production code, with no CLI flag to select it.** Rule 1 keeps it as the reference. Only tests select it.

Open questions:

- Do you accept the move of the first `BlockManager` into the engine from M3-T4 to M3-T3? I recommend yes.
- `release` gets a `freed` argument, which the caller must pass. Do you accept a required argument instead of a default `()`? I recommend required, because a default lets a caller forget it, and memory then leaks.
- The laptop is on battery now. The base `make bench-smoke` must run on AC before the first commit. Can you connect AC power? I recommend yes.

## What can go wrong

- The cut into blocks adds small copies to each prefill: 56 arrays per block. A 1,024-token prompt has 64 blocks. If TTFT in `make bench-smoke` gets clearly worse, the agent will report it before going on.
- While decoding, `BlockKV` holds about 1.78 times the KV of E7 (decision 0033). M3-T4 must count both copies.
- A wrong table length in a plan fails the step, not the server. A pool that runs out is a core bug, and the engine thread then stops.

## Words

| Word | Meaning |
|---|---|
| block table | The list of block ids that hold one sequence's KV, in token order. |
| working tensor | The contiguous `[rows, heads, tokens, dim]` KV of the running batch (E7). |
| store of record | The blocks. A sequence's KV comes back from them when it joins a batch. |
| `freed` | The blocks that became Free when the core freed a sequence. |

## Learn from this slice

The core owns decisions and the runner owns tensors. A block id is a decision, so the core makes it and the plan carries it. The runner only maps an id to arrays.

Two adapters make a seam real. `BlockKV` and `ContiguousKV` fill the same `KVStore` interface. The equivalence test swaps one for the other and compares tokens.

**Test yourself:** why can the pool not run out in M3-T3, when admission still counts slots?

<details><summary>Answer</summary>

Admission holds rows × the longest reservation to `max_kv_slots`. Each row needs `ceil(tokens / 16)` blocks, which is under `tokens / 16 + 1`. The sum is thus under `max_kv_slots / 16 + max_num_seqs`, which is the pool size.

</details>

## Result
