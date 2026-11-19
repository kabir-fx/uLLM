# M2-T1: Contiguous batch KV

**State:** approved. The author took the recommended answer to each open question. Built by `/build M2-T1`.
**Decision record:** [0019](../decisions/0019-contiguous-batch-kv.md) (the full design, for agents) · **Spec:** roadmap E7, task M2-T1

## The goal

The real model will decode many requests in one forward pass. This slice gives the runner one KV cache for the whole running batch.

## The problem

- `MLXRunner` refuses a decode of more than one sequence. Each sequence has its own cache, so the runner cannot build one `[B, 1]` forward.
- The current cache, `ContiguousKV`, copies all of its K and V on each new token. At batch 8 and 1,024 tokens, a decode step took 114 ms with that copy.
- Rule 1: at batch size 1, the tokens must stay equal to mlx-lm's tokens, with no allowance.
- Rule 11: the cache is our own code. The agent can read mlx-lm's `BatchKVCache`, but the engine cannot use it.
- E9 will measure other KV layouts against this one. Thus this cache must be fast enough to be a fair reference.

## How it works

```
Scheduler (core, no change)          MLXRunner
+----------------------------+       +-------------------------------------------+
| prefill one new request    | ----> | _prefilling: one BatchKV of 1 row per seq |
| decode every running seq   | ----> | _batch: one BatchKV, one row per seq      |
+----------------------------+       |   a new seq in a decode: repack (join)    |
                                     |   release of a seq: repack (leave)        |
                                     +-------------------------------------------+

One layer of _batch, B = 3, length 6 (p = padding, x = token, . = free space):
  row 0   p p x x x x . .
  row 1   x x x x x x . .
  row 2   p p p p x x . .
```

1. **`BatchKV`** (`src/ullm/runner/kv/batch.py`). It holds the K and V of a fixed set of sequences, one row for each sequence. Shorter rows get padding on the left, so all rows end at the same position. It keeps free space after the tokens, so a new token goes into the array in place. `BatchKV.pack` is the only way to change the set of rows. It copies the rows into one new array for each layer.
2. **`LayerKV`** (same file). It is the object that mlx-lm's model calls for each layer. It gives each row its own position for RoPE, and a mask that hides the padding.
3. **`MLXRunner`** (`src/ullm/runner/mlx_runner.py`). A new sequence prefills into its own `BatchKV` of one row. At its first decode, the runner packs it into the batch. At `release`, the runner packs the batch again without that row.
4. **`SeqLedger`** (`src/ullm/runner/ledger.py`). It checks each plan before the runner does any work. It gets two new rules for both runners. A decode must name every sequence that decodes now. A prefill is legal only before the first decode of its sequence.

**Facts that the agent measured first.** The agent built a throwaway prototype and ran it on your M4. It ran the 10 golden prompts, 128 tokens each, at batch sizes 1, 2, 4 and 8.

- At batch size 1, all 10 prompts gave the same tokens as mlx-lm.
- At larger batches, 8 or 9 of 10 prompts matched. Each difference was at a near-tie, with a margin of 0.0 or 0.25. Three different cache layouts gave the same tokens. Thus the batched forward causes the differences, not the cache.
- A decode step at batch 8 and 1,024 tokens took 33 ms with writes in place. It took 114 ms with a copy on each token.
- One change of the rows costs 23 to 87 ms at batch sizes 7 to 15.

## What you get at the end

- `MLXRunner` decodes up to `max_num_seqs` sequences in one forward pass. The engine does not change. M2-T2 lets the scheduler run more than one request at a time.
- A property test makes random joins, appends and releases. After each step, each row must hold exactly the K and V of its own sequence.
- Conformance tests on both runners check the two new protocol rules.
- A test on the real model checks that each sequence gets the same tokens in each plan order.
- The equivalence tests at batch size 1 stay exact.
- The pull request shows `make bench-smoke` before and after the change.

What you do not get yet:

- Equivalence tests at batch sizes 2, 4 and 8. M2-T3 adds them, with a measured near-tie threshold.
- Continuous batching in the engine. M2-T2 adds it.
- Blocks, a block table, and less padding. M3 brings them (E8, E9).

## Where this puts the project

```
     done █   this slice ▓   not started ░

M0   █████████       9 of 9
M1   ████████        8 of 8
M2   ▓░░░░░░░░░░░░   0 of 13, then 1 of 13 after this slice
```

This slice opens M2-T2, the scheduler. Then M2-T3 can test equivalence at batch sizes above 1.

## What you approve

1. **The protocol gets stricter.** A decode must name every sequence that decodes now. A prefill must come before the first decode. The other option was a runner that accepts any set and moves rows itself. No caller needs it, it hides a copy, and a failed step can then change sequences outside its plan. **This changes invariant 2 of decision 0008 and the C3 text in `runner/base.py`.**
2. **One repack for each join and each leave.** The other option was mlx-lm's method, which removes a row with a view. That method is cheaper, but later writes into a view were not measured. One operation for both directions keeps one code path.
3. **A sequence joins at its first decode, not at the end of its prefill.** A request can end on its first token. With an early join, it pays for a join and a leave for nothing.
4. **No `KVStore` interface now.** One adapter cannot prove an interface. E9 adds the interface when it adds a second layout.

Open questions, each with the recommended answer:

1. Can the agent delete `kv/contiguous.py` and `ContiguousKV`? **Recommended: yes.** Two caches would give two ways to do one task.
2. Where does the near-tie threshold for batches come from? **Recommended: M2-T3, as decision 0012 says.** This slice records the 0.0 and 0.25 margins as input. Note that 0.25 is equal to the allowance for other chips.
3. Is the stricter protocol acceptable in C3? **Recommended: yes.** E6, E11 and E13 all fit inside it.
4. Is a decode spike on each join and leave acceptable for M2? **Recommended: yes.** E9 is the slice that reduces it, and it needs this cost as its baseline.

## What can go wrong

- Each join or leave copies the whole batch. You will see a spike of 23 to 87 ms on the step chart.
- During a repack, the old and the new batch exist at the same time. The KV memory is then about two times the batch.
- Padding stays until the next repack. One long prompt makes every row as long as that prompt.
- When the rows have different lengths, each decode uses a mask array. In one run, it cost 50 ms where no mask cost 33 ms.
- At batch sizes above 1, a few tokens can differ from mlx-lm at near-ties. M2-T3 decides how its test treats them.

## Words

| Word | Meaning |
|---|---|
| batch KV | One K array and one V array for each layer, with one row for each running sequence. |
| left padding | Empty slots at the start of a shorter row, so all rows end at the same position. |
| repack | A copy of a set of rows into new arrays, to add or remove a row. |
| prefilling | The phase of a sequence from `add` to its first decode. |
| decoding | The phase of a sequence from its first decode to `release`. |
| near-tie | A position where the two most likely tokens have almost the same logit. |

## Learn from this slice

**Write in place, copy on change.** A decode adds one token to each row, and this occurs at each step. A change of the rows occurs only when a request starts or ends. Thus the design makes the frequent operation cheap and accepts a copy for the rare operation.

**Padding on the left keeps the end aligned.** All rows write their next token at the same index. Each row still needs its own position for RoPE, so `offset` is a vector. The mask hides the padding from attention.

Test yourself: the KV data is the same. Why can a batch of eight then select a different token at a near-tie?

<details><summary>Answer</summary>
The forward pass runs its matrix operations at a different shape. In bf16, that changes the logits by a small amount. At a near-tie, that small change can select the other token. The three cache layouts gave the same tokens, so the cache does not cause it.
</details>

## Result

**State: DONE.** Proof VERIFIED and review READY at `ab2db89`. The fix commit `fd75b65` is covered: its test fails without the fix and passes with it. `make ci` is green at `fd75b65`.

**Against the plan.** The build follows decision 0019 with no change to the design. Two small additions:
- `BatchKV` keeps a host copy of its padding, so `pack` never reads from the GPU.
- Capacity rounds up past the length to a multiple of 256. Thus the step after a repack has a free slot.

**What proof and review found.**
- Two reviewers found the same gap: the test of a failed batch ran only on `FakeRunner`. It now runs on both runners.
- Before the fix, deleting `MLXRunner`'s reset of the batch after a failed decode left every test green.
- After the fix, the same deletion fails the test with `KeyError` at `mlx_runner.py:208`.

**Not proven here.**
- Batched output against mlx-lm at batch sizes 2, 4 and 8. That is M2-T3.
- A mask that ignores padding survives the plan-order test. Only a comparison with batch-size-1 output caught it (the verifier's mutant). Thus M2-T3's batched equivalence is the end-to-end guard for the mask. The property test checks the mask values, and the plug-point test checks that the model uses them.
- Temperature sampling inside a batch.

**Measured.** `make bench-smoke`, both runs on battery:

| Run | Commit | Output tok/s | TPOT p50 (ms) |
|---|---|---|---|
| Before | `229304b` | 52.4 | 17.7 |
| After | `ab2db89` | 54.5 | 17.5 |

The near-tie margins at batch sizes above 1 were 0.0, 0.125 and 0.25.

**Passes to later slices.**
- M2-T2 should limit padded KV slots before `max_num_seqs` goes above 1. One prompt near 40,960 tokens at batch 8 needs about 37.6 GB of KV (inferred).
- M2-T3 sets the near-tie threshold for batched equivalence.

**Read these yourself.**
- `src/ullm/runner/mlx_runner.py:196`: the join packs the batch, and only then are the joiners removed from `_prefilling`. Tokens return in plan order.
- `src/ullm/runner/kv/batch.py:93`: `pack` right-aligns each row and computes the pads.
- `src/ullm/runner/ledger.py:60`: the two new rules run before any forward.
