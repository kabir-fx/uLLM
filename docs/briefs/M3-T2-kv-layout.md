# M3-T2: KV layout on the real model

**State:** in build. Read the decision record and the results, then wait for the Result.
**Decision record:** [0033](../decisions/0033-kv-layout.md) (the full design, for agents) · **Spec:** roadmap E9, task M3-T2

## The goal

This spike measures which physical KV layout the block store will use. It changes nothing in the server.

## The problem

- The block manager (M3-T1) counts blocks, but no KV tensors live in blocks yet.
- Roadmap section 4 measured this: a copy of all KV in each step costs more than a decode step.
- E9 names three layouts. Its rule takes A, then B: the first within about 20% of E7 at 8 rows and 1,024 tokens.
- Rule 7: every figure comes from a committed result.

## How it works

```
 workload A prompt --prefill once--> source KV
                                        |
        +-------------+-------------+---+---------+
        v             v             v             v
       E7             A             A2            B
   contiguous    block arrays   one pool,    E7 tensor plus
                 gather each    mx.take      a copy of each
                 step           each step    block that fills
        \             |             |             /
         +-------- decode 48 steps, 5 rounds ----+
                       time, memory, tokens
```

1. `scripts/kv_layouts.py` builds each layout from one prefilled KV. Each layout gives the model the same padded batch, offset and mask.
2. `measure` runs every cell: rows 1 to 16 and contexts 512 to 2,048. Each round starts with the next layout.
3. `Isolated` runs each sample in a new process. One process kept too much memory, and the system stopped it.
4. `measure_attention` times layout C alone. C reads a shared prefix once for all rows and merges each row's tail.
5. `report` writes `bench/results/m3/kv-layout/kv-layout.md` and the JSON beside it.

## What you get at the end

- The run: `bench/results/m3/kv-layout/kv-layout.md`. To run it again: `uv run python -m scripts.kv_layouts --out <dir>`, on AC power.
- The decision: layout B. At 8 rows and 1,024 tokens, B takes 24.06 ms per step and E7 takes 26.38 ms.
- The gather tax: layout A takes 93.85 ms there, 3.56 times E7.
- `tests/unit/test_kv_layouts.py` proves that A, A2 and B give the model the same batch as E7. It also proves that the split attention of C equals one fused attention.
- The tests also prove that the script refuses a run where the layouts give different tokens at one row.

What you do not get yet:

- The `KVStore` behind its interface, with equivalence against E7: M3-T3.
- A pool size from a memory budget: M3-T4.

## Where this puts the project

```
M3  █▓░░░░░
```

This slice opens M3-T3, the KV store.

## Choices the agent made

You approved the plan with A2 and a cap of 16 rows at 1,024 tokens. You can still change these before `/ship`.

- The script takes each sample in a new process. A full run takes 26 minutes, and each sample loads the model again.
- Each time is the median over 5 rounds of each round's mean. The report also gives the median step and the lowest and highest round.
- Layout C uses random K and V, not the model. C only needs an estimate of its gain.

## What can go wrong

- B holds about twice the KV of A, because the working tensor and the blocks each hold a copy. At 16 rows and 1,024 tokens, B holds 4,006 MiB and E7 holds 2,250 MiB. M3-T4 must count both copies.
- C is slower than the fused kernel in every measured cell. A gain for C needs an attention kernel of our own.
- The cause of the memory growth in one process is not known. A long server run with mixed sizes can show the same growth.

## Words

| Word | Meaning |
|---|---|
| layout | How the KV tensors sit in memory: one tensor per batch, or one array per block. |
| gather | A copy of each row's blocks into one padded batch tensor. |
| gather tax | The extra time per decode step that layout A pays for its gather. |
| working tensor | The contiguous batch KV of E7, which B keeps for decode. |
| footprint | The memory that macOS counts for a process, MLX memory included. |

## Learn from this slice

A gather copies every row's KV in each step. Thus its cost grows with rows and context. It is 1.16 times E7 at 1 × 512 and 4.63 times at 8 × 2,048.

Layout B moves the copy off the decode path. It copies a block once when the block fills, and it gathers a row only when the row joins. The cost is memory, not time.

A fast idea can lose to a fused kernel. C reads the prefix once, but MLX's own operations are slower than `mx.fast.scaled_dot_product_attention`.

Test yourself: B does all of E7's work and more. Why did the first quick run show B faster than E7?

<details><summary>Answer</summary>

E7 always ran first in each cell, on the GPU clock that the previous cell left. The script now starts each round with the next layout. In the full run, B is within the round range of E7.

</details>

## Result
