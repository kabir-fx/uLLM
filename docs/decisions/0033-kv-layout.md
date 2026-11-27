# 0033: The KV layout is B, gather on membership change

**Status:** accepted · **Task:** M3-T2 (E9)

## Context

E9 names three physical KV layouts for blocks and a decision rule. Take A, then B: the first whose decode step at B=8 with 1k-token contexts is within about 20% of E7. Report A's cost as the gather tax. Prototype C only far enough to estimate its gain. Section 4 predicted that a layout which re-materialises all KV every step pays for the whole context every step.

`scripts/kv_layouts.py` measured each layout on Qwen3-0.6B bf16 (`bench/results/m3/kv-layout/`, commit `e36837a`, Apple M4 16 GB, AC power, Low Power Mode off). The report marks the tree dirty: the only changes were the untracked `.keel/` and the result files it was writing. It is not a B5 session: it compares layouts on one machine, with no baseline server.

- Every cell prefills one long prompt from workload A once. Each candidate builds its cache from that KV. Row i holds `context - i` tokens, so every batch above one row is padded and uses the array mask.
- A decode sample is 5 warm-up steps and 48 timed steps, greedy, with `mx.eval` at the end of each step. 48 steps are three 16-token block cycles, so B's figure includes its block copies.
- 5 rounds per cell. Each round starts with the next candidate. Each time is the median over rounds of each round's mean.
- Each sample runs in a fresh process (see Consequences).
- Contexts 512, 1,024 and 2,048; rows 1, 2, 4, 8 and 16. The grid skips 16 × 2,048, which is the author's cap.

The candidates as built:

- **E7.** `BatchKV` from decision 0019.
- **A.** One array per 16-token block per layer. Each step writes the new token into each row's last block, then gathers the padded batch with `concatenate` and `stack`.
- **A2.** One preallocated pool per layer (2,304 blocks). Each step scatters the new token into the pool and gathers the batch with `mx.take`. The author added A2 to the plan.
- **B.** E7's working tensor, plus a block store of record. When a row's block fills, the block is copied out of the working tensor with `mx.contiguous`. A row that joins is gathered from its blocks.
- **C.** Split attention alone, on random bf16 K and V. The shared prefix is read once for all rows, by folding the rows into its query axis. Then it is merged with each row's tail through log-sum-exp.

## Decision

1. **The layout is B.** At B=8 and T=1,024, B's decode step is 24.06 ms [23.42, 26.16] against E7's 26.38 ms [23.35, 29.25], 0.91 of E7. A is 93.85 ms [75.53, 105.94], 3.56 of E7. A fails the rule, and B passes it.
2. **The gather tax is 3.56× at B=8, T=1,024.** It grows with context at every row count, and with rows up to 8. A/E7 is 1.16 at 1 × 512, 2.26 at 8 × 512, 3.56 at 8 × 1,024 and 4.63 at 8 × 2,048. At 16 rows it is 2.40 at 512 and 3.47 at 1,024, where E7's own rounds spread from 48.46 to 77.73 ms. A2, with one pool and `mx.take`, does not remove it: 3.35 at B=8, T=1,024.
3. **C shows no gain at these sizes with MLX's own operations.** With 8 rows and a 256-token tail, the split attention is slower than the fused kernel over the whole padded batch: +6.57 ms at a 1,024 prefix, +5.27 at 2,048 and +13.21 at 4,096 per decode step, over 28 layers. Reading the prefix once does save work: the split costs 0.28 to 0.37 of the same context as one segment through matmul and log-sum-exp. But that one-segment path costs 3.6 to 4.8 times the fused kernel. A gain for E15 needs an attention kernel of our own. That is outside M3.
4. M3-T3 builds B behind the `KVStore` interface, with E7 as the working tensor. The block manager (decision 0032) owns the tables. The store maps each block id to its K and V.

## Alternatives considered

- **A, gather every step.** It is the vLLM design and the simplest store. It costs 2.3 to 4.6 times E7 per step from 8 rows. Its tax grows with context because each step copies every row's KV.
- **A2, one pool with `mx.take`.** Section 4 measured that a write costs in proportion to the pool, so A2 scatters through one array per layer. It is no faster than A. It also allocates the whole pool up front: 4,032 to 4,034 MiB held in every cell.
- **Stop and discuss.** The rule says to stop only when neither A nor B qualifies. B qualifies.

## Consequences

- **B holds about twice the KV of A.** The working tensor and the blocks each hold a copy of every full block. Held memory at B=8, T=1,024 is 2,004 MiB for B, 1,124 for E7 and 896 for A, so B holds 1.78 times E7 there. At 16 × 1,024 it is 4,006, 2,250 and 1,792. M3-T4 must size the block pool and the working tensor from one memory budget that counts both copies. The rule decides on time only. The cost in memory is the trade that B makes.
- **A join costs a little more in B.** B gathers the joining row from its blocks: 51.95 ms against E7's 48.77 at 8 rows and 1,024 tokens. The leave there is 42.11 ms. A leave repacks the working tensor in the same way as E7.
- **Decode means near the grid's edge are noisy.** At 8 × 2,048, B's mean is 36.48 ms against E7's 33.82, and the medians are 34.18 and 33.34. At 16 × 1,024, B's mean is 69.10 against 62.12, and the medians are 67.40 and 63.47. In both cells the round ranges overlap. The decision cell's ranges are narrower and also overlap.
- **One process kept the high-water mark of its memory.** In the first full run, all in one process, the physical footprint grew from cell to cell while MLX's active memory stayed flat, with its cache cleared. The system killed the run at the 8 × 2,048 join measurement. That run's figures were not kept. A small test showed that the footprint keeps its peak after MLX frees buffers. A small test did not reproduce the growth across cells. The cause in MLX or Metal is inferred, not found. It is a candidate for a report to MLX. The script now takes each sample in a fresh process. M3-T3 and the server must watch the footprint under long runs with mixed sizes.
- At one row, every candidate gave E7's greedy tokens in every round. Above one row, no cell's tokens differed from E7's. M3-T3 still proves B by equivalence against E7 (E9, Q3).
