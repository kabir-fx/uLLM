# Step costs, offline

- `qwen3-0.6b-bf16`, one prompt of 461 tokens from workload A, 128 greedy tokens, 5 rounds. Each cell is the median over rounds of each round's median, [lowest, highest] round.
- Commit `c51200e20812` (dirty tree). Apple M4, 16 GB, macOS 27.2.1. AC True, Low Power Mode False. Not a B5 session: it explains a gap, it is not one.

## One row: ms per token after the first

| Loop | ms |
|---|---|
| `generate_step`, queues the next step first | 13.44 [12.84, 13.46] |
| The same call, waits on each step | 16.35 [15.35, 19.28] |

## `MLXRunner` decode step, ms

| Rows | Forward | Step total | Join repack | Leave repack |
|---|---|---|---|---|
| 1 | 14.43 [14.32, 14.55] | 14.44 [14.33, 14.56] | 0.35 [0.35, 0.37] | 0.31 [0.28, 0.34] |
| 2 | 15.08 [14.89, 15.69] | 15.08 [14.90, 15.70] | 0.78 [0.75, 0.81] | 0.23 [0.22, 0.27] |
| 4 | 16.57 [16.55, 18.88] | 16.58 [16.56, 18.96] | 1.41 [1.38, 1.43] | 0.41 [0.39, 1.30] |
| 8 | 21.20 [20.96, 22.74] | 21.21 [21.08, 22.84] | 2.82 [2.77, 3.14] | 0.83 [0.82, 4.22] |
| 16 | 56.15 [55.41, 58.56] | 56.36 [55.61, 58.76] | 5.91 [5.70, 5.99] | 7.55 [6.17, 8.03] |
