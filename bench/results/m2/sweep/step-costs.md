# Step costs, offline

- `qwen3-0.6b-bf16`, one prompt of 461 tokens from workload A, 128 greedy tokens, 5 rounds. Each cell is the median over rounds of each round's median, [lowest, highest] round.
- Commit `3305382d521d` (dirty tree). Apple M4, 16 GB, macOS 27.2.1. AC True, Low Power Mode False. Not a B5 session: it explains a gap, it is not one.

## One row: ms per token after the first

| Loop | ms |
|---|---|
| `generate_step`, queues the next step first | 13.06 [13.03, 13.29] |
| The same call, waits on each step | 15.94 [15.29, 15.97] |

## `MLXRunner` decode step, ms

| Rows | Forward | Step total | Join repack | Leave repack |
|---|---|---|---|---|
| 1 | 15.25 [15.23, 15.27] | 15.27 [15.26, 15.28] | 5.49 [5.18, 6.31] | n/a |
| 2 | 15.65 [15.64, 15.66] | 15.67 [15.66, 15.69] | 9.45 [9.18, 9.93] | 3.13 [2.97, 4.06] |
| 4 | 17.41 [17.21, 17.72] | 17.45 [17.25, 17.77] | 17.45 [17.17, 28.01] | 11.25 [10.26, 11.85] |
| 8 | 21.35 [21.18, 21.52] | 21.44 [21.27, 21.63] | 28.02 [27.86, 28.91] | 22.76 [22.60, 22.99] |
| 16 | 39.40 [39.27, 39.53] | 39.57 [39.44, 39.69] | 48.34 [48.07, 48.37] | 47.16 [46.87, 48.31] |
