# M2 sweep: workload A at concurrency 1 to 16, and why each gap exists

Decision 0026 defines the sessions. Each ratio is to batched `mlx_lm.server` without prompt cache, `mlx-lm-batched-no-cache`. Each ratio is paired prompt by prompt in each block, with a 95% block-bootstrap interval. A gap is claimed only when its median is outside the twin's floor (decision 0016). All sessions ran on the same machine:

- a fanless 16 GB M4 Air on AC power, with Low Power Mode off and the lid open;
- `qwen3-0.6b-bf16`;
- commit `c51200e`, the build of this task.

`env.json` in each session says "dirty". The cause is the untracked result folders and `.keel/`, not source changes.

| Session | Verdict | Blocks: Peak, Sustained, Transition | Files |
|---|---|---|---|
| c1 | VALID | 3, 17, 1 | [summary](c1/summary.md), [steps](c1/steps.md) |
| c2 | VALID | 0, 20, 1 | [summary](c2/summary.md), [steps](c2/steps.md) |
| c4 | VALID | 1, 19, 1 | [summary](c4/summary.md), [steps](c4/steps.md) |
| c8 | VALID | 0, 9, 1 | [summary](c8/summary.md), [steps](c8/steps.md) |
| c16 | DISCARDED: swap grew 1,188 MB | 1, 6, 1 | [summary](c16/summary.md), [steps](c16/steps.md) |
| c16, first run | DISCARDED: swap grew 326 MB, twin TPOT off by 5.56% | 0, 7, 1 | [summary](c16-discarded/summary.md) |
| c16-cache | DISCARDED: 38 of 138 requests failed | 0, 0, 1 | [summary](c16-cache/summary.md), [log](c16-cache/mlx-lm-batched.log) |

The table across concurrencies is [comparison.md](comparison.md). The offline step costs are [step-costs.md](step-costs.md).

## Sustained, the claimed figures

Only Sustained has blocks at every concurrency. Peak has 3 blocks at c1 and 1 at c4. Its figures are in [comparison.md](comparison.md) and are not claimed.

| uLLM, ratio to the baseline | c1 | c2 | c4 | c8 |
|---|---|---|---|---|
| Output tok/s | 1.006 [0.994, 1.020] | 1.079 [1.053, 1.099] | 1.093 [1.050, 1.117] | 1.120 [1.086, 1.170] |
| TTFT | 0.841 [0.793, 0.863] | 0.548 [0.505, 0.586] | 0.465 [0.424, 0.526] | 0.424 [0.375, 0.482] |
| TPOT | 1.014 [1.008, 1.024] | within noise | 1.044 [1.018, 1.080] | within noise |
| E2E | 0.989 [0.981, 0.999] | 0.919 [0.900, 0.938] | 0.912 [0.887, 0.930] | 0.891 [0.844, 0.921] |
| Peak footprint, GB, uLLM against the baseline | 1.96, 2.03 | 2.04, 2.81 | 2.52, 4.14 | 3.17, 6.64 |

c1's output tok/s gap is outside the twin's floor (±0.005), but its interval includes 1.

The cached `mlx-lm-batched` is within noise of the baseline at c2, c4 and c8. At c1 it is 0.993 for output tok/s. Its footprint is 2.77, 3.68, 5.84 and 9.82 GB at c1, c2, c4 and c8.

## Gap 1: uLLM's time to first token is lower, more so with concurrency

- **uLLM side, measured.**
  - A waiting prompt is prefilled at the engine's next step, one prompt per step, before the running rows decode (decision 0020).
  - The recordings show the cost: other requests' prefill takes up a running request's decode span, p50 [c2](c2/steps.md) 0.101, [c4](c4/steps.md) 0.218 and [c8](c8/steps.md) 0.311.
- **Baseline side, read in the source, not measured.**
  - When `mlx_lm.server` has a batch running, it runs `BatchGenerator.next()` for a time budget of 0.5 s before it takes a new request from its queue. See `TimeBudget(budget=0.5)` in `mlx_lm/server.py` lines 329–349 and the loop at line 942 (0.32.2).
  - A request that arrives mid-budget therefore waits for the rest of the budget before its prefill starts. This explains c2 to c8 (inferred).
- **c1 is not explained.**
  - At c1 the batch is empty when a request arrives, so the budget cannot delay it.
  - The 0.841 at c1 is 349 against 426 ms at p50. No measurement here splits the baseline's TTFT, so this part of the gap has no measured cause.

## Gap 2: uLLM's time per output token is higher at c1 and c4

- **c4, 1.044.**
  - This is the cost of Gap 1's admission policy. At c4, other requests' prefill takes 0.218 (p50) of a request's decode span.
  - The client sees the same in stream stalls: 24.44% of uLLM's stream time against 18.83% for the baseline ([c4 summary](c4/summary.md)).
  - mlx-lm prefills up to 8 waiting prompts together in one forward (`prefill_batch_size=8`, `mlx_lm/generate.py` line 1561). uLLM prefills one prompt per step, so the running rows stop more often. That this explains the difference is inferred.
  - At c2 and c8 the TPOT ratio is within noise.
- **c1, 1.014.**
  - The gap is 17.0 against 16.6 ms at p50.
  - Outside steps, uLLM is idle for 0.03 ms between steps at p50 ([c1 steps](c1/steps.md)), so the gap is inside the step.
  - Offline, at one row ([step-costs.md](step-costs.md)), uLLM's decode step takes 14.44 ms. mlx-lm's `generate_step` takes 13.44 ms per token, and it queues the next step before it waits on the current one.
  - uLLM calls `mx.eval` and then `tolist` each step (rule 4), so the GPU waits while the host builds the next graph. That this explains the 0.4 ms is inferred.
  - The script's own wait-on-each-step loop measured 16.35 ms, slower than uLLM's step. So that variant does not isolate the pipelining on its own.

## Gap 3: uLLM gives more output tokens per second from c2 on

- Output tok/s is 1.079 at c2, 1.093 at c4 and 1.120 at c8, with E2E 0.919, 0.912 and 0.891.
- TPOT is within noise at c2 and c8. So the gain comes from Gap 1: each request waits less for its first token, and the next request in the closed loop starts sooner (inferred from the ratios above).
- The baseline's TTFT p50 grows from 426 ms at c1 to 1,596 ms at c8 ([c8 summary](c8/summary.md)). uLLM's stays between 349 and 597 ms.

## The M1 gap at c1 is closed

- M1 claimed uLLM's TPOT at c1 as 1.167 times sequential `mlx_lm.server` (claims row 21, commit `41671f1`).
- At Peak, with the GPU at 1470 MHz for at least 94% of the time in both sessions:

  | Session | uLLM TPOT p50 | Comparison TPOT p50 |
  |---|---|---|
  | M1, [c1](../../m1/c1/summary.md) | 16.5 ms | 14.0 ms (sequential `mlx_lm.server`) |
  | This sweep, [c1](c1/summary.md) | 14.3 ms | 14.3 ms (the baseline) |

  This sweep's sequential row is 0.993 of the baseline at Peak.
- uLLM's decode path changed between the two commits. M2-T2 replaced it with one forward on `BatchKV` (`a229d86`, `7cf14da`). Which change closed the gap was not bisected, so the cause is inferred.

## c16: the baseline does not fit beside macOS on 16 GB

- c16 ran twice, and both runs were DISCARDED on swap.
- **The rerun, after a reboot.**
  - Swap grew from 445 to 1,454 MB by the end of the baseline's first chunk. It grew again, to 1,633 MB, during the baseline's block 1 chunk. MB here is 2^20 bytes, as in the gate.
  - Both baseline servers held a footprint of 11.48 to 11.50 GiB during those chunks. uLLM held 1.64 GiB ([probes](c16/probes.jsonl)).
  - Spotlight was indexing after the reboot: `mds_stores` at 108% CPU at the start ([env](c16/env.json)).
- **The first run.**
  - Swap grew from 404 to 730 MB.
  - The twin's Sustained TPOT median was 1.0556.
- **Cached `mlx-lm-batched` at c16** ([c16-cache](c16-cache/mlx-lm-batched.log)):
  - Metal ran out of memory: `kIOGPUCommandBufferCallbackErrorOutOfMemory`.
  - The generation thread died, and 38 of 138 requests failed. The smoke run in decision 0026 showed the same.
- **What the discarded rerun shows, for the record only.**
  - uLLM's peak footprint was 5.08 GB against the baseline's 11.56.
  - uLLM held 10.9 rows per decode step on average.
  - Its 16-row decode step took 46.11 ms at p50, Sustained ([c16 steps](c16/steps.md)). In the first run it took 78.58 ms ([c16-discarded steps](c16-discarded/steps.md)). The runs disagree, so neither is a figure for M3.
  - A DISCARDED session is not claimed, so c16 has no ratio in the claims ledger.

## Offline step costs: what M3 inherits

[step-costs.md](step-costs.md), five rounds on the idle Mac, one prompt of 461 tokens:

| Rows | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| Decode step, ms | 14.44 | 15.08 | 16.58 | 21.21 | 56.36 |
| Repack on join, ms | 0.35 | 0.78 | 1.41 | 2.82 | 5.91 |
| Repack on leave, ms | 0.31 | 0.23 | 0.41 | 0.83 | 7.55 |

- From 8 to 16 rows, the decode step grows 2.7 times. A sequence's repack on join or leave is then 10 to 13% of a step.
- `BatchKV` copies the whole batch's KV on each change. A paged KV (M3) does not.
