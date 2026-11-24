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

Only Sustained has blocks at every concurrency. Peak has 3 blocks at c1, 1 at c4 and 1 at c16. Only c1 has a Peak interval. uLLM's Peak gaps are covered under Gap 1 and Gap 3, the baselines' under "Gaps between the baselines". `summary.md` prints footprints in GiB under the label GB, and this page copies them as printed.

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
  - The 0.841 at c1 is 349 against 426 ms at p50. At Peak, c1's TTFT is 0.794 [0.779, 0.843].
  - No measurement here splits the baseline's TTFT, so the c1 gap has no measured cause.

## Gap 2: uLLM's time per output token is higher at c1 and c4

- **c4, 1.044.**
  - This is the cost of Gap 1's admission policy. At c4, other requests' prefill takes 0.218 (p50) of a request's decode span.
  - The client sees the same in stream stalls: 24.44% of uLLM's stream time against 18.83% for the baseline ([c4 summary](c4/summary.md)).
  - mlx-lm prefills up to 8 waiting prompts together in one forward (`prefill_batch_size=8`, `mlx_lm/generate.py` line 1561). uLLM prefills one prompt per step, so the running rows stop more often. That this explains the difference is inferred.
  - At c2 and c8 the TPOT ratio is within noise.
- **c1, 1.014.**
  - The gap is 17.0 against 16.6 ms at p50.
  - Outside steps, uLLM is idle for 0.03 ms between steps at p50 ([c1 steps](c1/steps.md)), so the gap is inside the step.
  - uLLM calls `mx.eval` and then `tolist` each step (rule 4), so the GPU waits while the host builds the next graph. The baseline's `BatchGenerator` queues the next step with `mx.async_eval` before it waits on the current one (`mlx_lm/generate.py` line 1434).
  - Offline, at one row ([step-costs.md](step-costs.md)), uLLM's decode step takes 15.27 ms and mlx-lm's `generate_step`, which pipelines the same way, 13.06 ms per token. But the script's own wait-on-each-step loop takes 15.94 ms, slower than uLLM's step. So the offline figures do not isolate pipelining, and the 0.4 ms has no measured cause. Pipelining is a candidate (guess).

## Gap 3: uLLM gives more output tokens per second from c2 on

- Output tok/s is 1.079 at c2, 1.093 at c4 and 1.120 at c8, with E2E 0.919, 0.912 and 0.891.
- TPOT is within noise at c2 and c8. So the gain comes from Gap 1: each request waits less for its first token, and the next request in the closed loop starts sooner (inferred from the ratios above).
- The baseline's TTFT p50 grows from 426 ms at c1 to 1,596 ms at c8 ([c8 summary](c8/summary.md)). uLLM's stays between 349 and 597 ms.
- At c1, E2E is 0.989 [0.981, 0.999] at Sustained. At Peak, E2E is 0.986 [0.975, 1.098] and output tok/s 1.035 [0.930, 1.043], with TPOT within noise. Each follows from c1's TTFT gap (inferred), whose own cause is not measured.

## Gaps between the baselines

These are mlx-lm's paths against each other. uLLM is not in them, and no measurement here explains them.

- **`mlx-lm-sequential-no-cache` at c1.** At Sustained, TTFT is 0.910 [0.894, 0.928], TPOT 0.995 [0.991, 1.001], E2E 0.981 [0.973, 0.989] and output tok/s 1.012 [1.008, 1.035]. At Peak, TTFT is 0.865 [0.863, 0.874], TPOT 0.993 [0.992, 0.996], E2E 0.981 [0.979, 0.985] and output tok/s 1.035 [1.032, 1.044]. A request with a `seed` is not batchable (`mlx_lm/server.py` line 771), so it takes `_serve_single` and `generate_step` and skips `BatchGenerator`. That this explains the gaps is a guess.
- **Cached `mlx-lm-batched` at c1.** At Sustained, TPOT is 1.004 [0.998, 1.014] and output tok/s 0.993 [0.974, 0.996]. At Peak, TTFT is 1.016 [1.015, 1.023] and E2E 1.003 [1.002, 1.007]. Workload A's prompts are distinct dialogues, so the prompt cache can reuse little beyond the chat template's first tokens and mostly adds work. That this is the cause is a guess.

## M1's gap at c1 does not recur

- M1 claimed uLLM's TPOT at c1, Sustained, as 1.167 times sequential `mlx_lm.server` [1.160, 1.172] (claims row 21, commit `41671f1`).
- This sweep's c1 session runs that same comparator. At Sustained, against the same baseline, uLLM's TPOT is 1.014 [1.008, 1.024] and the sequential path's is 0.995 [0.991, 1.001]. So uLLM is about 2% above the sequential path, not 17%. The two are not paired with each other, so there is no interval for that difference.
- The Sustained clocks differ: 1056, 928 and 1170 MHz led in M1, and 618, 796 and 928 MHz here. A host cost per step weighs less when each step's GPU work takes longer, so part of the change can come from the clocks (inferred).
- uLLM's decode path changed between the two commits: M2-T2 replaced it with one forward on `BatchKV` (`a229d86`, `7cf14da`). This was not bisected, so it is inferred.
- The Peak TPOT p50s, at 1470 MHz in both sessions, point the same way: in M1, uLLM 16.5 against 14.0 ms; here, uLLM 14.3 against the baseline's 14.3 ms. M1's Peak had 2 blocks and no interval, so this is context, not a claim.

## c16: the baseline does not fit beside macOS on 16 GB

- c16 ran twice, and both runs were DISCARDED on swap.
- **The rerun, after a reboot.**
  - Swap grew from 445 to 1,454 MB by the end of the baseline's first chunk. It grew again, to 1,633 MB, during the baseline's block 1 chunk. MB here is 2^20 bytes, as in the gate.
  - Both baseline servers held a footprint of 11.48 to 11.49 GiB during those chunks. uLLM held 1.61 to 1.64 GiB ([probes](c16/probes.jsonl)).
  - Spotlight was indexing after the reboot: `mds_stores` at 108% CPU at the start ([env](c16/env.json)). This may have added to the swap, so the rerun alone does not show the baseline as the only cause.
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
| Decode step, ms | 15.27 | 15.67 | 17.45 | 21.44 | 39.57 |
| Repack on join, ms | 5.49 | 9.45 | 17.45 | 28.02 | 48.34 |
| Repack on leave, ms | n/a | 3.13 | 11.25 | 22.76 | 47.16 |

- Each repack is timed as the extra time of the decode step that runs its copy, because MLX runs the copy lazily in that step's `mx.eval`. Join packs the batch at the given row count. Leave repacks the rows left after one leaves. At one row no row is left, so leave has no figure.
- From 8 to 16 rows, the decode step grows 1.85 times. At 16 rows a join repack costs 1.22 steps and a leave repack 1.19 steps.
- The one-row join, 5.49 ms, is the first decode after a prefill. It may hold first-step costs other than the copy (guess).
- `BatchKV` copies the whole batch's KV on each change. A paged KV (M3) does not.
