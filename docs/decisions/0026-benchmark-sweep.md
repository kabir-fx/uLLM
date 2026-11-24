# 0026: The concurrency sweep and how its gaps are explained

**Status:** accepted (the author chose the recommended answer to three questions before any logic was written) · **Task:** M2-T6 (B3, B6) · **Builds on:** 0016 (paired sessions), 0020 (admission), 0023 (recorder)

## Context

- B3 asks for closed loop at concurrency 1, 2, 4, 8 and 16. B6 asks that every gap to a baseline be explained with our own measurements, and every table state its conditions.
- The M1 sessions pinned uLLM to `--max-num-seqs 1`. M2-T2 made it batch. Neither M1 table shows uLLM batching.
- The open gap from M1: at c1, Sustained, uLLM's TPOT is 1.167 [1.160, 1.172] times sequential `mlx_lm.server`'s (claims row 21). Nothing explains it yet.
- Workload A has 256 requests (`bench/workloads/a.json`). `plan.plan` cuts `k` strata of `256 // k` and keeps the rest as the warm-up pool, so `k = 32` leaves no warm-up at c16.
- The benchmark Mac has 16 GB. At c8, Sustained, batched `mlx_lm.server` peaked at 6.78 GB without its prompt cache and 9.68 GB with it; uLLM at `--max-num-seqs 1` at 1.97 GB (`bench/results/m1/c8/probes.jsonl`). uLLM at `--max-num-seqs 8` peaked at 3.17 GB (`bench/results/m2/telemetry/probes.jsonl`). Nobody measured c16.
- `ullm serve --record PATH` opens `PATH` with `"wb"` (`server/app.py`). A server started per chunk therefore overwrites the previous chunk's recording. `bench/results/m2/telemetry/ullm-on.jsonl.gz` holds 32 requests: the last chunk's 8 warm-up and 24 measured.
- The `step` event's `ms` is the runner's `total_ms` (`core/engine.py`, `_step_event`): sequence packing, forward, sampling and `tolist`. The runner keeps `forward_ms` apart, but no event carries it.
- `MLXRunner._decode` repacks the whole batch KV with `BatchKV.pack` when a sequence joins, and `release` repacks when one leaves (`runner/mlx_runner.py`).
- mlx-lm's single path, `generate_step`, queues step `n + 1` with `mx.async_eval` before it waits on step `n` (`mlx_lm/generate.py` lines 466–474, 0.32.2). uLLM's decode calls `mx.eval` on the sampled tokens and then `tolist` (rule 4), so the host builds the next graph only after the GPU is idle. That this explains the c1 gap is inferred.

## Decision

1. **One session per concurrency**, groups `bench/groups/m2-c{1,2,4,8,16}.json`, all with one server per chunk, so every row of the sweep has the same lifecycle.

   | Group | Prompts / warm-up per chunk | Blocks | Settle (s) |
   |---|---|---|---|
   | m2-c1 | 12 / 2 | 21 | 240 |
   | m2-c2 | 12 / 2 | 21 | 600 |
   | m2-c4 | 12 / 4 | 21 | 600 |
   | m2-c8 | 24 / 8 | 10 | 600 |
   | m2-c16 | 30 / 16 | 8 | 600 |

2. **The same engines at every concurrency.** Baseline `mlx-lm-batched-no-cache` (`--prompt-cache-size 0`, no `seed`) and its twin. Engines `ullm` (`ullm serve --max-num-seqs 16 --record {out}/ullm-{block}.jsonl.gz`) and `mlx-lm-batched` (the defaults). `m2-c1` also runs `mlx-lm-sequential-no-cache` (`"seed": 0`), the engine with uLLM's exact tokens and M1's c1 baseline. uLLM runs one configuration everywhere, as one deployment would. **`m2-c16` runs without `mlx-lm-batched`** (the author's answer after the smoke in item 4): one block of `m2-c16-cache.json`, the same baseline against `mlx-lm-batched`, runs last in `make bench-sweep` and records whether it fits on the idle Mac.
3. **`{block}` placeholder.** `bench.session` replaces `{block}` in a launch when the server starts for a chunk. A `resident` group whose launch names `{block}` is refused when the group loads.
4. **Footprint limits come from smoke sessions**, one block each, on the agent's Mac with apps open. Footprint does not depend on other apps (0016), swap does. Each limit is the peak rounded up to the next GB plus 1 GB.

   | Server | Measured peak (GB) | Limit at c1 | c2, c4, c8 | c16 |
   |---|---|---|---|---|
   | `ullm` | 1.94 at c1, 4.92 at c16 | 3 | 6 (c16's) | 6 |
   | `mlx-lm-batched-no-cache` | 2.03 at c1, 6.78 at c8 (M1), 11.59 at c16 | 4 | 8 (c8's) | 13 |
   | `mlx-lm-batched` | 2.72 at c1, 9.68 at c8 (M1), 12.06 at c16 | 4 | 11 (c8's) | 14 (`m2-c16-cache`) |
   | `mlx-lm-sequential-no-cache` | 1.95 at c1 | 3 | | |

   At c16, `mlx-lm-batched` failed 40 of 46 requests: `[METAL] Command buffer execution failed: Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)` in its log, and its generation thread died. Swap grew 3,255 MB during that smoke, with apps open. c2 and c4 take c8's limits, since a footprint grows with the batch (inferred).
5. **`bench.steps`** reads a session's per-chunk recordings and writes `steps.md` beside `summary.md`, per regime, for the measured requests only (joined on request id). It reports:
   - decode step time, p50 and p90, by rows in the step;
   - prefill time per 1,000 prompt tokens;
   - the share of uLLM's busy span spent in prefill, in decode, and between steps;
   - for each request, the prefill time of other requests that ran while it was decoding, as a share of its decode span: the stall FCFS whole-prompt prefill puts on running rows (0020).
6. **`scripts.step_costs`** is an offline micro-benchmark on the real model, one process, no server. At 1, 2, 4, 8 and 16 rows it times uLLM's `MLXRunner` decode step and `BatchKV.pack`. At one row it times mlx-lm's `generate_step` per token and the same model forward with a synchronous `mx.eval` each step. It writes `bench/results/m2/sweep/step-costs.json` and `.md`. It lives in `scripts/`, as `scripts/make_goldens.py` does: the ruff ban on `mlx_lm.generate` covers engine and bench code, and rule 10 covers engine code.
7. **`bench.analysis --sweep`** prints one table per regime across the sessions: rows are engines, columns are concurrencies, cells the output tok/s ratio to `mlx-lm-batched-no-cache` with its interval and "within noise" against that session's twin. A DISCARDED session's column is headed with its verdict, so its cells never read as claims. A second table gives TTFT and TPOT ratios the same way. `make bench-sweep` runs the five sessions, the `m2-c16-cache` block, `scripts.step_costs`, `bench.steps` on each session, and writes `bench/results/m2/sweep/comparison.md`.
8. **The gap report** `bench/results/m2/sweep/README.md` names every gap the tables claim (outside the twin's floor) and its cause, each figure linked to a committed file. A gap without a measured cause says so.

## Alternatives considered

- **`--max-num-seqs` equal to the concurrency.** It answers "the best setting per load", not "one deployment under rising load". Rejected by the author.
- **A sequential row in every session.** About a quarter more session time for a row M1 already has at c8. Rejected by the author.
- **Add `forward_ms` to the `step` event.** It splits packing from forward live, but changes contract C2, the schema and the terminal's generated types for one benchmark. The micro-benchmark measures the same split offline.
- **Resident servers at c1, as in M1.** It keeps one recording per session, but the c1 row would differ in lifecycle from the rest of the sweep.
- **Open loop with Poisson arrivals.** Deferred to M7-T1 by the author: the task line names the closed-loop sweep.

## Consequences

- The sweep is five sessions on the author's idle Mac. M1's c1 and c8 sessions took 62 and 57 minutes (`session.json`).
- c16 has 8 blocks. If few of them are Peak, Peak has no interval, as in M1.
- M3-T9 and M7-T1 reuse `bench.steps` and `--sweep`.
- c16 was DISCARDED twice on swap growth: once in the full sweep, and once more after a reboot, as the author chose. The first run is kept in `bench/results/m2/sweep/c16-discarded/`. No c16 ratio is claimed. The swap growth is recorded as a measured finding (claims row 30).
