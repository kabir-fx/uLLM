# 0020: Continuous batching, and admission by reserved padded KV slots

**Status:** accepted (the author chose the recommended answer to each of the four questions before any logic was written) · **Task:** M2-T2 (E6)

## Context

M2-T1 (decision 0019) gave `MLXRunner` one left-padded KV for the running batch. `EngineConfig` still refuses `max_num_seqs != 1` (decision 0010), so the engine never batches. The scheduler's body already follows E6's general rule: admit the oldest waiting request while it fits, prefill it alone, else decode every running request.

Facts that shape admission:

- A batch holds `rows × longest row` slots, and the padding stays until a repack (decision 0019, consequences).
- One slot costs 114,688 bytes on Qwen3-0.6B and 1.7B: 2 (K and V) × 28 layers × 8 KV heads × 128 head dim × 2 bytes (bf16), read from the pinned model's `config.json`. One prompt near `max_model_len` (40,960) at batch 8 needs about 37.6 GB. The development machine has 16 GB (`sysctl hw.memsize`).
- A repack allocates the new batch before the old one is freed (`BatchKV.pack`), so the peak is about twice the batch.
- Nothing preempts until M3 (E11). An admitted request keeps its row until it ends, and every running row grows by one slot per decode.
- The client already refuses a request whose `prompt + max_tokens` exceeds `max_model_len` (`core/client.py`).
- One step prefills one whole prompt (DEC-7). E6's prefill token budget is therefore `max_model_len` until E13, and the client check already holds it.

## Decision

1. `EngineConfig.max_num_seqs` defaults to 16 (E6's starting value) and accepts any int ≥ 1.
2. `EngineConfig.max_kv_slots` defaults to 40,960. `Engine` refuses a value below the runner's `max_model_len`, so a lone request always fits and the bound has no exception.
3. Each request reserves `len(prompt) + max_tokens` slots, the most its row can ever hold. `Scheduler.schedule` admits the head of the waiting queue only when, with it, `rows ≤ max_num_seqs` and `rows × max(reservation over the rows) ≤ max_kv_slots`. If the head does not fit, nothing is admitted that step: no request skips ahead (FCFS).
4. Because the reservation is an upper bound and nothing is admitted past it, the running batch's padded slots stay at or below `max_kv_slots` for the whole run.
5. `hello.limits.max_num_seqs` reports the cap, and `hello.kv.capacity_tokens` reports `max_kv_slots`. Both fields exist in contract C2 v1.
6. `ullm serve` takes `--max-num-seqs` and `--max-kv-slots`. `ullm sim` runs at the defaults.
7. `check_trace` reports a step with more rows than `hello.limits.max_num_seqs`.
8. The engine-level batch-size-1 equivalence test pins `max_num_seqs=1`. Batched equivalence is M2-T3's.

## Alternatives considered

- **Count current lengths, not reservations.** It packs more rows, but decode growth can pass the budget, and without preemption nothing could bring it back. It belongs with E11.
- **A budget of 65,536 slots.** About 7.5 GB steady and 15 GB at a repack peak, which swaps on 16 GB.
- **A smaller budget with a lone-request exception.** The bound would then be soft, and the exception is one more rule to test.
- **An ordering-policy interface now.** FCFS is the waiting deque's order. An interface with one implementation is a speculative layer (rule 3). E16 adds it in M8 with its second policy.
- **Keep `max_num_seqs = 1` by default.** Batching is what M2 delivers, and the M1 c1 baselines run one client, so they stay at batch size 1.

## Consequences

- At workload A's sizes, `max_num_seqs` binds first. At 40,960 slots, two requests with `max_tokens` near the context cannot share a batch even when their prompts are short.
- A request with a large `max_tokens` holds room it may never use. E8's blocks (M3-T4) replace this account with free blocks, and E11 makes the current-length account safe.
- The default does not count weights, the MLX cache limit (decision 0015) or the repack peak. At the default the steady KV is about 4.7 GB and the peak about 9.4 GB.
- The prefill token budget has no code until chunked prefill (E13).
