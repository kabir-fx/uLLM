# 0015: A 256 MB limit on MLX's cache of freed buffers

**Status:** accepted · **Task:** M1-T7 (B5)

## Context

During the M1-T7 runs, `ullm serve` alone grew to a 12 GB footprint within 12 requests, pushed the machine into swap, and stalled its token stream for about 200 ms at a time. `mlx_lm.server` held about 1.5 GB.

MLX keeps freed GPU buffers in a cache for reuse. `mx.set_cache_limit`'s docstring in MLX 0.32.4 says the limit "defaults to the memory limit", which is 15.20 GB on this 16 GB M4. A buffer is reused only for a request of its size or a little larger. `ContiguousKV` (decision 0008) concatenates on every append, so each decode step frees 56 K and V buffers (28 layers) and asks for 56 that are 2 KB each larger. The freed ones never fit again, and the cache fills. mlx-lm bounds the same cache by calling `mx.clear_cache()` after prefill and every 256 decoded tokens (`mlx_lm/generate.py`, lines 463 and 479).

Measured on the M4: `MLXRunner` driven as the engine drives it, over workload A's first 12 requests, greedy (throwaway script in the session scratchpad). Each row changes one thing.

| Variant | MLX cache after 12 requests | Footprint | Swap growth | Decode p50 | Decode steps over 100 ms |
|---|---|---|---|---|---|
| No limit (before) | 9.07 GB (4.79 GB after the first) | 10.0 GB | 2.15 GB | 16.4 ms | 12 of 2345, 105 to 215 ms |
| `mx.clear_cache()` after `release` | 0.00 GB | 1.33 GB | none | 16.1 ms | 0 |
| `mx.set_cache_limit(256 MB)` | 0.25 GB | 1.60 GB | none | 16.2 ms | 0 |
| mlx-lm's preallocating `KVCache`, no limit | 2.11 GB, still growing | 3.44 GB | none | 15.1 ms | 0 |

Active memory was 1.11 GB in every variant, so nothing leaked: the growth was all freed buffers held for reuse. The stalls began once the footprint passed about 9.7 GB and swap grew, 8 to 9 tokens apart. 8 tokens of one layer's K is 16 KB, one page. That the stalls are new page-sized allocations under memory pressure is inferred, not traced in the allocator.

## Decision

`load_mlx` calls `mx.set_cache_limit(CACHE_LIMIT)` before loading the model, with `CACHE_LIMIT = 256 MB`. After that run, the same 12 requests gave a 1.57 to 1.61 GB footprint, no swap growth, decode p50 16.3 ms, and no step over 100 ms. `test_a_long_request_leaves_little_freed_memory_held_for_reuse` holds the bound: before the change it measured 5.12 GB held after one request.

## Alternatives considered

- **`mx.clear_cache()` on `release`.** It bounds memory between requests but not inside one: one request of 837 tokens left 4.79 GB. Workload C's 8k-token prompt would grow far more.
- **Preallocating K and V in steps, as mlx-lm does.** It cut decode p50 by about 1 ms, but alone it still let the cache grow. Replacing `ContiguousKV` is E7's job in M2-T1, and that cache will need this limit too.
- **No cache at all (`set_cache_limit(0)`).** Not measured. Every step would allocate from the system, and the 256 MB limit already showed no cost on decode p50.

## Consequences

- No planned feature removes the need for this limit. E7 copies on every batch membership change, E9's option A gathers into new tensors every step, and prefill intermediates vary in size with the prompt.
- uLLM's runs in `bench/results/m1` predate this change. Their memory, swap and stall figures describe the unbounded cache, not uLLM as it now runs.
- The limit is a process-wide MLX setting, applied by whoever loads the model. A process that builds `MLXRunner` without `load_mlx` keeps MLX's default.
- E8's memory budget (M3-T4) should count this limit alongside the weights and the KV pool.
