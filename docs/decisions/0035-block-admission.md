# 0035: Admission by free blocks, a pool sized from memory, and E8's metrics

**Status:** accepted (the author chose the recommended answer to each of the three questions before any logic was written) · **Task:** M3-T4 (E6, E8, O3)

## Context

- Decision 0034 gave the `Scheduler` a `BlockManager` of `ceil(max_kv_slots / block_size) + max_num_seqs` blocks. Admission still holds `rows × longest reservation ≤ max_kv_slots` (decision 0020), so allocation cannot fail. Nothing asks the pool whether a request fits.
- E6 admits "enough free blocks for the prompt after prefix hits plus a small safety watermark". That is safe only when a decode that finds the pool empty can preempt (E11, M3-T6). Before M3-T6, a decode that cannot get a block has no legal move: it cannot stall one row of a batched step, and ending the reply early changes its tokens (rule 1).
- Layout B (decision 0033) holds two copies while decoding: the blocks, and E7's left-padded working tensor of `rows × capacity(longest)` slots. Blocks do not remove padding from the working tensor. 0020's padded bound is what bounds it.
- One slot costs 114,688 bytes on Qwen3-0.6B and 1.7B (0020), so one 16-token block costs 1,835,008 bytes. The core does not know this figure: it comes from the model's config.
- `Engine.run` hands `Load(waiting, running)` to the client every loop turn. `EngineClient.health()` returns it, and `GET /metrics` sets its gauges from it (decision 0024). No other path carries engine state to a metric without touching the engine thread.

## Usage

```python
# Scheduler._fits, per waiting head
rows <= max_num_seqs
and rows * longest_reservation <= max_kv_slots                       # the working tensor (0020)
and sum(blocks_for(r.kv_reservation) for r in running + [req]) <= self._blocks.num_blocks

# startup
EngineConfig(max_num_seqs=16, max_kv_slots=40_960, kv_pool_bytes=None)  # pool: 2,560 blocks
EngineConfig(kv_pool_bytes=2 * 2**30)  # 1,170 blocks at 1,835,008 bytes per block
# ullm serve --kv-pool-gib 2

# a scrape
health = client.health()   # Health(alive, in_flight, waiting, running, kv=KvUsage(...))
metrics.render(waiting=health.waiting, running=health.running, kv=health.kv)
# ullm_kv_blocks{state="free"} 2380.0
# ullm_kv_blocks{state="in_use"} 180.0
# ullm_kv_blocks{state="cached"} 0.0
# ullm_kv_waste_ratio 0.031
# ullm_kv_max_concurrent_seqs 1.0
```

## Shape

1. **Admission by reserved blocks.** Each running request reserves `ceil((prompt + max_tokens) / block_size)` blocks, the most its table can reach. The head is admitted only when the running requests' reservations plus its own fit in `num_blocks`. The sum is computed from the running list each time, not kept as a counter, so it cannot drift from `remove`. Allocated blocks never exceed reserved blocks, so `allocate` and `append_slot` cannot raise `OutOfBlocks`. If one does, it is a core bug and the engine thread dies, as in 0034. FCFS holds: a head that does not fit admits nobody.
2. **0020's padded bound stays**, unchanged, for the working tensor. Both checks run. The block check binds when the pool holds fewer tokens than `max_kv_slots`, or by block rounding at the edge.
3. **The pool is sized from bytes.** `RunnerInfo.kv_bytes_per_token: int | None = None`. `MLXRunner` fills it from the model's args: `2 × num_hidden_layers × num_key_value_heads × head_dim × dtype size`. `FakeRunner` and the sim leave it `None`. `EngineConfig.kv_pool_bytes: int | None = None`. With `None`, `num_blocks = ceil(max_kv_slots / block_size)`: the pool holds as many tokens as the working tensor may pad to. With a figure, `num_blocks = kv_pool_bytes // (kv_bytes_per_token × block_size)`. `Engine` raises `ValueError` when a figure is given and the runner reports `None`, and when `num_blocks < ceil(max_model_len / block_size)`, so a lone request always fits (as 0020 does for slots).
4. **`ullm serve --kv-pool-gib X`** sets `kv_pool_bytes = int(X × 2**30)`. GiB in the name, so the unit cannot be misread.
5. **The pool's state rides the existing `Load`.** `KvUsage(num_blocks, free, in_use, cached, empty_slots)` is a frozen dataclass in `ullm.core.block_manager`. `BlockManager.usage()` builds it. `num_in_use` becomes `num_blocks - free - cached`, O(1). `empty_slots` is the unwritten slots in each table's last block, the only waste E8 allows, summed over tables. `Load` and `Health` gain `kv: KvUsage`. The engine builds it once per loop turn, after the step. A shared block (M3-T5) is full, so it adds no empty slots and is counted once.
6. **E8's metrics** are gauges set at scrape time from `Health.kv`, like `queue_depth` (rule 5: the engine thread never touches a metric):
   - `ullm_kv_blocks{state}` with `free`, `in_use`, `cached`. KV utilisation is `in_use / sum` in PromQL, not a second series.
   - `ullm_kv_waste_ratio`: `empty_slots / (in_use × block_size)`, 0 with no block in use.
   - `ullm_kv_max_concurrent_seqs`: `min(max_num_seqs, num_blocks // ceil(max_model_len / block_size))`, the requests the pool can hold at the longest legal reservation. It is set once from `hello`'s figures.
   `Metrics.render` takes the `KvUsage` beside `waiting` and `running`.
7. **`hello.kv`** gets `block_size`, `num_blocks` and `capacity_tokens = num_blocks × block_size`. 0020 point 5 kept `capacity_tokens` null because the terminal divides unpadded `step.kv_tokens` by it. The pool counts unpadded tokens too, so the quotient is now KV utilisation. C2 stays v1: these fields exist and default to null.
8. **Grafana.** One "KV usage" panel in `deploy/grafana/dashboards/ullm.json`: blocks in use and Cached as a share of the pool, and the waste ratio, in percent.

What this does not do: admit by prompt blocks with a watermark, preempt, or bound the working tensor's repack peak. M3-T6 replaces point 1's reservation with E6's rule once a decode can preempt. The physical memory of the store is not measured here: M3-T9 measures memory waste.

## Failure and operation

- **Twice.** `Scheduler.remove` of an unknown request raises (unchanged). The reservation sum is derived, so a double count cannot exist.
- **A raise halfway.** The check runs before `allocate`. `allocate` changes nothing when it raises (0032 point 7).
- **Concurrent writers.** None. `KvUsage` is an immutable value built on the engine thread and read on the event loop.
- **Trust boundary.** `max_tokens` is bounded by `max_model_len` in the client and the engine (0020 point 4). The startup check makes every legal request fit an empty pool, so no head waits for ever.
- **In production.** The four gauges and the panel. A queue that grows while `kv_blocks{state="free"}` stays high shows the padded bound binding, not the pool.

## Trade-offs accepted

- We accept that, at the default pool, admission admits the same batches as 0020, give or take block rounding, in exchange for a check that cannot run the pool out before preemption exists. `sum(reservations) ≤ rows × longest reservation`, so the padded bound is the tighter one whenever the pool holds `max_kv_slots` tokens.
- We accept a reservation of `max_tokens` that a reply may never use. M3-T6 removes it.
- We accept an O(rows) sum per admission check and an O(tables) `empty_slots` per loop turn, at most 16 and a few hundred small ints.
- We accept a pool flag in GiB beside a slot flag, two knobs, in exchange for a core that knows nothing of layout B's two copies.

## Alternatives considered

- **E6's rule now: prompt blocks plus a watermark.** It packs more requests, but a decode that finds the pool empty has no legal move until M3-T6. It needs preemption pulled into this task.
- **One `--kv-memory-gib` budget that replaces `--max-kv-slots`.** The engine would split it between the working tensor and the pool. It lost because the split encodes layout B's two copies in the core (information leakage), and E7 has only one copy.
- **The pool flag in blocks (`--kv-blocks`).** No bytes in `RunnerInfo`. It lost because E8 sizes the pool "from a memory budget", and a count of blocks hides the model's bytes per block from the operator.
- **A reserved-blocks counter in the `Scheduler`.** One add at admission and one subtract at remove. It lost to the derived sum: a second writer to the same figure, for 16 additions saved.
- **KV usage on a new C2 event.** The `Load` path already crosses the threads once per turn and feeds `/metrics`. A new event would make the hub parse what the engine already counted.

## Consequences

As built, with these changes to the sketch above:

- `Metrics.render(waiting=, running=, kv=)` takes the `KvUsage`, not the whole `Health`: `ullm.telemetry` then imports only the MLX-free `ullm.core.block_manager`, not the client.
- `ullm_kv_max_concurrent_seqs` is set when `Metrics.observe` sees `hello`, from its `limits` and `kv`. It is 0 before `hello`. No new wiring carries the figure.
- `pool_blocks(config, info)` in `ullm.core.engine` sizes the pool. `Engine` builds its `Scheduler` from it, and `EngineClient` builds its idle `Load` from it before the engine thread starts, so `health()` is never without `kv`. After the engine exits, `health()` reports that idle pool again.
- `KvUsage` carries `block_size`, so `waste_ratio` is a property of the value.
- The pool no longer adds `max_num_seqs` blocks (decision 0034's sizing): admission now checks the pool, so the margin has no job.
- `load_mlx` takes the KV dtype from the embedding: `scales` for a quantized one, `weight` otherwise. On the three cached Qwen3-0.6B variants (bf16, 8-bit, 4-bit) it gives 114,688 bytes a token, equal to the K and V a forward wrote, measured with a throwaway script. `test_the_block_store_holds_the_reported_kv_bytes_for_each_token` holds it for the default model.
- `/health`'s JSON gains a `kv` object, because `Health` is a dataclass that `asdict` serialises.
- The terminal's KV figure now divides `step.kv_tokens` by the pool's `capacity_tokens`, so it shows KV use as a share of the pool.
- A `--max-kv-slots` of 10^9 still builds a pool of 62.5 million blocks at startup, as in M3-T3, because the default pool follows the slot budget.
