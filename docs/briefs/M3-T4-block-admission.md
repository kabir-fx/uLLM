# M3-T4: Admission by free blocks, memory metrics, KV usage panel

**State:** NOT DONE. The Grafana claim waits for Docker. Read the Result.
**Decision record:** [0035](../decisions/0035-block-admission.md) (the full design, for agents) · **Spec:** roadmap E6, E8 and O3, task M3-T4

## The goal

The scheduler will admit a request only when the block pool can hold it. You will see the pool's state in `/metrics` and in a Grafana panel.

## The problem

- The `Scheduler` builds a `BlockManager` but never asks it whether a request fits. Admission counts padded slots only (decision 0020).
- E6 admits by prompt blocks plus a watermark. That is safe only with preemption, which M3-T6 builds. Before that, a decode with no free block has no legal move.
- Layout B keeps a padded working tensor beside the blocks (decision 0033). Blocks do not bound that tensor, so 0020's slot bound must stay.
- The pool size comes from `max_kv_slots`, not from memory. E8 asks for a pool sized from a memory budget.
- Rule 5: the engine thread never blocks on a metric. Rule 3: `ullm.core` never imports MLX.

## How it works

```
 engine thread                                   event loop
 Scheduler._fits ── BlockManager                 EngineClient.health()
   rows, padded slots, reserved blocks             └─ Health(..., kv=KvUsage)
 Engine.run ── Load(waiting, running, kv) ─────►       │
                                                 Metrics.render(health) ── GET /metrics
                                                       │
                                                 Prometheus ── Grafana "KV usage"
```

1. `Scheduler._fits`, in `src/ullm/core/scheduler.py`, adds a third check. Each running request reserves the blocks for `prompt + max_tokens`. The head joins only when all reservations fit in the pool.
2. `RunnerInfo.kv_bytes_per_token`, in `src/ullm/runner/base.py`, gives the bytes of one token of KV. `load_mlx` reads it from the model's config.
3. `EngineConfig.kv_pool_bytes` sets the pool. `ullm serve --kv-pool-gib` fills it. Without it, the pool holds `max_kv_slots` tokens.
4. `BlockManager.usage()` returns a `KvUsage` value. The engine puts it in `Load` once per loop turn.
5. `Metrics.render` sets three gauges from it: `ullm_kv_blocks{state}`, `ullm_kv_waste_ratio` and `ullm_kv_max_concurrent_seqs`.
6. `hello.kv` gets `block_size`, `num_blocks` and `capacity_tokens`. The terminal can then show KV use as a share of the pool.
7. `deploy/grafana/dashboards/ullm.json` gets one "KV usage" panel.

One block costs 1,835,008 bytes on Qwen3-0.6B. That is 16 slots of 114,688 bytes, from decision 0020.

## What you get at the end

- `ullm serve --kv-pool-gib 8` starts with 4,681 blocks. The command `curl 127.0.0.1:8000/metrics | grep kv_` shows the gauges.
- `ullm serve --kv-pool-gib 2` stops at startup. Its 1,170 blocks cannot hold one request of 40,960 tokens, which needs 2,560 blocks.
- `make grafana` shows the "KV usage" panel next to the M2 panels.
- Unit tests show that a head which overfills the pool waits, and that no request skips it.
- The scheduler property test checks that allocation never runs out of blocks at any pool size.
- A metrics test checks each gauge against a known `KvUsage`.
- A startup test checks that a pool below one `max_model_len` request is refused.

What you do not get yet:

- Admission by prompt blocks with a watermark comes with preemption in M3-T6.
- Cached blocks and prefix hits come in M3-T5.
- The KV block grid in the terminal comes in M3-T8.
- A measured memory figure for the store comes in M3-T9.

## Where this puts the project

```
M3  █ █ █ ▓ ░ ░ ░ ░ ░
```

This slice opens M3-T5. Prefix caching needs an admission check that counts Cached blocks as room.

## What you approve

- **Reserved blocks, not prompt blocks.** Each request reserves blocks for `prompt + max_tokens`. It beat E6's prompt blocks plus a watermark. That rule needs preemption, and preemption is M3-T6.
- **Two knobs.** `--max-kv-slots` still bounds the padded working tensor. `--kv-pool-gib` bounds the blocks. It beat one memory budget that the engine splits. A split puts layout B's two copies into the core.
- **A derived sum.** The scheduler adds the reservations of the running list for each check. It beat a counter, which is a second writer to the same figure.

Open questions, with my recommended answers:

1. Is a reserved-blocks rule acceptable until M3-T6? At the default pool it admits the same batches as 0020, give or take block rounding. I recommend yes. The check then binds only when you set a smaller pool.
2. What is the default pool? I recommend "as many tokens as `--max-kv-slots`", 2,560 blocks at the default. A fixed GiB default is the other option.
3. What does `ullm_kv_max_concurrent_seqs` mean? I recommend the requests the pool can hold at `max_model_len` each, capped at `max_num_seqs`. The other option is the peak number of running requests seen.

## What can go wrong

- A large `max_tokens` holds blocks that the reply never uses. You see a queue while free blocks stay high.
- A queue with many free blocks can also mean that the padded slot bound binds first.
- A pool flag on `--fake` stops startup with an error, because the fake runner has no KV bytes.
- A pool smaller than one `max_model_len` request stops startup with an error.

## Words

| Word | Meaning |
|---|---|
| reservation | The most KV slots a request can hold: its prompt plus `max_tokens`. |
| pool | The fixed number of KV blocks that the `BlockManager` gives out. |
| working tensor | Layout B's padded batch KV, a second copy beside the blocks. |
| waste ratio | The unwritten slots in last blocks, as a share of the slots in use. |

## Learn from this slice

An admission rule is safe only with a recovery path for the case it does not cover. E6's watermark rule lets memory run out during a decode. Preemption is the recovery path, so the safe rule before preemption reserves the worst case.

Two physical copies need two bounds. Blocks count unpadded tokens. The working tensor counts padded slots. One bound cannot hold both unless the core knows the layout.

Test yourself: why does a reserved-blocks check never bind before the slot check at the default pool?

<details><summary>Answer</summary>

The sum of the reservations is at most the row count times the longest reservation. The default pool holds `max_kv_slots` tokens. So the slot check fails first, except by block rounding.

</details>

## Result

State: NOT DONE. The review is READY at `84cbecc`, and its fix commits run to `b15a51a`. The proof is INCONCLUSIVE at `84cbecc`. Claims 1 to 6 passed. Claim 7, the Grafana panel with live data, did not run because the Docker daemon was off. `make ci` passed at `b15a51a`.

Changes against the plan:
- `/health` now carries a `kv` object. C4 and decision 0011 planned that for M3.
- `Metrics.render` takes the `KvUsage`, not the whole `Health`.
- The pool example is now 8 GiB. A 2 GiB pool cannot hold one request of 40,960 tokens, so the engine refuses it.

Bugs that review found:
- No test checked the GiB unit.
- No test checked ceil against floor when `max_model_len` is not a multiple of 16.
- The decision record quoted 8-bit and 4-bit figures from a script that is not in the repo.
- Roadmap E6 did not name the rule that holds until M3-T6.

All four are fixed. A test now fails on each named mutation and passes without it.

What is not proven:
- The panel queries against a live Prometheus.
- The KV bytes for the quantized models.
- The block check under real-model load. At the default pool, the slot bound binds first.

`make bench-smoke` ran on battery. Base `4a68e01` gave 59.8 output tok/s at the start and 35.1 when rerun later. Head `f721c87` gave 35.4. The machine state moved the figure, and the change did not.

Read these yourself:
- `src/ullm/core/scheduler.py:80`: the three checks in `_fits`.
- `src/ullm/core/engine.py:315`: `pool_blocks` and its two refusals.
- `src/ullm/runner/mlx_runner.py:55`: the dtype that `_kv_bytes_per_token` reads.

