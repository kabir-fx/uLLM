# 0014: The M1 baselines: what each target runs, and why

**Status:** accepted · **Task:** M1-T7 (B4, B5, B6)

## Context

B4 asks for uLLM in sequential mode and `mlx_lm.server` "configured sequential and in its default batched mode", with the flags worked out and documented. Rule 7 asks for the same model, template, thinking setting, prompts and `max_tokens` everywhere. Decision 0013 left two open questions from M1-T6: repeats 2 and 3 of a B5 probe ran 3.5 times slower than repeat 1, and `mlx_lm.server` returned other token counts than uLLM.

Facts from the installed source, `mlx_lm/server.py` in mlx-lm 0.32.2:

- A request is batchable when the model is, `args.seed is None` and `--kv-bits` is unset (`_is_batchable`, line 768). Every batchable request goes through one `BatchGenerator`, even alone (lines 893–922). A request that is not batchable takes `_serve_single` (line 905). No flag selects `_serve_single` without changing numerics: `--kv-bits` quantizes the KV cache.
- `--decode-concurrency` (default 32) and `--prompt-concurrency` (default 8) cap the `BatchGenerator` (lines 2024–2035). `--prompt-cache-size` defaults to 10 (line 2043).
- `GET /v1/models` lists every MLX repo in the Hugging Face cache, then the `--model` path (lines 1814–1860). A request's `model` that is not `--model` is loaded by that name (line 477), so a repo id loads the cache's `main` ref, not a pinned revision.

Measured on the M4, 8 requests of workload A, greedy, non-streaming, `mlx_lm.server` given the pinned snapshot, compared with `ullm serve` reply by reply (throwaway script, runs in `bench/.scratch/`):

| `mlx_lm.server` path | Prompt cache | Replies identical to uLLM's |
|---|---|---|
| single (`"seed": 0`) | off (`--prompt-cache-size 0`) | 8 of 8 |
| single | default | 4 of 8 |
| batched (default) | off | 2 of 8 |
| batched | default | 0 of 8 |

The first differing character falls mid-reply (a-002 at character 91), so the paths differ by numerics, not by template or stop handling: prompt token counts match on every request. The batched path's log shows it prefills the last few prompt tokens in separate steps (`436/440`, `439/440`, `440/440`), and the default prompt cache reuses the template tokens every prompt starts with. Each splits prefill differently from a single pass, which changes bf16 results at near-ties (decision 0006). That link is inferred from the log and the table, not traced kernel by kernel.

The repeat slowdown was contention, not the engine. In the M1-T6 probe, `env.json` lists a `node` process at 99% CPU, and its `powermetrics` log shows the GPU clock fall from about 1,465 MHz to 450–900 MHz during repeats 2 and 3 with thermal pressure Nominal. The same 8 requests and 3 repeats on a quiet machine (`bench/.scratch/t7-quiet-ullm`, on battery, smoke) gave TPOT p50 16.1 ms in every repeat and 55.5 to 58.3 output tokens per second.

## Decision

**Targets.** `make bench-baselines` runs `bench/baselines.sh`: six B5 runs of all 256 workload A requests, 4 warm-up requests, 3 repeats, 120 s cooldowns, 120 s idle between runs, into `bench/results/m1/<target>-c<concurrency>/`, then `bench/results/m1/comparison.md` from `bench.analysis --compare`.

| Target | Concurrency | Launch | Request fields beyond the common ones |
|---|---|---|---|
| `ullm` | 1 | `ullm serve` | none |
| `mlx-lm-sequential-no-cache` | 1 | `mlx_lm.server --model <snapshot> --prompt-cache-size 0` | `"seed": 0` |
| `mlx-lm-sequential` | 1 | `mlx_lm.server --model <snapshot>` | `"seed": 0` |
| `ullm` | 8 | `ullm serve` | none |
| `mlx-lm-batched` | 8 | `mlx_lm.server --model <snapshot>` | none |
| `mlx-lm-batched-no-cache` | 8 | `mlx_lm.server --model <snapshot> --prompt-cache-size 0` | none |

- **Sequential `mlx_lm.server` is its single-request path**, selected by `"seed": 0` in each request. At temperature 0 the seed changes no token: the sampler is argmax. With the prompt cache off it gives uLLM's exact tokens, so the two compare per token on identical work. The author chose this over flags only.
- **Batched `mlx_lm.server` is its defaults** at concurrency 8. Its replies differ from uLLM's, so their output lengths differ. Throughput counts real tokens from `usage` (rule 7), and the report states the difference.
- **uLLM at concurrency 8** still runs one sequence at a time (`max_num_seqs = 1` until M2). Its TTFT includes queueing. It shows what batching buys before M2 builds it.
- **The prompt cache runs on and off** for both `mlx_lm.server` modes, as the author chose in M1-T6: every repeat resends the same prompts, so the default cache turns repeats into exact prefix hits.

**Harness changes.**

- `bench.run --model <id>` sends that id and refuses a target whose `GET /v1/models` does not list it. The `mlx_lm.server` runs pass the pinned snapshot path, which is the model it loaded. Without it the first listed repo would be sent, and another cached model or revision could load.
- `bench.run --extra-body <json>` adds request fields for one target. It may not replace a field every target gets (`COMMON_FIELDS` in `bench/loadgen.py`). `config.json` records it as `extra_body`, and `summary.md` names it.

**Checks on the results** before they are reported:

- 0 failed requests and 0 prompt mismatches in every run.
- `ullm-c1` and `mlx-lm-sequential-no-cache-c1` agree on `completion_tokens` for every request of every repeat, the whole-workload sign that they generated the same tokens.
- Each run's `powermetrics` log shows no GPU clock drop while busy, and `env.json` lists no busy process. A run that fails this is rerun, not reported.

## Alternatives considered

- **`--decode-concurrency 1 --prompt-concurrency 1` as sequential.** Flags only, and every target gets the same body. But it is still the `BatchGenerator` path, so 6 of 8 replies differ from uLLM's, and at concurrency 1 it is the same path as batched mode.
- **`--kv-bits` to force the single path.** It quantizes the KV cache, so the model differs from the bf16 one every other target runs.
- **`"seed"` for every target.** It would send batched mode down the single path too.
- **Concurrency 1 only, or 64 requests.** The author chose the six-run matrix on all 256 requests. Concurrency sweeps wait for M2-T6.

## Consequences

- Claim 18 names the one exception to "every target gets the same request": fields recorded in `extra_body`.
- The matrix needs about 3 h on an idle Mac on AC power. `powermetrics` needs root without a prompt for that long: a sudoers rule for `/usr/bin/powermetrics` only (decision 0013).
- B5 cannot detect another busy process. `env.json` records the top processes, and the `powermetrics` log shows the GPU clock, so a reader can check both.
- vllm-metal, vllm-mlx and Ollama (the rest of B4) are not in this matrix. M1-T7 names the M1 baselines only.
