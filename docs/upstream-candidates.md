# Upstream candidates

Bugs, documentation gaps and benchmark problems found in MLX, mlx-lm or a baseline engine while building uLLM (roadmap 3.7). One of these becomes the upstream contribution (P5). Add a row the moment something is found, with enough detail to reproduce it later.

| # | Project and version | What | How to reproduce | Status |
|---|---|---|---|---|
| 1 | mlx-lm 0.32.2, `mlx_lm.server` | `GET /v1/models` lists every MLX repo in the Hugging Face cache before the `--model` path, and a request naming a listed repo id loads that repo's `main` ref (`server.py` lines 477, 1814–1860). A client that sends the first listed id can get another model or revision than `--model`. | Cache two MLX repos, start `mlx_lm.server --model <snapshot path>`, send the first id from `/v1/models` | noted |
| 2 | mlx-lm 0.32.2, `mlx_lm.server` | No flag serves requests one at a time on the single-request path: every request without a `seed` goes through `BatchGenerator` (`_is_batchable`, line 768). Greedy replies then differ from uLLM's, which equal mlx-lm's own generation loop (claim 14), on 6 of 8 workload A requests, and the default prompt cache changes 4 of 8 even on the single path (decision 0014). | Send workload A greedy with and without `"seed": 0`, with and without `--prompt-cache-size 0`, and compare with `ullm serve` | noted |
| 3 | mlx-lm 0.32.2, `mlx_lm.server` | The server listens with `socketserver`'s queue of 5 (`ThreadingHTTPServer`, `request_queue_size`) and has no flag to change it. While it is busy, macOS refuses connections past the queue: the first `mlx-lm-batched-c8` run of workload A lost 4 of 768 requests to `ConnectError`, all opened together after a cooldown. | `tests/integration/test_mlx_lm_server_backlog.py` against `mlx_lm.server` instead of `scripts/mlx_lm_server.py`: stop the server with `SIGSTOP` and open 16 connections at once; 5 connect | reproduced |
| 4 | mlx-lm 0.32.2, `mlx_lm.server` | At concurrency 16 on workload A (16 GB M4), the batched generation thread died on `kIOGPUCommandBufferCallbackErrorOutOfMemory`. The process stayed up and answered every later request with HTTP 404 `{"error": "generation thread died"}`. | `bench.run --requests 32 --repeats 2 --concurrency 16` against `mlx_lm.server --model <Qwen3-0.6B bf16 snapshot>` | noted |
| 5 | mlx-lm 0.32.2, `mlx_lm.server` | Batched `mlx_lm.server` holds 6.78 GB of `phys_footprint` without its prompt cache and 9.23 GB (peak 9.68) with it, for Qwen3-0.6B bf16 at concurrency 8 on a 16 GB M4, against 1.97 GB peak for uLLM. `BatchGenerator` clears MLX's buffer cache only after each prefill step and every 512 decode steps (`mlx_lm/generate.py` lines 1230, 1238, 1824), and the server raises the wired limit to the recommended working set (`server.py` line 2080), so freed buffers pile up in between. | `bench/results/m1/c8/probes.jsonl`: the M1-T7 c8 session, 10 chunks of 24 workload A prompts at concurrency 8 per server (decision 0016) | noted |

Status is one of: noted, reproduced, reported (link), pull request open (link), merged (link), not a bug (why).

## What makes a good candidate

- Small and self-contained: a doc fix, a missing check, a wrong default, a benchmark flag that does not do what it says.
- Reproducible from a short script against the pinned version, and still present on the project's main branch.
- Found through real use in this project, so the pull request description can say why it matters.
