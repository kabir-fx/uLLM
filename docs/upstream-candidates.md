# Upstream candidates

Bugs, documentation gaps and benchmark problems found in MLX, mlx-lm or a baseline engine while building uLLM (roadmap 3.7). One of these becomes the upstream contribution (P5). Add a row the moment something is found, with enough detail to reproduce it later.

| # | Project and version | What | How to reproduce | Status |
|---|---|---|---|---|
| 1 | mlx-lm 0.32.2, `mlx_lm.server` | `GET /v1/models` lists every MLX repo in the Hugging Face cache before the `--model` path, and a request naming a listed repo id loads that repo's `main` ref (`server.py` lines 477, 1814–1860). A client that sends the first listed id can get another model or revision than `--model`. | Cache two MLX repos, start `mlx_lm.server --model <snapshot path>`, send the first id from `/v1/models` | noted |
| 2 | mlx-lm 0.32.2, `mlx_lm.server` | No flag serves requests one at a time on the single-request path: every request without a `seed` goes through `BatchGenerator` (`_is_batchable`, line 768). Greedy replies then differ from uLLM's, which equal mlx-lm's own generation loop (claim 14), on 6 of 8 workload A requests, and the default prompt cache changes 4 of 8 even on the single path (decision 0014). | Send workload A greedy with and without `"seed": 0`, with and without `--prompt-cache-size 0`, and compare with `ullm serve` | noted |

Status is one of: noted, reproduced, reported (link), pull request open (link), merged (link), not a bug (why).

## What makes a good candidate

- Small and self-contained: a doc fix, a missing check, a wrong default, a benchmark flag that does not do what it says.
- Reproducible from a short script against the pinned version, and still present on the project's main branch.
- Found through real use in this project, so the pull request description can say why it matters.
