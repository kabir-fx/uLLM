# Decision records

One short file per design decision, numbered in order (roadmap 3.3). A decision records what was chosen, why, what else was considered, and what follows from it. Records are not rewritten later. When a decision changes, a new record supersedes the old one, and the old one's status says so.

| # | Decision | Task |
|---|---|---|
| [0001](0001-repo-conventions.md) | Repository conventions: Go module path, `scripts/`, stacked pull requests merged with merge commits, failing placeholder targets | M0-T1 |
| [0002](0002-python-tooling.md) | Python tooling: uv, MLX only on Apple Silicon, ruff with the rule-11 bans, strict mypy, pytest and hypothesis | M0-T3 |
| [0003](0003-contracts-v0.md) | Contracts v0: the OpenAI API, telemetry protocol v1, the runner protocol sketch | M0-T4 |
| [0004](0004-synthetic-telemetry-fixtures.md) | Synthetic telemetry fixtures and the trace checker | M0-T5 |
| [0005](0005-ci.md) | CI layout, and what MLX can use on GitHub's macOS runners | M0-T6 |
| [0006](0006-models-and-goldens.md) | Pinned models, golden fixtures, and what crossing machines does to them | M0-T7 |
| [0007](0007-adopt-keel.md) | Adopt keel as the session workflow: `/build` and `/ship` replace `/task`; `QUALITY.md`, `make ci`, a pre-push hook; measured figures allowed in documents | none |
| [0008](0008-runner-protocol.md) | Runner protocol v1: lifecycle and position invariants, the hidden `L-1` then `1` prefill split, `MLXRunner` with a concatenating cache, `FakeRunner` with a latency model, an MLX-free `Tokenizer` | M1-T1 |
| [0009](0009-sampler-stop-detokenizer.md) | Sampler keyed by seed and position with masks before temperature, a byte-level streaming detokenizer, a stop check where the earliest-ending stop string wins | M1-T2 |
| [0010](0010-engine-loop-and-thread-bridge.md) | Request state machine with emitting transitions, an engine loop that owns scheduler and runner, an asyncio `EngineClient` that bounds open requests and routes one delivery per engine turn, `ullm sim` checked by the trace checker | M1-T3 |
| [0012](0012-m1-tests.md) | The M1 tests: 50 SDK clients over a real socket, raw SSE framing, margins on an equivalence mismatch, `make equivalence`; no near-tie allowance for uLLM against mlx-lm on one machine, 0.25 only across chips for the goldens | M1-T5 |
| [0013](0013-bench-harness.md) | The benchmark harness: closed-loop raw-SSE client, one C5 directory per run, workload A from UltraChat sized with uLLM's tokenizer, full runs refused unless B5 holds and `powermetrics` logs | M1-T6 |
| [0014](0014-m1-baselines.md) | The M1 baselines: `mlx_lm.server` sequential through its single-request path (`"seed": 0`) and batched at its defaults, prompt cache on and off; `bench.run --model` and `--extra-body`; the repeat slowdown was CPU contention | M1-T7 |
| [0015](0015-mlx-cache-limit.md) | A 256 MB limit on MLX's cache of freed buffers, set by `load_mlx`: without it `ContiguousKV`'s unreusable buffers held 9 GB after 12 requests and the machine swapped | M1-T7 |
| [0018](0018-gate-verdict-coverage.md) | `/gate` accepts a commit after the verdicts when keel's `verdicts.md` covers it: listed under "Covered since" with before and after evidence, or re-checked READY | gate G1 |
| [0019](0019-contiguous-batch-kv.md) | Contiguous batch KV: one left-padded `BatchKV` for the running batch, written in place, repacked on each join and leave; a decode names every decoding sequence and a prefill comes before the first decode | M2-T1 |
| [0020](0020-continuous-batching-admission.md) | Continuous batching: `max_num_seqs` 16 by default; admit the waiting head only while rows × the longest reservation (`prompt + max_tokens`) stays within `max_kv_slots` (40,960, never below `max_model_len`); FCFS, no skipping | M2-T2 |
| [0021](0021-batched-equivalence.md) | Batched equivalence: batch size 1 exact; at 2, 4 and 8 mlx-lm scores every uLLM token on uLLM's own prefix, which must be in its top 2 and within 0.25 logprob; scheduler properties on the real `Engine` with `FakeRunner` | M2-T3 |
| [0022](0022-core-panels.md) | The step chart, batch strip and requests table: own Canvas 2D on `requestAnimationFrame`, no chart library; per-second candles with processed and sampled token bars; strip lanes reused by later requests; a windowed table with fixed 20 px rows | M2-T7 |
| [0024](0024-prometheus-metrics.md) | Prometheus metrics: a `Metrics` registry fed by the telemetry hub on the loop thread, `/metrics` flushes the hub before it renders, gauges from `EngineClient.health()`, token counters count work done, no `/metrics` with telemetry off | M2-T5 |
| [0025](0025-load-launcher.md) | The load launcher: a separate `bench.control` process on 127.0.0.1:8090 with a loopback target fixed at startup runs one load at a time, closed loop or seeded Poisson `open_loop`; the playground calls `/v1/chat/completions` directly; the control types are generated from its JSON Schema | M2-T8 |
| [0026](0026-replay-and-hosted-build.md) | Replay controls and the hosted build: `ReplaySource` with play, pause, speed and seek (a seek back resends the trace from its `hello`); bundled runs in `terminal/src/runs.ts`, labelled from their session's `env.json`; `vite build --mode hosted` plays only bundled runs; `pages.yml` publishes only on `workflow_dispatch` | M2-T9 |
| [0028](0028-benchmark-sweep.md) | The concurrency sweep: one per-chunk session per concurrency 1 to 16 against batched `mlx_lm.server` without prompt cache, uLLM at `--max-num-seqs 16` recorded per chunk; gaps explained by `bench.steps` from the recordings and the offline `scripts.step_costs` | M2-T6 |
| [0029](0029-grafana-dashboard.md) | Grafana for `ullm serve` on the Mac: `make grafana` runs pinned Prometheus and Grafana in Docker on 127.0.0.1, scraping `host.docker.internal:8000` every 5 s, a name the server accepts on `GET /metrics` only; Grafana anonymous Viewer, login off, foreign `Host` redirected; one provisioned dashboard on metrics that exist now | M2-T11 |
| [0030](0030-readme-claims-verify.md) | README v1: `bench.readme` writes the Sustained c1 to c8 table from the committed sweep between markers, and a ci-core test fails when it is stale; `make verify` runs equivalence at 128 tokens, the property tests, `ullm sim` and a bench smoke run; `make demo-gif` records the hosted replay | M2-T12 |
| [0031](0031-gate-renumber-and-merge-commits.md) | `/gate` also accepts a commit after the verdicts that only renumbers decision, row or claim references outside docs, or a merge of the base with no hand-resolved hunk; amends 0018 | gate A |
| [0032](0032-block-manager.md) | Block manager (E8): `BlockManager` owns each `SeqId`'s table; sharing goes through `allocate(..., shared=...)` and only blocks with a key are shared; keys are opaque until E10; eviction is implicit, least recently cached first, a freed table tail first; sizing and metrics in M3-T4 | M3-T1 |
| [0033](0033-kv-layout.md) | KV layout (E9): B, the E7 working tensor plus blocks copied out as they fill; A's gather tax is 3.56× at B=8, 1k tokens; B holds 1.78× E7's KV memory there; C shows no gain with MLX's own operations | M3-T2 |
| [0034](0034-kv-store.md) | KV store (E9): C3 v2 plans carry each sequence's block table and `release` names the freed blocks; the scheduler owns the `BlockManager`; `BlockKV` (layout B) by default, `ContiguousKV` (E7) as the reference; block output equals contiguous output | M3-T3 |
| [0035](0035-eval-harness.md) | Eval harness (Q5): GSM8K's first 200 test items scored by exact match through `ullm serve`, one request at a time, greedy, thinking off; WikiText-2 raw test perplexity on the weights in 1,024-token windows; datasets copied byte for byte and hash-checked; tool calls left to M6-T2 | M4-T1 |

## Template

```markdown
# NNNN: Title

**Status:** proposed | accepted | superseded by NNNN · **Task:** M?-T?

## Context
What forces the decision, in a few lines.

## Decision
What was chosen, specifically enough to check against the code.

## Alternatives considered
Each with the reason it lost.

## Consequences
What this makes easier, harder, or necessary later.
```
| [0027](0027-terminal-quality-gates.md) | Terminal quality gates: a seeded `stress` trace; Playwright smoke and keyboard tests on the hosted build in `check-ts`; `make terminal-perf` checks 60 fps, flat memory over a 10× stream and time to interactive ≤ 1 s, locally; Space and arrow keys in replay, WAI-ARIA tabs, a focus ring | M2-T10 |
