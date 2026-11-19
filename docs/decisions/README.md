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
