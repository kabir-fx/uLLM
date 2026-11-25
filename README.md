# uLLM

A small, readable LLM inference engine for Apple Silicon, written from scratch on [MLX](https://github.com/ml-explore/mlx), behind a Go router, with a real-time terminal that shows every scheduling decision as it happens.

> **Status: work in progress (foundations done; the engine is next).** Nothing here serves a model yet. This README makes no performance claims; measured results will appear here only once they are generated from committed benchmark runs. Every claim the project makes is listed, with its evidence, in the [claims ledger](docs/CLAIMS.md).

## What it will contain

- **Engine** (Python, MLX): continuous batching, a block-based KV cache, prefix caching and preemption, behind an OpenAI-compatible streaming API. Greedy output is tested token for token against [mlx-lm](https://github.com/ml-explore/mlx-lm) run on the same machine.
- **Router and gateway** (Go): prefix-aware routing across engine replicas, health checks, retries and a circuit breaker.
- **Terminal** (TypeScript, React, canvas): a dense live view of requests, engine steps and KV blocks, with a hosted replay of recorded runs.
- **Proof**: equivalence tests, property tests, quality evals that gate CI, reproducible benchmarks, and a claims ledger that maps every claim to the evidence for it.

## What exists today

- **Contracts:** the OpenAI-compatible API models (checked against the official OpenAI SDK), the telemetry protocol with generated TypeScript types, and a sketch of the runner boundary ([decision 0003](docs/decisions/0003-contracts-v0.md)).
- **Telemetry fixtures:** synthetic, clearly labelled traces and a checker that verifies a trace tells a consistent story ([decision 0004](docs/decisions/0004-synthetic-telemetry-fixtures.md)).
- **CI:** Linux jobs for the core (with MLX absent), the router and the terminal, and a macOS job that runs MLX on GitHub's hosted GPU ([decision 0005](docs/decisions/0005-ci.md)).
- **Goldens:** pinned models and mlx-lm reference outputs with per-token margins ([decision 0006](docs/decisions/0006-models-and-goldens.md)).

## Benchmarks

<!-- bench.readme: generated from bench/results/m2/sweep by `make readme` -->
| uLLM ÷ `mlx-lm-batched-no-cache`, Sustained | c1 | c2 | c4 | c8 |
|---|---|---|---|---|
| Output tok/s | 1.006 [0.994, 1.020] | 1.079 [1.053, 1.099] | 1.093 [1.050, 1.117] | 1.120 [1.086, 1.170] |
| TTFT | 0.841 [0.793, 0.863] | 0.548 [0.505, 0.586] | 0.465 [0.424, 0.526] | 0.424 [0.375, 0.482] |
| TPOT | 1.014 [1.008, 1.024] | 1.003 [0.995, 1.027], within noise | 1.044 [1.018, 1.080] | 1.008 [0.964, 1.038], within noise |
| E2E | 0.989 [0.981, 0.999] | 0.919 [0.900, 0.938] | 0.912 [0.887, 0.930] | 0.891 [0.844, 0.921] |
| Prompts / blocks | 204 / 17 | 240 / 20 | 228 / 19 | 216 / 9 |

Median [95% block-bootstrap interval]; below 1 is faster for TTFT, TPOT and E2E, above 1 for output tok/s. `mlx-community/Qwen3-0.6B-bf16`, workload A, closed loop. Apple M4, 16 GB, macOS 27.2.1; mlx 0.32.4, mlx-lm 0.32.2; commit `c51200e`. Every session VALID under the run conditions (decision 0016).
<!-- bench.readme: end -->

## Development

Needs an Apple Silicon Mac for anything that runs the model; everything else also runs on Linux. Tools: [uv](https://docs.astral.sh/uv/), Go 1.25, Node 24 and pnpm.

```sh
make check      # lint, types, unit and property tests, contract freshness (no model needed)
make models     # download Qwen3-0.6B at its pinned revision (about 1.2 GB)
make test-mlx   # tests that need MLX and the model, including golden reproduction
make            # list every target
```

## Repository

| Path | What |
|---|---|
| `src/ullm/` | The engine: `core/` (MLX-free decisions), `runner/` (tensors), `server/`, `telemetry/` |
| `tests/` | `unit/`, `property/`, `equivalence/` (goldens), `integration/` |
| `router/` | The Go router and gateway |
| `terminal/` | The real-time terminal, and its fixture traces |
| `scripts/` | Model fetch, contract export, fixture and golden generation, the MLX probe |
| `docs/` | The [plan](docs/ROADMAP.md), the [full specification](docs/AGENT_ROADMAP.md), [decisions](docs/decisions/), the [claims ledger](docs/CLAIMS.md), the [AI log](docs/AI_LOG.md) |

## License

[MIT](LICENSE)
