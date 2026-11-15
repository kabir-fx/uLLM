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
