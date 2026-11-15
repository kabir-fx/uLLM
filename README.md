# uLLM

A small, readable LLM inference engine for Apple Silicon, written from scratch on [MLX](https://github.com/ml-explore/mlx), behind a Go router, with a real-time terminal that shows every scheduling decision as it happens.

> **Status: work in progress.** Nothing here is ready to use yet. This README will carry measured results, generated from committed benchmark runs, once they exist. Until then it makes no performance claims.

## What it will contain

- **Engine** (Python, MLX): continuous batching, a block-based KV cache, prefix caching and preemption, behind an OpenAI-compatible streaming API. Greedy output is tested token for token against [mlx-lm](https://github.com/ml-explore/mlx-lm).
- **Router and gateway** (Go): prefix-aware routing across engine replicas, health checks, retries and a circuit breaker.
- **Terminal** (TypeScript, React, canvas): a dense live view of requests, engine steps and KV blocks, with a hosted replay of recorded runs.
- **Proof**: equivalence tests, property tests, quality evals that gate CI, reproducible benchmarks, and a claims ledger that maps every claim to the evidence for it.

## Plan

- [`docs/ROADMAP.md`](docs/ROADMAP.md): the plan in plain language.
- [`docs/AGENT_ROADMAP.md`](docs/AGENT_ROADMAP.md): the full specification and build catalog.

## License

[MIT](LICENSE)
