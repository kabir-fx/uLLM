# uLLM

A proof-of-work project: a small, readable LLM inference engine for Apple Silicon written from scratch on MLX, a Go router, and a dense real-time terminal, with tests, evals and benchmarks that prove each claim. It is not a product and does not compete with other engines; they are baselines.

## Start here

`docs/AGENT_ROADMAP.md` is the single source of truth. Read its **section 0 (current state)** first, then the task in section 7 and the catalog entries (section 6) it names. Update section 0 at the end of every session. Other files in `docs/` are background; the roadmap wins over them.

## Commands

- `make check`: lint, types, unit and property tests. No model needed. Must pass before every pull request.
- `make test`, `make equivalence` (needs the model), `make bench-smoke`, `make sim`, `make up`, `make verify`.
- Targets not built yet exit 2 and name the task that builds them.
- Python runs through `uv` (`uv run ...`). The router is a Go module in `router/`. The terminal uses `pnpm` in `terminal/`.
- Project commands: `/task <ID>` starts a task, `/gate <milestone>` runs a gate, `/quiz <milestone>` quizzes the author.

## Rules in brief (full text: roadmap 3.2)

1. Greedy output equals mlx-lm's token IDs. Never loosen the equivalence test to make it pass.
2. Only the engine thread touches MLX, the scheduler and KV state. Threads talk through queues. No locks in `core`.
3. `ullm.core` never imports `mlx`. A test enforces it.
4. Call `mx.eval` at the end of each step; never take a timing before it.
5. Never block the engine thread on telemetry, metrics, logging or the network.
6. No request IDs as Prometheus labels.
7. No invented numbers: every figure comes from a committed result. Unmeasured means a placeholder.
8. Tests ship with behaviour, in the same pull request.
9. Readable over clever. Comments explain why, not what.
10. Build the machinery, do not import it: no `generate_step`, `stream_generate`, `batch_generate`, `BatchGenerator`, `BatchKVCache` or `LRUPromptCache` in engine code.
11. Verify library APIs against the installed source in `.venv`, not memory.
12. Pin dependencies; justify each new one in the pull request.
13. Bind to 127.0.0.1. No secrets in the repo.
14. Ask the author before anything outward-facing or hard to undo: pushing, publishing, deleting files, force operations.

## How a session runs (roadmap 3.4)

- One task per session; stop at the task boundary.
- `[design]` and `[spike]` tasks: present the design or measurement plan and wait for approval before production code.
- Write the anchoring tests first. Branch `m<N>/t<K>-<slug>`, conventional commits, one pull request per task; merge only when CI is green.
- Record design choices in `docs/decisions/NNNN-title.md`. New claims get a row in `docs/CLAIMS.md`.
- End with teach-back notes for the author: up to seven bullets, three interview questions, open doubts.
- Do not write `docs/DESIGN.md`; the author writes it.

## Writing

- No dates, deadlines or time estimates in any document. A milestone is done when its gate passes.
- No novelty claims ("first", "only"). Slower than a baseline is fine when measured and explained.
