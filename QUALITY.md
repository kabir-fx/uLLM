# Quality bar

A change may never weaken this file, a check, or a test to get green. If a line here is wrong, say so and propose the edit on its own.

The bar is the engineering rules and the definition of done in `docs/AGENT_ROADMAP.md` sections 3.2 and 3.3. This file names the commands that enforce them and adds what they do not say.

## Checks

| Check | Command | When | Typical time |
|---|---|---|---|
| Fast | `make check` | after every increment; the pre-push hook runs it | 20 s on an M4, Chromium installed |
| Full | `make ci` | before proof, before a pull request | 166 s on an M4 with the model downloaded |
| Performance | `make bench-smoke` | a performance-relevant change: on the base before the first commit, and again after | 82 s on an M4, model load included |

`make bench-smoke` skips the B5 checks, so its numbers are for before and after only, never a claim. Run both on the same power source: the summary records AC, battery and Low Power Mode. The pull request carries both summary tables (roadmap 3.3).

`make ci` runs the `ci-core`, `ci-router`, `ci-terminal` and `ci-mlx` targets, and each CI job runs its own target (decision 0007). `ci-mlx` runs `make equivalence` first, then the other tests marked `mlx`. It needs MLX and the model (`make models`).

## Floor

Always true on `main`. A change that breaks one of these does not merge.

- `make ci` passes, and the four required CI jobs are green.
- Ruff lint and format are clean, and strict mypy finds no errors: `ci-core`.
- Engine code imports none of mlx-lm's generation loop, batching or caches (rule 11): the `TID251` bans in `pyproject.toml`, run by `ci-core`.
- `ullm.core` never imports MLX (rule 3): `tests/unit/test_core_is_mlx_free.py`, run by `ci-core`.
- Every test outside the `mlx` marker passes with MLX absent: the core job's MLX-absent step on Linux, then `ci-core`.
- Greedy output equals mlx-lm's token IDs through the runner, the engine and the HTTP server (rule 1): `make equivalence`, run by `ci-mlx`.
- Batched greedy output at `max_num_seqs` 2, 4 and 8 keeps every token in mlx-lm's top 2 and within 0.25 logprob on uLLM's own prefix (decision 0021): `make equivalence`, run by `ci-mlx`.
- mlx-lm reproduces the committed goldens, and a missing model fails instead of skipping: `ci-mlx`.
- Lockfiles match their manifests: `uv run --locked` and `pnpm install --frozen-lockfile`, in `ci-core` and `ci-terminal`.
- The generated TypeScript contract types match the schema, and the terminal typechecks strictly: `ci-terminal`.
- The hosted replay opens, plays and draws in headless Chromium with no console error (decision 0027): `terminal/e2e/`, run by `ci-terminal`.
- The README's benchmark table is what `bench.readme` makes from the committed sweep (decision 0030): `tests/integration/test_readme_tables.py`, run by `ci-core`.
- Go code is gofmt-clean and vet-clean, and its tests pass under the race detector: `ci-router`.

## Ratchets

Each may only move in the direction shown. Record the new value when a change improves one.

- **Suppression comments** (`type: ignore`, `noqa`, `eslint-disable`, `@ts-ignore`, `nolint`, `pragma: no cover`) in tracked code: **9**, down. Count:

  ```sh
  git grep -nE 'type: *ignore|noqa|eslint-disable|@ts-(ignore|expect-error|nocheck)|nolint|pragma: *no cover' -- src tests scripts bench router terminal | wc -l
  ```

## Review standards

- Each rule in roadmap 3.2 that no check above covers holds for the change: one owner thread (2), `mx.eval` before any timing (4), the engine thread never blocks (5), low-cardinality labels (6), fair benchmarks (7), no invented numbers (8), tests ship with behaviour (9), local binding and no secrets (14).
- `core` plus `runner` stays near 2,500 lines (rule 10). Today: 1,698. Count: `find src/ullm/core src/ullm/runner -name '*.py' | xargs wc -l`.
- A Makefile target that is not built yet exits 2 and names the task that builds it (decision 0001).
- Every GitHub Action is pinned to a commit SHA, with its release named in a comment (decision 0005).
- A new CI check goes into its job's `make ci-*` target, not into `ci.yml` (decision 0007).
