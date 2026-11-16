# 0007: Adopt keel as the session workflow

**Status:** accepted · **Task:** none (repository chore)

## Context

M0 ran on `/task`, a project command that carried one roadmap task from orient to teach-back. It had no independent proof and no review: the agent that wrote a change also judged it done.

The author works with keel, a set of agent rules and skills. `/build` ends a change with `prove`, where a verifier that did not write the change exercises the real artifact, and `vet`, a review against the contract. `/ship` will not open or merge a pull request whose verdicts no longer cover the head commit. Keel's skills read a fast check, a full check and a quality bar from the repo.

The repo had no full check that matched CI. `make check` left out five things the CI jobs run: gofmt, `go test -race`, the Hypothesis `ci` profile, the tests outside `tests/unit` and `tests/property`, and the whole mlx job. Nothing ran a check before a push.

## Decision

1. Every roadmap task runs through keel: `/architect` for a `[design]` task, then `/build`, then `/ship`. Roadmap 3.4 points at them and keeps what belongs to this repo:
   - one task per session
   - approval before production code on `[design]` and `[spike]` tasks
   - anchoring tests first
   - branch names
   - section 0
   - teach-back notes
2. `/task` is removed. `/gate` and `/quiz` stay as the project commands (F3).
3. The roadmap stays the spec. Keel reads the task in section 7 and its catalog entries in section 6.
4. The pull request records the proof and review verdicts with the commit they cover (3.3).
5. `QUALITY.md` at the root names the command behind each part of the bar. Roadmap 3.2 and 3.3 remain the bar.
6. Each CI job runs one Makefile target: `ci-core`, `ci-mlx`, `ci-router` or `ci-terminal`.
   - `make ci` runs all four and is the full check.
   - Three steps stay in the workflow because they only make sense on a hosted runner: the MLX-absent assertion on Linux, the MLX probe and its artifact, and the model cache.
   - Job names are unchanged, so the required checks still match. The job table in decision 0005 still holds.
7. `make check` stays the fast check. `scripts/git-hooks/pre-push` runs it. A clone turns the hook on with `git config core.hooksPath scripts/git-hooks`.
8. Keel's bar guard is not a CI job. It runs when `vet` runs.

## Alternatives considered

- **Keep `/task` beside `/build` and `/ship`.** Two workflows would exist for the same task, and a session could pick the one with no proof or review.
- **Regenerate the spec with keel.** The roadmap already is the spec, and M0 was built and gated against it. A second spec would compete with section 0 as the source of truth.
- **Restart the repo under keel.** It would throw away M0, which passed gate G0, only to change how later tasks run.
- **`make ci` repeating CI's commands, with the workflow unchanged.** It gives two lists that drift apart.
- **The bar guard as a non-required CI job.** The author declined it for now.
- **An untracked `.git/hooks/pre-push`.** It is not reviewed, and it is lost with the clone.

## Consequences

- A commit pushed after the verdicts is not covered by them. `/ship` asks for fresh verdicts before a merge.
- `/build` stops before pushing, and `/ship` asks before every push and merge, as rule 15 requires.
- The keel skills live on the author's machine, not in this repo. A reader with only the repo has 3.4's pointers but not the skills behind them.
- A new CI check goes into its job's Makefile target, not into `ci.yml`. For example, M1-T8's terminal steps go into `ci-terminal`, and M5-T1's chaos test goes into `ci-router`.
- `make ci` needs MLX and the model, so it runs only on Apple Silicon.
- In CI, Ruff's annotations come from `RUFF_OUTPUT_FORMAT=github` on the core job.
