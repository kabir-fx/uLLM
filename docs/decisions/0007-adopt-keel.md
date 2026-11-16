# 0007: Adopt keel: a quality bar file, one list of CI commands, a pre-push hook

**Status:** accepted · **Task:** none (repository chore)

## Context

The author works with keel, a set of agent rules and skills. Keel's skills read three things from a repo: a fast check, a full check, and a file that states the quality bar.

Before this change, passing locally did not predict CI. `make check` left out five things the CI jobs run: the gofmt check, `go test -race`, the Hypothesis `ci` profile (500 examples, not 100), the tests outside `tests/unit` and `tests/property`, and the whole mlx job. Nothing ran a check before a push.

## Decision

1. `QUALITY.md` at the root names the command that enforces each part of the bar. Roadmap 3.2 and 3.3 remain the bar. `QUALITY.md` links to them and adds only what they do not say. Roadmap 5.2 lists the file.
2. Each CI job runs one Makefile target: `ci-core`, `ci-mlx`, `ci-router` or `ci-terminal`. `make ci` runs all four and is the full check. Three steps stay in the workflow because they only make sense on a hosted runner: the MLX-absent assertion on Linux, the MLX probe and its artifact, and the model cache. Job names are unchanged, so the required checks on `main` still match. The job table in decision 0005 still holds.
3. `make check` stays the fast check. `scripts/git-hooks/pre-push` runs it. A clone turns the hook on with `git config core.hooksPath scripts/git-hooks`.
4. Keel's bar guard is not added as a CI job. It runs only when an agent runs keel's review.

## Alternatives considered

- **A `make ci` target that repeats CI's commands, with the workflow unchanged.** It gives two lists that drift apart.
- **The bar guard as a non-required CI job.** It would list each pull request's new suppressions, skipped tests and stubs on the diff. The author declined it for now.
- **An untracked `.git/hooks/pre-push`.** It is not reviewed, and it is lost with the clone.

## Consequences

- A new CI check goes into its job's Makefile target, not into `ci.yml`. For example, the terminal's lint, test and build steps in M1-T8 go into `ci-terminal`, and the router chaos test in M5-T1 goes into `ci-router`.
- `make ci` needs MLX and the model, so it runs only on Apple Silicon.
- In CI, Ruff's annotations come from `RUFF_OUTPUT_FORMAT=github` on the core job, not from a flag in the Makefile.
