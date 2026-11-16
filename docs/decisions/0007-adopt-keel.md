# 0007: Adopt keel as the session workflow

**Status:** accepted · **Task:** none (repository chore)

## Context

M0 ran on `/task`, a project command that carried one roadmap task from orient to teach-back. It had no independent proof and no review: the agent that wrote a change also judged it done.

The author works with keel, a set of agent rules and skills.
- `/build` ends a change with two skills: `prove`, which exercises the real artifact, and `vet`, a review against the contract. `/build` picks a rigor from the change's stakes.
- At every rigor, `vet`'s reviewers are agents that did not write the change. From Standard rigor up, an independent verifier also runs the proof.
- `/ship` asks for fresh verdicts when code, tests or check configuration changed after them.
- Keel's skills read a fast check, a full check and a quality bar from the repo.

The repo had no full check that matched CI. `make check` left out five things the CI jobs run: gofmt, `go test -race`, the Hypothesis `ci` profile, the tests outside `tests/unit` and `tests/property`, and the whole mlx job. Nothing ran a check before a push.

## Decision

1. Every roadmap task except `[user]` tasks runs through keel: `/architect` for a `[design]` task, then `/build`, then `/ship`.
   - `/build` and `/ship` are user-only skills: the author types them, and an agent outside `/build` writes no production code for a task.
   - Roadmap 3.4 points at the skills and keeps what belongs to this repo:
     - one task per session
     - the in-flight and prerequisite check
     - approval before production code on `[design]` and `[spike]` tasks
     - anchoring tests first
     - `make equivalence` and `make bench-smoke` where 3.3 requires them
     - small conventional commits
     - branch names and stacked bases
     - section 0
     - the teach-back notes
2. `/task` is removed. `/gate` and `/quiz` stay as the project commands (F3).
3. The roadmap stays the spec. Keel reads the task in section 7 and its catalog entries in section 6.
4. The pull request body is the one verdict record. It holds VERIFIED proof and READY review verdicts, with the commit they cover (3.3).
   - `/ship` writes it and updates it when the verdicts are renewed. `/gate` reads it with `gh pr view`.
   - `/build`'s done marker is the tick in section 7. The roadmap has no `Status:` lines.
   - A `/ship` started in a new chat cannot see `/build`'s report, so it runs proof and review again.
   - No history rewrite after the verdicts, so the commit they name stays on the branch.
5. `/build` keeps choosing the rigor. The repo sets no minimum.
6. `QUALITY.md` at the root names the command behind each part of the bar. Roadmap 3.2 and 3.3 remain the bar.
7. Each CI job runs one Makefile target: `ci-core`, `ci-mlx`, `ci-router` or `ci-terminal`.
   - `make ci` runs all four and is the full check.
   - Three steps stay in the workflow because they only make sense on a hosted runner: the MLX-absent assertion on Linux, the MLX probe and its artifact, and the model cache.
   - Job names are unchanged, so the required checks still match. The job table in decision 0005 still holds.
8. `make check` stays the fast check. `scripts/git-hooks/pre-push` runs it. A clone turns the hook on with `git config core.hooksPath scripts/git-hooks`.
9. Keel's bar guard is not a CI job. It runs when `vet` runs.
10. Roadmap section 1's no-deadlines rule allows measured figures. `QUALITY.md` records how long the checks take. The rule exists to keep deadlines and schedules off the project, not measurements.

## Alternatives considered

- **Keep `/task` beside `/build` and `/ship`.** Two workflows would exist for the same task, and a session could pick the one with no proof or review.
- **Regenerate the spec with keel.** The roadmap already is the spec, and M0 was built and gated against it. A second spec would compete with section 0 as the source of truth.
- **Restart the repo under keel.** It would throw away M0, which passed gate G0, only to change how later tasks run.
- **A minimum rigor of Standard for roadmap tasks.** It would guarantee an independent verifier on every task. The author left the choice to `/build`, which raises the rigor for concurrency, public interfaces and anything hard to undo.
- **The verdict commit on the task's tick in section 7.** It would be a second record, and it goes stale. A merge from `main` renews the verdicts only in the pull request body, and tidying history before the push rewrites the hash.
- **`make ci` repeating CI's commands, with the workflow unchanged.** It gives two lists that drift apart.
- **The bar guard as a non-required CI job.** It would list each pull request's new suppressions, skipped tests and stubs on the diff. The author declined it for now.
- **An untracked `.git/hooks/pre-push`.** It is not reviewed, and it is lost with the clone.

## Consequences

- The repo overrides two keel steps through the Keel block in `CLAUDE.md`: `/build` writes no `Status: done` line (its step 4.7), and `/ship` does not tidy history after the verdicts (its step 2).
- At Light rigor, the building agent proves its own change, so the proof is not independent for those tasks. The review still is.
- A commit after the verdicts that touches code, tests or check configuration needs fresh verdicts. `/ship` checks for one with `git diff --stat <verdict commit>..HEAD`.
- `/build` stops before pushing, and `/ship` asks before every push and merge, as roadmap rule 15 (3.2) requires.
- The keel skills live on the author's machine, not in this repo. A reader with only the repo has 3.4's pointers but not the skills behind them.
- A new CI check goes into its job's Makefile target, not into `ci.yml`. For example, M1-T8's terminal steps go into `ci-terminal`, and M5-T1's chaos test goes into `ci-router`.
- `make ci` needs MLX and the model, so it runs only on Apple Silicon.
- In CI, Ruff's annotations come from `RUFF_OUTPUT_FORMAT=github` on the core job.
