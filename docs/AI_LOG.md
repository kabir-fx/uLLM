# AI log

This project is built with an AI coding agent (Claude Code), and the author reviews its work. This log records the agent's mistakes, accurately and without softening (roadmap 3.7): what went wrong, who or what caught it, and what changed afterwards. Entries the author caught are the point of the log; entries caught by tests or CI are kept too, because they show which safeguards work.

| # | Where | What the agent got wrong | Caught by | Fix, and what changed |
|---|---|---|---|---|
| 1 | M0-T6, PR #6 (commit 4873131) | Pushed a commit while `make check` was failing (mypy errors in the new MLX probe). The shell command ran the push after a `grep` of the check output instead of after the check itself, so the failure did not stop it | CI `core` job (mypy step) | Fixed in 81febc1. Since then, every push is chained after `make check` with `&&` |
| 2 | M0-T5, PR #5 | The fixture simulation sent released blocks straight to the free list, so the KV grid fixture had no Cached blocks. That contradicts E10 and E11, where full blocks stay Cached when released | The coverage test `test_prefix_pressure_covers_what_the_kv_grid_needs`, written alongside the fixtures | Released full blocks now stay Cached and are evicted LRU, and a preempted request can hit its own blocks on resume. Fixed before the PR opened |
| 3 | M0-T7, PR #7 | Expected the M4's goldens to reproduce exactly on CI. mlx-lm diverged on the runner's GPU in 5 of 10 prompts, all at near-ties | CI `mlx` job (golden reproduction test) | Equivalence now compares against mlx-lm on the same machine; goldens are strict only on the machine that made them (decision 0006) |

## How to add an entry

Two to four lines per column at most. Name the pull request or commit. If the author caught it, say how they noticed: reading the diff, a test they asked for, a question that had no good answer. Those are the stories interviewers ask for.
