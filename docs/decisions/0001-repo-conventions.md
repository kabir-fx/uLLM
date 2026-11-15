# 0001: Repository conventions

**Status:** accepted · **Task:** M0-T1

## Context

The monorepo layout is fixed by section 5.2 of the agent roadmap. A few things it leaves open need a single answer before work starts.

## Decision

1. **Go module path** is `github.com/kabir-fx/uLLM/router`, rooted in `router/`. The Go module is independent of the Python and TypeScript tooling, so each part builds and tests on its own.
2. **`scripts/`** is added to the layout for maintenance scripts that are not part of the `ullm` package: fetching models, exporting contracts, generating goldens. This keeps helper code out of the importable package and out of the engine line count (rule 10).
3. **Pull requests stack while review is pending.** Each task is one branch (`m0/t1-bootstrap`) and one pull request. When a task starts before the previous pull request is merged, its branch starts from the previous branch and its pull request targets it. Stacks are merged bottom-up with merge commits, not squash, so the remaining pull requests stay mergeable without rewriting history.
4. **Makefile targets that are not built yet fail.** They exit with status 2 and name the task that implements them. Nothing reports success for work that does not exist.

## Consequences

- `main` history contains merge commits, one per task pull request.
- Adding a directory to the layout is a roadmap change (3.6): it goes into 5.2 in the same pull request.
