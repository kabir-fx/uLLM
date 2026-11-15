# Upstream candidates

Bugs, documentation gaps and benchmark problems found in MLX, mlx-lm or a baseline engine while building uLLM (roadmap 3.7). One of these becomes the upstream contribution (P5). Add a row the moment something is found, with enough detail to reproduce it later.

| # | Project and version | What | How to reproduce | Status |
|---|---|---|---|---|

Status is one of: noted, reproduced, reported (link), pull request open (link), merged (link), not a bug (why).

## What makes a good candidate

- Small and self-contained: a doc fix, a missing check, a wrong default, a benchmark flag that does not do what it says.
- Reproducible from a short script against the pinned version, and still present on the project's main branch.
- Found through real use in this project, so the pull request description can say why it matters.
