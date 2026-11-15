---
description: Start a roadmap task by ID and carry it through the session loop in roadmap 3.4
argument-hint: <task ID, e.g. M1-T3>
---

Start task **$ARGUMENTS** from `docs/AGENT_ROADMAP.md`. Follow roadmap section 3.4 exactly.

1. **Orient.** Read section 0 (current state). Find $ARGUMENTS in section 7 and read every catalog entry (section 6) it names, plus any contract (6.2) it touches. If section 0 says another task is in flight, or the task's prerequisites are not merged, stop and say so.
2. **Restate.** In a few lines: what the task delivers, its proof items, and the definition of done (3.3) as it applies here.
3. **Design gate.** If the task is tagged `[design]` or `[spike]`, write the design or measurement plan (interfaces, data shapes, alternatives with trade-offs, what will be measured and the decision rule) and **stop for the author's approval** before any production code.
4. **Verify libraries.** Before designing against MLX or mlx-lm, read the installed source in `.venv` for the pinned version (rule 12).
5. **Anchors first.** Where an anchor exists (equivalence, invariants, contract tests), write those tests before the implementation.
6. **Build** on a branch named `m<N>/t<K>-<slug>` in small conventional commits. Ask before pushing if this session has no standing approval to push.
7. **Check.** Run `make check` (and `make equivalence` for engine changes, `make bench-smoke` before and after for performance-relevant changes). Report failures in full.
8. **Close out.** Open the pull request. Add decision records under `docs/decisions/`, a `docs/CLAIMS.md` row if a new claim became possible, update section 0, and tick the task in section 7.
9. **Teach back.** End with up to seven bullets on what was built and why it is shaped that way, three questions an interviewer could ask about it, and anything you were unsure of.

One task per session. Stop at the task boundary.
