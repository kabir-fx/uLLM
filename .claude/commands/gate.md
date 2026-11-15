---
description: Run a milestone's gate checklist and report each item as passed or failed, with evidence
argument-hint: <milestone, e.g. M1 or G1>
---

Run the gate for **$ARGUMENTS** from section 7 of `docs/AGENT_ROADMAP.md`.

This is an audit, not a work session. Do not fix anything while running it.

1. Find the milestone and its gate. List every item in the gate, then add these implied items:
   - every task in the milestone is ticked, except those tagged `[user]`, which are listed separately;
   - each ticked task's proof items (from its catalog entries) exist;
   - the decision records the tasks call for exist in `docs/decisions/`;
   - every claim the milestone makes possible has a row in `docs/CLAIMS.md` (from M2 on).
2. Check each item by running something: the make target, the test, the CI run (`gh run list`, `gh pr checks`), or the file. "It should work" is not evidence.
3. Report a table: item · PASSED / FAILED / NOT APPLICABLE · evidence (the command and the key output line, a file path, or a CI run link).
4. Give the verdict: the gate passes only if every item passes. For each failure, name the smallest piece of work that would fix it.
5. If the gate passes, say which section 0 fields and release tag (P4) should change, and offer to run `/quiz $ARGUMENTS`.
