# 0031: The gate accepts renumbering commits and clean merges of the base

**Status:** accepted · **Task:** gate A

## Context

Decision 0018 lets a commit after the verdicts pass the gate only as a fix of a verdict's finding or as a record of the result. The first `/gate M2` failed on three commits that are neither:

- `be3c604` (#26) and `240abb2` (#30) renumber a decision and claims rows after a merge of `main` took their numbers. In code and test files, the removed and added lines are identical once the number after "decision", "row" or "claim" is replaced (9 and 8 lines).
- `9834d77` (#30) merges `main` into the branch. `git show --cc` shows no hunk, so no line was resolved by hand.

Neither kind changes behaviour, and pull requests stack, so both will recur. Every other gate item for M2 passed.

## Decision

Amends 0018. A commit after the verdicts also passes the gate when it is one of these:

- **A renumbering.** In every file outside `docs/` and other than `*.md`, each changed line differs from the line it replaces only in a number after "decision", "decisions", "row", "rows", "claim" or "claims", and the pull request body names the commit as a renumbering.
- **A clean merge of the base.** A merge commit whose second parent is the base branch, already through its own CI, and for which `git show --cc <commit>` prints no hunk.

Any other change in such a commit, such as a numeric constant or a hand-resolved conflict, fails the gate as before. The author accepted `be3c604`, `240abb2` and `9834d77` on this rule.

## Alternatives considered

- **A scoped review of the three commits.** It repeats what a mechanical diff check settles. The author chose against it.
- **Accept any commit that changes only digits.** Too loose: it would pass a changed timeout or threshold.

## Consequences

- `/gate` checks both conditions by running them, not by reading the pull request body.
- A renumbering that also rewords a comment fails; split it or name it under "Covered since" with its own evidence.
- Gate A's other two items, `docs/DESIGN.md` for M1 and M2 and the quiz (M2-T13), are left to the author. M3 starts before they are done, at the author's call.
