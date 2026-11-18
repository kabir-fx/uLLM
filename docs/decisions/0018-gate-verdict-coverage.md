# 0018: The gate accepts commits that keel's verdicts cover

**Status:** accepted · **Task:** gate G1

## Context

`/gate` (`.claude/commands/gate.md`, written in M0-T2) required that no commit after a pull request's verdict commit change code, tests or check configuration. Decision 0007 later adopted keel. Keel's `verdicts.md` says a verdict also covers a later commit that fixes one of its findings without changing the design, when the finding's reproduction or mutant fails before the fix and passes after, and the full check is green. A design-changing fix gets one scoped re-check. `/ship` lists such commits under "Covered since".

The first `/gate M1` failed 7 of 8 pull requests (#11, #13–#15, #17–#19) on the old rule. Every failing commit was a review fix listed under "Covered since" with its evidence, or a scoped re-check's subject. The old rule would fail any pull request whose review found something, so following it means proof runs again after every review round.

## Decision

The gate item matches keel: each commit after the verdict commit that changes code, tests or check configuration must be covered as `verdicts.md` defines it. The pull request body must name it under "Covered since" with the finding it fixes and the before and after evidence, or a scoped re-check named in the body must have reached READY on it. A later commit that is neither fails the gate.

The same item now finds a pull request by its task ID in the title or by its branch name. Only #18 of the M1 pull requests has the task ID in its title.

## Alternatives considered

- **Keep the old rule and re-prove each pull request at its merged head.** Stricter, but it repeats proof that the "Covered since" evidence already gives for each fix. The author chose against it.
- **Accept any commit the body mentions.** Too loose: a commit named without before and after evidence would pass.

## Consequences

- The proof verdict covers fewer commits directly: a fix is checked by its own failing-then-passing test and green CI, not by a fresh run of the verifier. This is weaker than re-proving, and the author accepted it.
- The gate must read "Covered since" commit by commit, so a missing or evidence-free entry still fails.
- Roadmap 3.6 forbids weakening a gate. This record is the author's explicit call to align the gate's implied item with keel (decision 0007). Gate G1's own text in section 7 is unchanged.
