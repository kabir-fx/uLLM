# 0018: The gate accepts commits that keel's verdicts cover

**Status:** accepted, amended by 0031 · **Task:** gate G1

## Context

Decision 0007 adopted keel and, in the same pull request (#10), gave `/gate` an implied item: no commit after a pull request's verdict commit may change code, tests or check configuration. 0007's Context states the same rule for `/ship`.

Keel's `verdicts.md` is less blunt. A verdict also covers a later commit that does one of two things:
- fixes a finding of that verdict without changing the design, with the finding's reproduction or mutant failing before the fix and passing after it, and the full check green;
- records the result.

A fix that changes the design gets one scoped re-check. `/ship` lists covered commits under "Covered since".

The first `/gate M1` failed 7 of 8 pull requests (#11, #13–#15, #17–#19) on the old rule. In six of them every failing commit is named under "Covered since" or is the subject of a scoped re-check. #11 has no "Covered since" (see Consequences). Under the old rule, any pull request whose review found something would fail, so proof would have to run again after every review round.

## Decision

The gate item matches keel's `verdicts.md`, and amends 0007's statement of the rule. Each commit after the verdict commit that changes code, tests or check configuration must be named in the pull request body under "Covered since", in one of two ways:
- as a fix of a verdict's finding that does not change the design, with the reproduction or mutant failing before and passing after, and the full check green;
- as recording the result.

The other way to pass is a scoped re-check named in the body that reached READY on the commit. Any other later commit fails the gate.

The same item now finds a pull request by its task ID in the title or by its branch name. Only #18 of the M1 pull requests has the task ID in its title.

## Alternatives considered

- **Keep the old rule and re-prove each pull request at its merged head.** Stricter, but it repeats the proof that each fix's before and after evidence already gives. The author chose against it.
- **Accept any commit the body mentions.** Too loose: a design change, or a commit named without evidence, would pass.

## Consequences

- A fix made after the verdicts is checked by its own failing-then-passing test and a green full check, not by a fresh run of the verifier. This is weaker than re-proving. Roadmap 3.6 forbids weakening a gate, so this record is an exception the author approved: the gate now enforces the workflow 0007 adopted, not a stricter one.
- The gate reads "Covered since" commit by commit, so a missing entry, or one without evidence, still fails.
- Gate G1's own text in section 7 is unchanged.
- #11 (M1-T1) passes as an exception the author approved. After its proof commit `4076d5e`, `03cb260` adds a test and `3aa4e42` rewords a docstring in `src/ullm/runner/base.py`. Its body names neither under "Covered since" and gives no before and after evidence, but its full review reached READY at the head `3dcd616`, after both. No later pull request may pass this way.
