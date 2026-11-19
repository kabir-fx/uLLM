# M2-T3: Batching tests

**State:** DONE at `c73fad4`. Read the Result, then run `/ship`.
**Decision record:** [0021](../decisions/0021-batched-equivalence.md) (the full design, for agents) · **Spec:** roadmap Q2 and Q3, task M2-T3

## The goal

Tests will prove that batched greedy replies agree with mlx-lm at batch sizes 1, 2, 4 and 8. Tests will also prove that the scheduler ends every request, keeps FCFS order, and leaks nothing.

## The problem

- Since M2-T2, `ullm serve` batches by default. But the engine equivalence test used only `max_num_seqs=1`.
- A batched forward runs its matrix operations at a different shape. In bf16, a near-tie can then select the other token.
- Rule 1 and decision 0012: batch size 1 stays exact. A batch size above 1 needs a threshold from a measurement.
- Rule 1 also says: never loosen the equivalence test to make it pass.
- Q2: no test checked the scheduler over random arrivals, aborts and limits.

## How it works

1. `tests/equivalence/test_engine_equivalence.py` submits all 10 golden prompts at once, with `max_num_seqs` 1, 2, 4 and 8. It asserts that the widest step had exactly that many rows.
2. At batch size 1, each reply must equal mlx-lm's tokens exactly.
3. Above 1, mlx-lm reads the prompt and then uLLM's own tokens. At each position, uLLM's token must be in mlx-lm's top 2. It must also be at most 0.25 logprob behind mlx-lm's best token.
4. `tests/property/test_scheduler.py` runs the real `Engine` on `FakeRunner` with Hypothesis. It draws the limits and up to 24 loop turns of submits and aborts.
5. A wrapper runner checks the row count, the reserved slots and the written slots on each step. It also checks that a decode names every held row.
6. After each run, the test checks that each request ended once and that no request skipped ahead. It also checks that no sequence leaked.

The agent measured the threshold on your M4, with all 10 golden prompts.

- At 32 tokens, 7 to 8 of 10 replies matched mlx-lm exactly at batch sizes 2, 4 and 8.
- At each difference, uLLM selected mlx-lm's second choice.
- The worst gap was 0.125 at 32 tokens, and 0.25 at batch 8 and 128 tokens.
- Two runs gave the same tokens. Thus batched output is deterministic for one arrival order.

The agent also measured a mask that ignores the padding. Its gaps stayed at 0.125 at 32 tokens. But it selected mlx-lm's third choice for `emoji`. Thus the top-2 rule catches this bug, and a gap rule alone does not.

## What you get at the end

Run `make equivalence` to see the four batch sizes pass. Run `make check` for the scheduler property test.

- The equivalence test fails on a mask that ignores padding, and on RoPE positions that ignore padding.
- The property test fails on six scheduler and engine bugs:
  - no slot budget, or one row too many
  - a count of current lengths in place of reservations
  - a request that skips ahead
  - an abort that leaks its sequence
  - a decode that drops a row

What you do not get yet:

- A batched test over HTTP. The server test still sends one request at a time.
- A slot budget check in `check_trace`. The `hello` event has no field for the budget.

## Where this puts the project

```
M0  █████████
M1  ████████
M2  ██▓░░░░░░░░░░
```

This slice opens M2-T6, the benchmark sweep. Batched replies are then proven before the sweep measures them.

## What you approve

You chose the rule before the agent wrote the test.

- Every token is scored, not only the first difference. The first-difference rule leaves 12 of 32 `palindrome` tokens unchecked.
- The top-2 rule stays. Without it, the padding mask bug passes at 32 tokens.
- The threshold is 0.25. A threshold of 0.125 fails correct code at batch 8 and 128 tokens.

## What can go wrong

- The top-2 rule catches the mask bug on one token of one prompt. A change to the prompts can hide it. `tests/property/test_batch_kv.py` still checks the mask values.
- The 0.25 threshold comes from one machine and one model. Another chip can move a near-tie further. The test then fails, and you must measure again.
- `make equivalence` takes about 38 s for the engine test at 32 tokens, against about 10 s before.

## Words

| Word | Meaning |
|---|---|
| near-tie | A position where the two most likely tokens have almost the same logit. |
| teacher-forced | mlx-lm reads uLLM's tokens as its input, and scores each one. |
| gap | mlx-lm's best logprob minus the logprob of uLLM's token, at one position. |

## Learn from this slice

A comparison that stops at the first difference checks only a prefix. When mlx-lm scores uLLM's own tokens, every position gets a check.

A tolerance must be measured on correct code and on a known bug. Here, the gap of the bug was inside the noise. The rank of the token told them apart.

Test yourself: why is the gap exactly 0 on a prefix where uLLM matches mlx-lm?

<details><summary>Answer</summary>

On that prefix, mlx-lm reads the same tokens that its own `generate_step` reads, in the same order. Thus it does the same arithmetic, and its best token is uLLM's token.

</details>

## Result

**State:** DONE. The proof is VERIFIED at `461d2b3`. The review is READY at `c73fad4`, after the fixes. `make ci` passed at `817a5bd`.

**Proof.** One reviewer agent exercised six claims on a copy of the repo. All six passed.
- `make equivalence` passed, and each batch size reached its row count.
- A tie-break mutant in the sampler failed batch size 1.
- The padding mask mutant failed batch sizes 2, 4 and 8. The RoPE mutant failed with gaps above 10.
- At 128 tokens, the engine test passed.
- Six scheduler and engine mutants failed the property test in 30 of 30 runs.

**Changes against the plan.**
- The review found that rule 1, Q3 and R-2 in the roadmap still gave the old rule. They now cite decision 0021.
- The review found claims 23 and 24 below the table in `docs/CLAIMS.md`. They are now rows of the table.
- The property test first fixed its own example count. It now uses the Hypothesis profile of the repo.

**Not proven.** Batched replies over HTTP are not tested. The server test sends one request at a time. A near-tie that passes at batch sizes above 1 leaves no record.

**Read these yourself.**
- `tests/equivalence/test_engine_equivalence.py:39`: the scorer. Look at the feed order and the top-2 count.
- `tests/property/test_scheduler.py:95`: the limit checks of the wrapper runner.
- `docs/AGENT_ROADMAP.md:91`: rule 1 with the new text for batch sizes above 1.
