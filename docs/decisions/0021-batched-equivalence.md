# 0021: Batched equivalence scores every token under mlx-lm, and the scheduler's properties

**Status:** accepted (the author chose the recommended rule from the measurements below before the test was written) · **Task:** M2-T3 (Q2, Q3)

## Context

Decision 0012 left batch sizes above 1 without a near-tie threshold until one was measured. Decision 0019 measured that batched decode flips near-ties: at 128 tokens, `palindrome` at token 20 (margin 0.0) for B ≥ 2 and `kv-cache` at token 32 (margin 0.25) at B = 8. Since M2-T2 (decision 0020) the engine batches by default, but its equivalence test pinned `max_num_seqs=1`.

Q3 says: on a mismatch, log the top-2 margin; a near-tie under the threshold is recorded, anything else fails. Read literally, that compares only up to the first difference. After a flip, the reply continues from another prefix, and mlx-lm's own reply says nothing about it.

**Measured** on the author's M4, Qwen3-0.6B, all 10 golden prompts submitted at once to `EngineClient` with `max_num_seqs = B`, greedy, against mlx-lm on the same machine in the same session. Throwaway scripts in the session scratchpad, not committed. Each run reached exactly B rows. Two runs at 32 tokens gave the same tokens, so batched output is deterministic for a fixed arrival order.

First difference against `reference()` (mlx-lm's `generate_step`), with mlx-lm's top-2 margin there:

| Tokens | B=1 | B=2 | B=4 | B=8 |
|---|---|---|---|---|
| 32 | 10/10 | 7/10: `palindrome` 20 (0.0), `translate` 14 (0.125), `summarise` 28 (0.125) | 8/10: `palindrome` 20 (0.0), `summarise` 28 (0.125) | 8/10: as B=4 |
| 128 | 10/10 | 9/10: `palindrome` 20 (0.0) | 9/10: as B=2 | 8/10: `palindrome` 20 (0.0), `kv-cache` 32 (0.25) |

At every difference, uLLM's token was mlx-lm's second choice. The set of differences changes with `max_tokens` because requests finish at other steps, so the batch composition, and the shapes the forward runs at, change too.

**Teacher-forced**: mlx-lm fed each prompt, then uLLM's own tokens, the way `generate_step` feeds them (prefill all but the last prompt token, then one token per call). At each position, the gap is mlx-lm's best logprob minus the logprob of uLLM's token. On a prefix that matches mlx-lm, this is mlx-lm's own arithmetic, so the gap is 0.

| Tokens | B=1 | B=2 | B=4 | B=8 |
|---|---|---|---|---|
| 32, worst gap | 0.0 | 0.125 | 0.125 | 0.125 |
| 128, worst gap | 0.0 | 0.0 | 0.125 | 0.25 |

Each case had at most one position with a gap above 0: its first difference. Every later token was mlx-lm's own choice for uLLM's prefix. `palindrome`'s flip has gap 0: an exact tie in bf16.

**A mask that ignores padding** (`create_causal_mask` without `left_padding`), the mutant that decision 0019's verifier found only a batch-size-1 comparison catches:

| Tokens | B=2 | B=4 | B=8 |
|---|---|---|---|
| 32, worst gap | 0.125 | 0.125 | 0.125 |
| 128, worst gap | 2.0 | 0.875 | 0.125 |

At 32 tokens its gaps stay inside the real runs' range. What tells it apart: at B = 2, 4 and 8, `emoji` token 1 is mlx-lm's third choice, with two tokens above it. No real run picked a token below mlx-lm's second.

## Decision

1. **Batch size 1 stays exact**, through `reference()`, with the margin at the first difference (decision 0012).
2. **Batch sizes 2, 4 and 8 score every token.** mlx-lm is fed the prompt and then uLLM's own tokens. Each uLLM token must be in mlx-lm's top 2 (at most one token has a strictly higher logprob) and at most `BATCH_NEAR_TIE = 0.25` logprob behind mlx-lm's best. A failure names the case, the position, the gap and the count above.
3. **0.25 is the worst gap measured**, at B = 8 and 128 tokens: two bf16 steps at a logit between 16 and 32. It equals decision 0012's cross-machine allowance by measurement, not by choice. It applies only to batch sizes above 1.
4. **The test proves the batch size.** It submits all 10 prompts at once and asserts that the widest step had exactly `max_num_seqs` rows.
5. **A reply ends at EOS or at `max_tokens`**, checked for every case at every batch size.
6. **The scheduler property test (Q2)** drives the real `Engine` on `FakeRunner` with Hypothesis: `max_num_seqs` 1 to 8, `max_kv_slots` from `max_model_len` to 4 times it, and up to 24 loop turns of up to 4 submits or aborts each. A wrapping runner checks every step: held rows ≤ `max_num_seqs`, rows × the longest reservation ≤ `max_kv_slots`, rows × the longest written row ≤ `max_kv_slots`, and a decode names every held sequence. After the run: every request ends exactly once (`Finished` with `max_tokens` tokens unless aborted), each admission comes after every earlier arrival is admitted or ended, adds equal releases, the last `Load` is empty, and `check_trace` finds nothing.

## Alternatives considered

- **First difference only (Q3's literal reading).** Exact up to the first difference, a near-tie there, the rest of the case not compared. It leaves 12 of `palindrome`'s 32 tokens unchecked at every B ≥ 2.
- **Every token, 0.25, no rank condition.** The padding-mask mutant passes at 32 tokens, the length CI runs.
- **0.125 with the rank condition.** Green at 32 tokens, red at 128 tokens and B = 8 (`kv-cache`, 0.25), so the full-length run would fail on correct code.
- **Compare against mlx-lm's batched output.** mlx-lm's `BatchGenerator` prefills prompts together, uLLM one per step (DEC-7), so the shapes differ and the tokens would not match exactly either. The reference would also be a second engine to trust.
- **A Hypothesis stateful machine on `Scheduler` alone.** The engine is where requests end, release and abort, and leaks live there. Driving `Engine` tests the scheduler through the caller that owns it.

## Consequences

- Rule 1 now reads, for batch sizes above 1: every token within mlx-lm's top 2 and 0.25 logprob, on uLLM's own prefix. Batch size 1 is unchanged.
- The rank condition is what catches the padding-mask mutant at 32 tokens, on one token of one prompt. `tests/property/test_batch_kv.py` checks the mask values themselves.
- `make equivalence` runs the engine test four times: about 38 s for all four at 32 tokens on the M4 on battery; at 128 tokens, 100 s in one run and 56 s in another, both on battery.
- The server equivalence test still sends one request at a time. What batching changes is below the engine, and the engine test covers it.
- `check_trace` still does not check the slot budget: `hello` has no field for it (decision 0020). The property test checks it at the runner.
- M3's variants (blocks against contiguous, prefix cache, preemption) set their own thresholds the same way: measure first.
