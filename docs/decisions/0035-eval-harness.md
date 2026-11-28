# 0035: Eval harness: GSM8K exact match through `ullm serve`, WikiText-2 perplexity on the weights

**Status:** accepted · **Task:** M4-T1 (Q5)

## Context

Q5 compares bf16, 8-bit and 4-bit on perplexity over a fixed text, exact match on a 100 to 200 item task set with one right answer each, tool-call validity (Q6), tokens per second and memory. The datasets live in the repo under licenses that allow it. Decoding is greedy, with one template and thinking setting, and prompts are never tuned on the eval set. M4-T2 runs the three models and builds the table. M4-T3 gates CI on about 20 items. `ullm serve` returns no logprobs (`Choice.logprobs: None`), and it has no tool calling until A2 (M6-T1).

A throwaway probe on mlx-lm 0.32.2 (`generate`, greedy, thinking off, the prompt in 3) with `qwen3-0.6b-bf16` scored 15 of the first 30 GSM8K test items. Outputs averaged 274 tokens, and some reached the 512-token cap. It took 116 s, one item at a time.

## Decision

1. **Datasets** (the author's choice), byte-for-byte copies with a `README.md` each:
   - `evals/datasets/gsm8k/test-200.jsonl`: the first 200 lines of `grade_school_math/data/test.jsonl` at commit `3101c7d5` of openai/grade-school-math, MIT, with its `LICENSE`.
   - `evals/datasets/wikitext-2/wiki.test.raw`: from `wikitext-2-raw-v1.zip` (sha256 `ef7edb56…`), CC BY-SA 3.0 or GFDL. The notice covers that file alone.
   - `evals/data.py` records each copy's sha256 and refuses (`DataChanged`) a copy that differs.
2. **Exact match goes through `ullm serve`**, because it measures what a client gets. `evals.exact_match.run` starts `ullm serve --model KEY --max-num-seqs 1 --telemetry off` on a free loopback port, sends the 200 items one at a time, non-streaming, and stops the server. Any failed request stops the run, because a score over fewer items would mislead.
3. **One prompt:** the question, then `\nPlease reason step by step, and put your final answer within \boxed{}.`, the usual Qwen math prompt. Greedy (`temperature` 0), `chat_template_kwargs` `{"enable_thinking": false}` (F4), `max_tokens` 512. No system prompt and no few-shot examples.
4. **Scoring** (`evals.score`): the gold answer is the one number after `####`. The model's answer is the last number in the last `\boxed{...}`, with braces matched. When that last box is not closed, the reply has no answer, even if an earlier box is closed. Commas inside a number are dropped. Answers compare as `Decimal`, so `18.00` matches `18`. No box, or a box with no number, is no answer and counts as wrong. A reply cut off at `max_tokens` counts as it stands.
5. **One request at a time.** Batched greedy output can flip at bf16 near-ties (decision 0021), so a batched run's score could change with arrival timing. With `--max-num-seqs 1`, a rerun on the same machine sends the same token sequence through the same arithmetic. In the M4-T1 session, two reruns of the first 20 items gave the committed run's replies item for item.
6. **Perplexity runs on the weights, not through the server**, since the server has no logprobs. `evals.perplexity.load` loads the pinned snapshot with mlx-lm's `load` and refuses to download one. `measure` tokenizes the whole text, cuts it into non-overlapping 1,024-token windows, drops the short tail, and runs batches of 8 windows. Each window scores its last 1,023 tokens, and the first token is context only. It reports the mean loss in nats per token, `exp` of it, and the delta-method standard error that mlx-lm's `eval_ppl` uses. A test holds `measure` equal to `eval_ppl` on the real model.
7. **Cost figures:** `output_tok_s` is the sum of `completion_tokens` over the wall time from the first send to the last reply. It includes prefill and the HTTP round trip. `server_peak_footprint_bytes` is `phys_footprint_peak` summed over the server's process group (`bench.conditions.footprint`), read after the last reply. A run records `env.json` but does not apply the B5 checks, so these figures are for the trade-off table's context, not benchmark claims. M4-T2 decides the conditions for the table.
8. **Results:** `evals.run --model KEY --out DIR` (`make evals EVAL_MODEL=… EVAL_OUT=…`) writes `result.json` (`EvalResult`), `items.jsonl` (one `ItemResult` per item, with the full reply), `env.json` and `server.log` into a new directory. It refuses a directory that already holds files. `env.json` is taken before any output exists, so `dirty` describes the code that ran. `ExactMatch.of` derives every count from the items.
9. **Tool-call validity** (Q6) is M6-T2's (the author's choice). `EvalResult.tool_calls` is `None` until then.

## Alternatives considered

- **ARC-Challenge or MMLU** (one letter, cheap in CI). The author chose GSM8K: a 0.6B model sits nearer chance on letter tasks, so a quantization drop is harder to see.
- **A public-domain Gutenberg text for perplexity.** It has no attribution terms, but no published figure compares with it.
- **Batched exact match** (faster, about 8 times at `--max-num-seqs 8`). Lost on 5.
- **Exact match on mlx-lm directly.** It would measure the weights twice and never the engine, so an engine regression could not move the score M4-T3 gates on.
- **mlx-lm's `evaluate` (lm-eval).** It adds the `lm_eval` dependency and runs mlx-lm's own `batch_generate`, not uLLM.
- **Extending `bench.loadgen.Reply` to keep text.** The bench streams to time every chunk. The eval needs only the final text, and a non-streaming request gives it with less code.

## Consequences

- The first run, `evals/results/m4/t1-qwen3-0.6b-bf16/` (claims row 43), scored 129 of 200 with 6 items unanswered and 7 cut off at the cap, and a perplexity of 23.12 ± 0.14 over 298,716 tokens. It ran on battery power. Its 49.7 output tok/s is lower than the 71 tok/s of the mlx-lm probe in the same session. M4-T2 measures under its own conditions before it reports a speed.

- M4-T2 runs `make evals` for the three keys and builds its table from the `result.json` files.
- M4-T3 can take a fixed subset of `data.gsm8k()`. At a mean near 274 output tokens, 20 items one at a time is about 5,500 tokens on the hosted runner.
- A change to the prompt, the cap, the scoring or a dataset changes every score. It needs a new decision and new runs of all three models.
- The standard error covers perplexity only. Exact match on 200 items has a binomial standard error near 3.5 points at a score of 0.5, which M4-T3's noise margin must respect.
