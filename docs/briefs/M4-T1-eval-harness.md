# M4-T1: Eval harness and datasets

**State:** in build. You read this page to follow the build.
**Decision record:** [0035](../decisions/0035-eval-harness.md) (the full design, for agents) · **Spec:** roadmap M4-T1 (Q5)

## The goal

You run one command for one pinned model and get its quality and cost in one results directory. M4-T2 runs it for bf16, 8-bit and 4-bit and builds the trade-off table.

## The problem

- No check measures output quality. The equivalence tests compare uLLM with mlx-lm on the same weights only (rule 1).
- Quantization changes the weights, so equivalence cannot say how much quality a smaller model loses.
- `ullm serve` returns no logprobs. Perplexity cannot go through the HTTP API.
- Q5 wants the datasets in the repo, under licenses that allow it, with attribution.
- Q5 wants greedy decoding, one template and thinking setting, and no prompt tuning on the eval set.

## How it works

```
evals.run --model KEY
   │
   ├─ perplexity: mlx_lm.load ── forward pass over WikiText-2 test, in 1,024-token windows
   │
   └─ exact match: ullm serve --model KEY --max-num-seqs 1
          ▲   200 GSM8K questions, one at a time, greedy, thinking off
          └─ footprint: the peak memory of the server process group
   │
   ▼
evals/results/m4/<run>/  result.json  items.jsonl  env.json
```

1. `evals/datasets/`: the first 200 GSM8K test questions and the WikiText-2 raw test text. Each file is a copy of its source, byte for byte. Each folder has a `README.md` with the source, the commit or archive hash, and the license.
2. `evals/score.py`: reads the gold answer after `####`. It reads the model answer as the last number in the last `\boxed{}`. Two answers match when they are equal numbers.
3. `evals/perplexity.py`: cuts the tokens into windows and runs the model on each window. It gives the mean loss, the perplexity and its standard error.
4. `evals/run.py`: runs the two parts and writes the results directory.
5. `evals/schema.py`: the shape of `result.json` and of each line of `items.jsonl`.

The agent measured one fact before the design. On mlx-lm with the bf16 model, the prompt below gave 15 correct answers out of 30. The mean output was 274 tokens, and some outputs reached 512 tokens.

## What you get at the end

Run `make evals MODEL=qwen3-0.6b-bf16`. The command writes a results directory under `evals/results/m4/`. The agent commits one bf16 run as the proof of the harness.

New tests:

- The scorer reads the gold answer of all 200 items. It reads model answers in the forms that the probe showed.
- Each dataset file has the hash that its `README.md` names.
- The exact-match part runs against `ullm serve --fake` and writes one line for each item.
- On the real model, `evals.perplexity` gives the same perplexity as mlx-lm's own `eval_ppl` on the same windows.

What you do not get yet:

- The table for the three models: M4-T2.
- The CI gate on about 20 items: M4-T3.
- Tool-call validity: M6-T2, after tool calling (A2) exists. `result.json` keeps an empty field for it.

## Where this puts the project

```
M4  T1 ▓  T2 ░  T3 ░
```

After this slice, M4-T2 can run in Lane B.

## Choices the agent made

You chose GSM8K, WikiText-2 and the move of tool calls to M6-T2. You can still change the choices below before `/ship`:

- The prompt is the question, then "Please reason step by step, and put your final answer within \boxed{}." This is the usual prompt for Qwen math tasks. The agent did not tune it on the eval set.
- The harness sends one request at a time, with `--max-num-seqs 1`. Batches change the bf16 arithmetic, so a near-tie can flip one token (decision 0021). One at a time, a second run gives the same answers.
- `max_tokens` is 512. An answer that the cap cuts off counts as wrong.
- The tokens per second include prefill and the HTTP round trip. These figures come from one run without the B5 checks, so they are not a benchmark claim.

## What can go wrong

- The model writes no `\boxed{}`. The item counts as wrong, and `result.json` counts the items with no answer.
- The model puts units or words in the box, such as `\boxed{18 dollars}`. The scorer takes the last number, so the answer is 18.
- The text in the box has no number. The item counts as wrong.
- A model download is missing. The run stops before it starts the server. It names the command that downloads the model.

## Words

| Word | Meaning |
|---|---|
| perplexity | The exponential of the mean loss for each token on a fixed text. Lower is better. |
| exact match | The share of items where the model answer equals the gold answer. |
| `phys_footprint` | The memory that macOS charges to a process, compressed and swapped pages included. |

## Learn from this slice

A test asks whether the code is correct, and it gives the same result each run. An eval asks whether the output is good enough, and it gives a score that you compare with a threshold. Both need fixed inputs, but an eval also needs a fixed model, prompt and decoding.

Perplexity measures the weights alone, so it needs no server. Exact match measures what a user gets from `ullm serve`, so it goes through the server.

**Test yourself:** why does the harness send one request at a time, when batches finish the 200 items faster?

<details><summary>Answer</summary>

A batch changes the arithmetic, and a near-tie can then pick a different token. The answers can then change from run to run. One request at a time gives the same answers on each run. A change in the score then comes from the model.

</details>

## Result
