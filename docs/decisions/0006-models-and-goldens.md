# 0006: Pinned models, golden fixtures, and what crossing machines does to them

**Status:** accepted · **Task:** M0-T7

## Models

`ullm.models` pins each model to a commit of its Hugging Face repository, so weights, tokenizer and chat template cannot change under the goldens, evals or benchmarks. `make models` (or `scripts/fetch_models.py`) downloads them into the Hugging Face cache.

| Key | Repository | Revision | Download | Role |
|---|---|---|---|---|
| `qwen3-0.6b-bf16` | `mlx-community/Qwen3-0.6B-bf16` | `42096995` | 1.21 GB | default: goldens, equivalence, CI, multi-replica runs |
| `qwen3-0.6b-8bit` | `mlx-community/Qwen3-0.6B-8bit` | `11de9687` | 0.65 GB | quantization evals |
| `qwen3-0.6b-4bit` | `mlx-community/Qwen3-0.6B-4bit` | `73e3e38d` | 0.35 GB | quantization evals |
| `qwen3-1.7b-bf16` | `mlx-community/Qwen3-1.7B-bf16` | `9cd66928` | 3.46 GB | headline benchmark tables |

Facts about the default model, read from the installed mlx-lm 0.32.2 and transformers 5.23.0:

- The mlx-lm tokenizer wrapper's `apply_chat_template` returns `list[int]`.
- `enable_thinking: false` makes the template emit an empty `<think>\n\n</think>\n\n` block after `<|im_start|>assistant\n`. Without the flag, the template leaves thinking on.
- The only EOS id is 151645 (`<|im_end|>`).
- The tokenizer has 151,669 entries; the model's vocabulary dimension is 151,936.

## Goldens

`scripts/make_goldens.py` runs mlx-lm's own greedy `generate_step` (the reference; never engine code, rule 11) over the ten prompts in `tests/equivalence/prompts.json`, with thinking off and up to 128 tokens. `tests/equivalence/goldens/qwen3-0.6b-bf16.json` stores, per prompt: the rendered prompt text, the prompt token IDs, the generated token IDs, the finish reason, the decoded text (for people only), and **the top-1 minus top-2 log-probability margin at every generated position**. The file also records the library versions, the model revision, and the chip and device that produced it.

The margins make every future divergence classifiable as a near-tie or a real bug without re-running anything.

`tests/equivalence/test_goldens.py` (marker `mlx`) checks on every `mlx` CI run that:

- the installed versions are the ones the goldens were made with;
- every prompt still renders to the same text and token IDs (machine-independent, so strict everywhere);
- mlx-lm reproduces the golden tokens for the first 32 positions (catalog F2's CI budget). `make goldens-check` compares all 128.

A missing model skips these tests locally, with a message to run `make models`. In CI, `ULLM_REQUIRE_MODEL=1` turns that skip into a failure, so the job can never go green by skipping.

## Findings

1. **Exact bf16 ties are common.** Of 495 generated positions, 6 have a top-2 margin of exactly 0, and 11 more are at 0.125 or less. Three of the ten prompts hit an exact tie within their first 32 tokens. The margins are quantized in steps of 0.0625 and 0.125: log-probabilities come from bf16 logits whose magnitude puts their resolution there. Risk R-2 is therefore concrete. Anything that changes the arithmetic, such as batching or a different kernel, can flip these picks.
2. **Same machine: exact.** On the author's M4, regenerating gives identical tokens for all 128 positions of every prompt, run after run. The M4's CPU path also matches the GPU goldens for the first 32 tokens of all ten prompts.
3. **Different machine: near-tie divergence.** On the hosted `macos-26` runner's paravirtualized GPU (CI run 43705255930), mlx-lm itself diverges from the M4 goldens in 5 of 10 prompts. Every divergence is at a position where the M4 saw a near-tie: margins 0, 0, 0.0625, 0.125 and 0.125. The runner's own margins at those positions were also near-ties: 0, 0.0625, 0.125, 0.125 and 0.125. None is at a confident position. The tolerant check passed with these as warnings in run 43705619599.

## Decisions that follow

- **Golden reproduction is exact on the golden machine and near-tie tolerant elsewhere.** When the current chip differs from the one in `made_on`, a divergence is accepted only where the golden margin is at most 0.25, about two bf16 steps at Qwen3's logit magnitude. It is reported as a test warning. A divergence at a confident position fails on every machine. The 0.25 is provisional and applies only to mlx-lm against its own goldens.
- **uLLM's equivalence (rule 1, Q3, from M1-T5) compares against mlx-lm run on the same machine in the same session, not against these files.** That holds in CI and for `make verify` on someone else's Mac, which would diverge from M4 goldens at near-ties just as the runner does. The committed goldens remain the record of the reference, the template and tokenization check, and the source of the margins. M1-T5 sets the documented near-tie threshold for rule 1, using these margins.
- Regenerate the goldens (`make goldens`) after any change to the model revisions, the prompts or the pinned libraries. The version check fails until that is done.
