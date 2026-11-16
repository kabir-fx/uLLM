# 0008: Runner protocol v1, MLXRunner and FakeRunner

**Status:** accepted · **Task:** M1-T1 `[design]`

## Context

C3 (decision 0003) sketched the boundary: the core owns every decision, the runner owns every tensor and samples. M1-T1 fixes the exact signatures, the invariants a caller must keep, the error modes, and the two adapters behind the seam. Constraints:

- Rule 1: greedy token IDs equal mlx-lm's on the same machine.
- Rules 2 and 3: the runner is called only from the engine thread; `ullm.core` imports `runner.base` and must stay MLX-free.
- Rule 11: no `generate_step` and no mlx-lm caches in engine code; our own per-layer cache plugs into `model(inputs, cache=[...])` (roadmap section 4).
- M1 runs one sequence at a time (`max_num_seqs = 1`, M1-T3). Batched KV is E7 (M2-T1). The sampler beyond greedy is E2 (M1-T2).

**Measured fork.** mlx-lm's `generate_step` (0.32.2, `generate.py`) prefills `prompt[:-1]`, evaluates the cache, then runs the last prompt token alone and samples from that. A throwaway run on the author's M4 compared greedy output against `reference()` from `scripts/make_goldens.py` over the 10 golden prompts, 128 tokens each:

| Our prefill | Matches mlx-lm |
|---|---|
| `[1, L-1]`, then `[1, 1]` | 10 of 10 |
| One `[1, L]` forward | 8 of 10 (uncommitted throwaway run; reproduce by making `_forwards` in `mlx_runner.py` return one chunk and running the equivalence test): `palindrome` differs at token 20 (reference top-2 margin 0.0), `translate` at token 14 (margin 0.125) |

The cache in both runs concatenated on append; mlx-lm's preallocates in steps of 256. Layout did not change a single token.

## Decision

**Usage.** The core (M1-T3) and the runner equivalence test call it like this:

```python
runner, tokenizer = load_mlx(DEFAULT_MODEL)  # engine startup, before threads start
prompt = tokenizer.render(messages, {"enable_thinking": False})

runner.add(seq, SamplingParams(temperature=0.0, top_p=1.0, top_k=0, seed=None))
step = runner.execute(PrefillPlan(seq=seq, token_ids=prompt, start=0))
token = step.tokens[0]
while token not in runner.info().eos_token_ids and n < max_tokens:
    step = runner.execute(DecodePlan(seqs=(seq,), token_ids=(token,)))
    token = step.tokens[0]
runner.release(seq)
```

**Shape.** `runner/base.py` keeps the v0 dataclasses (`SamplingParams`, `RunnerInfo`, `PrefillPlan`, `DecodePlan`, `StepPlan`, `StepResult`) and the four-method `Runner` protocol unchanged in signature. It adds an MLX-free tokenizer protocol:

```python
class TemplateMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class Tokenizer(Protocol):
    def render(
        self, messages: Sequence[TemplateMessage], template_kwargs: Mapping[str, object]
    ) -> tuple[int, ...]: ...  # chat template, then token IDs; one call, as the goldens do
```

The streaming detokenizer joins this protocol in M1-T2. The server maps `ChatMessage` to `TemplateMessage` with the `template_role` property. `MLXTokenizer.render` refuses every `apply_chat_template` parameter except the template inputs `tools` and `documents` (transformers 5.23: `conversation`, `tokenize`, `return_tensors`, `return_dict`, `return_assistant_tokens_mask`, `add_generation_prompt`, `continue_final_message`, `chat_template`, `truncation`, `max_length`, `padding`, `tokenizer_kwargs`, `sanitize_control_tokens`), and refuses a batch of conversations; which other `chat_template_kwargs` a client may send is the server's decision (M1-T4). Special-token text inside message content renders as control tokens, as in mlx-lm; whether to escape it is also M1-T4's decision.

**Invariants the protocol states** (in `base.py` docstrings, checked by every adapter, and covered by one conformance test that runs on both):

1. Lifecycle per `SeqId`: `add` once, then `execute` any number of times, then `release` once. `add` of a live id, or `execute` or `release` of an unknown id, raises `ValueError`.
2. Positions are contiguous. A `PrefillPlan` must have `start` equal to the tokens the runner holds for that sequence. A `DecodePlan` row appends one token at the next position. `start + len(token_ids)` must not exceed `max_model_len`, and every token ID must be in `[0, vocab_size)`. Sequence ids, positions and token IDs must be `int` (not `bool`), `token_ids` and `seqs` tuples, and `sample` a `bool`; plans refuse other types when built, and `add` and `release` refuse a non-`int` id. A violation raises `ValueError` before any forward pass and changes nothing.
3. `StepResult.tokens` has one entry per sampled row, in plan order: one for a `PrefillPlan` with `sample=True`, none with `sample=False`, `len(seqs)` for a `DecodePlan`.
4. An `execute` that fails after validation, in the forward pass or the eval, marks every sequence in the plan as failed. The only legal call for a failed sequence is `release`.
5. Not thread-safe by contract: only the engine thread calls it (rule 2). No locks.

**`MLXRunner`** (`runner/mlx_runner.py`):

- `load_mlx(spec) -> tuple[MLXRunner, MLXTokenizer]` calls `mlx_lm.load(spec.repo, revision=spec.revision)` once. `RunnerInfo` comes from the loaded model and tokenizer: `vocab_size` is the logits width (151,936, not the tokenizer's 151,669), `max_model_len` from the config, EOS from `tokenizer.eos_token_ids`.
- Per sequence: one `ContiguousKV` per layer (`runner/kv/contiguous.py`), with `offset`, `update_and_fetch` (concatenate on append) and `make_mask`, which mirrors mlx-lm's (`None` for one token, `"causal"` otherwise).
- A `PrefillPlan` with `sample=True` and more than one token runs `token_ids[:-1]`, then `token_ids[-1:]`, and samples from the second. This split is hidden inside one step, because the measurement above shows it is needed for rule 1. With `sample=False` it is one forward pass.
- Sampling is greedy in M1-T1: `argmax` of the last position's log-probabilities, `logits - logsumexp(logits)`, as mlx-lm's `greedy_sampler` does. In bf16 that subtraction can turn two close logits into a tie that `argmax` breaks by index, so `argmax` of the raw logits could pick differently. `add` with any temperature other than 0, NaN included, raises `ValueError` until M1-T2 adds the sampler in `runner/sampler.py`.
- `DecodePlan` with more than one sequence raises `ValueError` naming M2-T1. Batched decode needs one batch cache (E7), not a loop over per-sequence caches.
- One `mx.eval` per step, on the sampled tokens, or on every layer's K and V when nothing is sampled; then the timings (rule 4). `forward_ms` covers forward through eval; `total_ms` adds input building and `tolist()`.

**`FakeRunner`** (`runner/fake_runner.py`), MLX-free:

- `FakeRunner(info: RunnerInfo, latency: Latency, sleep: Callable[[float], None] = time.sleep)`. `Latency` has `prefill_base_ms`, `prefill_per_token_ms`, `decode_base_ms`, `decode_per_row_ms`.
- Token at position `p` of sequence `s` is `blake2b(seed or 0, s, p)` taken modulo the number of non-EOS ids, then mapped onto those ids in order, so it is never EOS, every other id is reachable, and output is deterministic and independent of batching.
- Sleeps for the modelled time and reports it as `forward_ms` and `total_ms`, so `ullm sim` telemetry shows the latency model, not the host's noise.
- Enforces invariants 1–4 exactly as `MLXRunner` does, through the same `SeqLedger`. It accepts `DecodePlan`s of any size. E1's failure injection is the `sleep` argument: a `sleep` that raises fails the step mid-flight. It refuses a `RunnerInfo` whose vocabulary is all EOS.

**Tests** (written first, in `/build`):

- `tests/unit/test_runner_conformance.py`: invariants 1–4, parametrized over `FakeRunner` always and `MLXRunner` under the `mlx` mark.
- `tests/unit/test_fake_runner.py`: determinism, batch independence, the latency model with an injected `sleep`.
- `tests/equivalence/test_runner_equivalence.py`: the loop above, greedy, every golden prompt, against `reference()` on the same machine in the same session; exact match, with no near-tie allowance. `test_goldens.py` allows near-ties only across machines, and here the reference runs on the same machine; the 0.25 allowance would also pass the rejected single `[1, L]` prefill (margins 0.0 and 0.125). Also: `MLXTokenizer.render` equals each golden's `prompt_ids` (E1's chat-template golden test).

## Alternatives considered

- **One `[1, L]` prefill.** DEC-7's literal shape, but it misses rule 1 on 2 of 10 golden prompts (measured above).
- **Core plans the split** (`PrefillPlan(sample=False)` for `L-1` tokens, then the last prompt token as the first decode). It shows the split to the core, and every prefill costs two engine steps and two telemetry `step` events. The split exists only to match one library's numerics, so the runner hides it.
- **Implicit lifecycle** (no `add`; a `PrefillPlan` at `start=0` creates the sequence and carries `SamplingParams`). Three methods instead of four, but a stale or duplicate `SeqId` from a core bug would silently create a sequence instead of raising.
- **Opaque sequence handles** returned by the runner (`handle = runner.open(...)`; `runner.decode([handles])`). The core would hold runner-owned objects, plans would stop being plain data that telemetry and tests can record, and `SeqId` bookkeeping moves into two places.
- **Sequential decode loop for `B > 1` in `MLXRunner`.** Correct, but it is a slow path that M2 benchmarks could pick up without anyone noticing. An error is loud, and E7 replaces it.
- **mlx-lm's `KVCache`.** Allowed by rule 11's letter, but E1 and DEC-5 ask for our own cache objects. A concatenating cache is 20 lines and measured to give the same tokens.

## Consequences

- TTFT includes one extra one-token forward pass per prefill, roughly one decode step.
- `ContiguousKV` copies the whole sequence's K and V on every append, so decode cost grows with context. This is a readable placeholder for one sequence; M1-T6 and M1-T7 measure it against mlx-lm, and E7 replaces it in M2-T1.
- One synchronous `mx.eval` per step (rule 4), where mlx-lm overlaps the next step's graph construction with `async_eval`. uLLM's sequential mode will be slower by that overlap; M1-T7 measures and explains the gap.
- Prompts longer than 2,049 tokens are not covered by equivalence: mlx-lm also chunks `prompt[:-1]` into 2,048-token forwards. Chunked prefill (E13) can reproduce that from the core with `sample=False` chunks.
- Where a prefill is split changes bf16 logits by up to about 0.47 and so flips near-ties, the first token included (measured in review on a 36-token prompt: chunks `[35, 1]` match mlx-lm; `[5, 31]` and `[18, 18]` do not). `ContiguousKV` is bit-identical to mlx-lm's `KVCache` under the same chunks. So rule 1 holds only for the chunk boundaries mlx-lm uses; prefix caching (E12) and chunked prefill (E13) must state their equivalence target with this in mind.
- Preemption (E11, M3) keeps a sequence's sampler state while its KV is freed. M1-T2 should derive each token's random key from `(seed, position)`, so `release` and a later re-`add` reproduce it without a new method.
- `load_mlx` imports mlx-lm, which imports MLX. It is called once at startup; the returned tokenizer is plain Hugging Face code that the API thread may call (rule 2 is about MLX arrays).
