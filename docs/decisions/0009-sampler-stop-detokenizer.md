# 0009: Sampler, stop check and streaming detokenizer

**Status:** accepted · **Task:** M1-T2

## Context

E2 asks for a per-row sampler with greedy, temperature, top-k and top-p, seeded per request. E3 asks for a stop check (end tokens, `max_tokens`, stop strings held back and never emitted) and one streaming detokenizer per request that holds back incomplete characters. Constraints:

- Rule 1: greedy output stays equal to mlx-lm's. Decision 0008 fixed greedy as the argmax of log-probabilities in the model's dtype.
- Rules 2 and 3: the stop check decides when a request finishes, so it lives in `ullm.core`, which is MLX-free; the detokenizer reaches it through the `Tokenizer` protocol (decision 0008: "the streaming detokenizer joins this protocol in M1-T2").
- Rule 11 allows mlx-lm's detokenizer utilities but not its sampler loop; `sample_utils` is not on the allowed list.
- Decision 0008 asked that each token's random key come from `(seed, position)`, so preemption (E11) needs no sampler state in the runner beyond the seed.

## Decision

**Usage.** The engine loop (M1-T3) does, per request:

```python
runner.add(seq, SamplingParams(temperature=0.7, top_p=0.9, top_k=40, seed=7))
check = StopCheck(
    tokenizer.detokenizer(), runner.info().eos_token_ids, max_tokens=256, stop=("\n\nUser:",)
)
token = runner.execute(PrefillPlan(seq=seq, token_ids=prompt, start=0)).tokens[0]
chunk = check.add(token)  # Chunk(text, finish_reason); push chunk.text to the client
while chunk.finish_reason is None:
    token = runner.execute(DecodePlan(seqs=(seq,), token_ids=(token,))).tokens[0]
    chunk = check.add(token)
runner.release(seq)
```

**`SamplingParams`** refuses, when built, a temperature that is neither 0 nor a normal float32 (about `1.18e-38` to `3.4e38`), a `top_p` outside `(0, 1]`, a `top_k` that is not an int `>= 0`, a seed that is not an int or `None`, and bools anywhere. The temperature range is a limit on input: the sampler divides float32 log-probabilities by the temperature, and beyond that range float32 rounds it to 0 or infinity. It stores `temperature` and `top_p` as floats, since an int past int64 cannot reach MLX. `seed=None` promises no reproducibility: `MLXRunner` draws a 64-bit seed with `secrets` at `add`; `FakeRunner` keeps decision 0008's `seed or 0`.

**Sampler** (`runner/sampler.py`): `sample(logits [B, vocab], rows) -> [B]`, lazy, one `SampleRow(params, seed, position)` per row.

- Temperature 0: `argmax(logits - logsumexp(logits))` in the model's dtype, as before.
- Otherwise, in float32: log-probabilities; a keep-mask from top-p, then top-k, both on the **untempered** distribution, as mlx-lm's `make_sampler` orders them (0.32.2, `sample_utils.py`); then `categorical(where(keep, (logprobs - max) / temperature, -inf))`. Subtracting the max first leaves the most likely token at 0, so even the smallest allowed temperature cannot scale every token to `-inf`. Review found both extremes of the temperature producing NaN draws before the range above was set.
- Top-p keeps the most likely tokens until their mass reaches `top_p`, and at least one (review found a `top_p` that rounds to 0 in float32 kept every token). Top-k keeps `top_k`; `top_k` at or past the vocabulary is off. Both keep tokens tied with the last kept one.
- Each sampled row draws with `mx.random.key(blake2b(f"{seed}:{position}"))`, where `position` is the index the token takes in its sequence: `start + len(token_ids)` for a prefill, held tokens plus one for a decode. A row's token depends only on its logits, settings, seed and position, never on the batch.

**Detokenizer** (`runner/detokenizer.py`, MLX-free): `ByteDetokenizer(token_bytes)` feeds each token's bytes to `codecs`' incremental UTF-8 decoder (`errors="replace"`), which holds back exactly an unfinished character. `flush()` ends the stream; an unfinished character becomes U+FFFD, as `decode` gives. `MLXTokenizer` builds the table once at load: GPT-2's byte-level alphabet for ordinary tokens, the UTF-8 of the content for added tokens, `b""` for ids past the tokenizer (the logits are 151,936 wide, the tokenizer 151,669; `decode` gives `""` for them). It refuses a tokenizer whose decoder is not `ByteLevel`; every model in `ullm.models` is Qwen3, which is.

**Stop check** (`core/stop.py`): `StopCheck(detokenizer, eos_token_ids, max_tokens, stop).add(token) -> Chunk(text, finish_reason)`.

- An end token finishes with `"stop"`; its text is never decoded. Reaching `max_tokens` flushes the detokenizer and finishes with `"length"`. An end token as the last allowed token is `"stop"`.
- After each token, the pending text is searched for every stop string. When stop strings overlap, the occurrence that **ends first** wins (the longer one on a tie at the same end). That is the occurrence a stream completes first, so the result does not depend on where tokens split the text. A stop string found in text released at the finish still wins over `"length"`.
- Otherwise the longest end of the pending text that starts some stop string is held back; the rest is emitted. `add` after the finish raises.
- `FinishReason` here is `"stop" | "length"`. The server's own `FinishReason` (`server/schemas.py`) will gain `"tool_calls"` with A2, which the server, not the engine, decides; the two are kept apart on purpose.

## Alternatives considered

- **mlx-lm's `BPEStreamingDetokenizer`.** Allowed by rule 11 and byte-based, but it trims a leading space from the reply: measured on Qwen3, `' leading space'` streams as `'leading space'`, so its text is not `decode(tokens)`.
- **Decoding a window of recent tokens** (vLLM's and TGI's prefix-offset method: decode from the last complete point, hold while the text ends in U+FFFD). Tokenizer-independent, and built first. Review found it holds a real U+FFFD token (Qwen3 id 5691) until the end, decoding a window that grows by one token each step, so the cost grows with the square of the run (measured in review, not committed). Bounding the hold to three tokens fixed that, but Qwen3 also has tokens that end one character and start the next (ids 44680, 57160, 72496; 72496 is bytes `92 E1 9E`). A loop of one leaves no clean token boundary to restart the window from. Only the bytes can tell an unfinished character from a finished one.
- **Temperature before the masks** (vLLM, Hugging Face). The common order, but then the same settings would sample a different distribution here than in mlx-lm, the baseline uLLM is measured against (rule 7).
- **Stop at the earliest-starting occurrence.** Depends on the split: with stops `abcd` and `bc`, text arriving as `abcd` in one token would cut at 0, but `a|b|c|d` completes `bc` first.
- **A stateful random generator per sequence.** Needs state the runner must keep, or rebuild after preemption; a key from `(seed, position)` needs nothing.

## Consequences

- A sequence rebuilt after preemption draws with the same keys, but its tokens match the first run only while its logits do. A prefill computes bf16 logits that differ slightly from decode's (decision 0008), and in the proof's throwaway run (not committed) a rebuilt reply diverged partway through, greedy included. With `seed=None`, `release` drops the drawn seed, so a rebuilt unseeded sequence draws new keys; the core (M3) can resolve the seed before `add` if that matters.
- The sampler sorts the vocabulary for top-p and runs `topk` for top-k, per row, every step. M1-T6 measures the cost; greedy is unchanged.
- The token-bytes table is built once, at load. Detokenizing is then one table lookup and one incremental decode per token, with no window that grows. M1-T6 can measure both.
- Engine code now imports `tokenizers` (the decoder type check), so `pyproject.toml` pins it at the version mlx-lm already brought in.
- A model with a non-`ByteLevel` tokenizer (SentencePiece, WordPiece) needs a new detokenizer before it can be added to `ullm.models`; `MLXTokenizer` refuses it loudly.
- Which OpenAI behaviour the "ends first" rule matches is untested; the contract (C1) does not specify overlapping stop strings.
