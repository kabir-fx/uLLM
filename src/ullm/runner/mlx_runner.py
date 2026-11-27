"""The runner for the real model: mlx-lm's model modules, our own loop and KV (E1, decision 0008).

Only the engine thread calls ``MLXRunner`` (rule 2). ``MLXTokenizer`` holds no
MLX arrays, so other threads may call it.
"""

import secrets
import time
from collections.abc import Mapping, Sequence
from typing import Any, cast

import mlx.core as mx
from mlx_lm import load
from tokenizers import decoders

from ullm.models import ModelSpec
from ullm.runner.base import (
    BlockId,
    DecodePlan,
    PrefillPlan,
    RunnerInfo,
    SamplingParams,
    SeqId,
    StepPlan,
    StepResult,
    TemplateMessage,
)
from ullm.runner.detokenizer import ByteDetokenizer
from ullm.runner.kv.batch import BatchKV
from ullm.runner.ledger import SeqLedger
from ullm.runner.sampler import SampleRow, sample

# MLX keeps freed buffers for reuse up to its memory limit by default. The concatenating
# cache before BatchKV freed a smaller K and V every step and never reused them, so the
# default held 9 GB after 12 requests and swapped (decision 0015). Each repack still frees
# buffers of a size that does not recur (decision 0019).
CACHE_LIMIT = 256 * 2**20


def load_mlx(spec: ModelSpec) -> tuple["MLXRunner", "MLXTokenizer"]:
    """Load weights, tokenizer and chat template once, at engine startup."""
    mx.set_cache_limit(CACHE_LIMIT)
    loaded = load(spec.repo, revision=spec.revision)
    model, wrapped = loaded[0], loaded[1]
    info = RunnerInfo(
        model_id=spec.key,
        vocab_size=model.args.vocab_size,  # the logits width, larger than the tokenizer
        eos_token_ids=frozenset(wrapped.eos_token_ids),
        max_model_len=model.args.max_position_embeddings,
    )
    return MLXRunner(model, info), MLXTokenizer(wrapped, info.vocab_size)


# Every apply_chat_template parameter except the template inputs tools and documents, as of
# transformers 5.23: each replaces the messages or the template, changes control-token
# escaping (M1-T4 decides), or changes the type or length of what render returns.
FIXED_TEMPLATE_OPTIONS = frozenset(
    {
        "add_generation_prompt",
        "chat_template",
        "continue_final_message",
        "conversation",
        "max_length",
        "padding",
        "return_assistant_tokens_mask",
        "return_dict",
        "return_tensors",
        "sanitize_control_tokens",
        "tokenize",
        "tokenizer_kwargs",
        "truncation",
    }
)


class MLXTokenizer:
    def __init__(self, wrapped: Any, vocab_size: int) -> None:
        self.wrapped = wrapped  # mlx-lm's TokenizerWrapper
        self._token_bytes = _token_bytes(wrapped, vocab_size)

    def render(
        self, messages: Sequence[TemplateMessage], template_kwargs: Mapping[str, object]
    ) -> tuple[int, ...]:
        fixed = FIXED_TEMPLATE_OPTIONS & template_kwargs.keys()
        if fixed:
            raise ValueError(f"render fixes these chat template options: {sorted(fixed)}")
        # Control-token text inside a message is encoded as text, so a message cannot close
        # its own turn or open another (decision 0011).
        ids = self.wrapped.apply_chat_template(
            list(messages),
            add_generation_prompt=True,
            sanitize_control_tokens=True,
            **template_kwargs,
        )
        if not all(type(t) is int for t in ids):
            raise ValueError("render takes one conversation, not a batch")
        return tuple(ids)

    def detokenizer(self) -> ByteDetokenizer:
        return ByteDetokenizer(self._token_bytes)

    def token_bytes(self) -> tuple[bytes, ...]:
        return self._token_bytes


def _token_bytes(wrapped: Any, vocab_size: int) -> tuple[bytes, ...]:
    """Each token id's bytes, for every id the model can sample (decision 0009)."""
    if not isinstance(wrapped.backend_tokenizer.decoder, decoders.ByteLevel):
        raise ValueError("the streaming detokenizer needs a ByteLevel tokenizer")
    alphabet = _byte_level_alphabet()
    added = {i: t.content.encode() for i, t in wrapped.added_tokens_decoder.items()}
    spelled = wrapped.convert_ids_to_tokens(list(range(len(wrapped))))
    # Ids past the tokenizer pad the logits to the model's width; they decode to nothing.
    spelled += [None] * (vocab_size - len(spelled))
    return tuple(
        added[i] if i in added else b"" if s is None else bytes(alphabet[c] for c in s)
        for i, s in enumerate(spelled)
    )


def _byte_level_alphabet() -> dict[str, int]:
    """GPT-2's byte-level alphabet: the printable character a ByteLevel vocabulary spells
    each byte with, mapped back to the byte."""
    printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1)]
    printable += range(ord("®"), ord("ÿ") + 1)
    unprintable = [b for b in range(256) if b not in printable]
    return {chr(b): b for b in printable} | {chr(256 + n): b for n, b in enumerate(unprintable)}


class MLXRunner:
    def __init__(self, model: Any, info: RunnerInfo) -> None:
        self.model = model  # an mlx-lm model module: model(inputs, cache=[one per layer])
        self._info = info
        self._ledger = SeqLedger(info)
        # Each sequence's KV is in exactly one place: its own BatchKV until its first
        # decode, then a row of _batch (decision 0019). A failed one may have none.
        self._prefilling: dict[SeqId, BatchKV] = {}
        self._batch: BatchKV | None = None
        self._sampling: dict[SeqId, tuple[SamplingParams, int]] = {}

    def info(self) -> RunnerInfo:
        return self._info

    def add(self, seq: SeqId, sampling: SamplingParams) -> None:
        self._ledger.add(seq)
        self._prefilling[seq] = BatchKV.empty(seq, len(self.model.layers))
        seed = secrets.randbits(64) if sampling.seed is None else sampling.seed
        self._sampling[seq] = (sampling, seed)

    def release(self, seq: SeqId, freed: tuple[BlockId, ...]) -> None:
        self._ledger.release(seq, freed)
        del self._sampling[seq]
        if self._prefilling.pop(seq, None) is None and self._batch and seq in self._batch.seqs:
            rest = [(self._batch, s) for s in self._batch.seqs if s != seq]
            self._batch = BatchKV.pack(rest) if rest else None

    def execute(self, plan: StepPlan) -> StepResult:
        with self._ledger.step(plan):
            began = time.perf_counter()
            match plan:
                case PrefillPlan():
                    tokens, forward_began, forward_done = self._prefill(plan)
                case DecodePlan():
                    try:
                        tokens, forward_began, forward_done = self._decode(plan)
                    except BaseException:
                        # The ledger fails every sequence in the plan, which is every row.
                        self._batch = None
                        raise
        done = time.perf_counter()
        return StepResult(
            tokens=tokens,
            forward_ms=(forward_done - forward_began) * 1000,
            total_ms=(done - began) * 1000,
        )

    def _prefill(self, plan: PrefillPlan) -> tuple[tuple[int, ...], float, float]:
        """A sampled prefill runs all but its last token, then the last token alone, as
        mlx-lm's ``generate_step`` does: one ``[1, L]`` forward picks different greedy
        tokens at near-ties (decision 0008)."""
        ids = plan.token_ids
        chunks = [ids[:-1], ids[-1:]] if plan.sample and len(ids) > 1 else [ids]
        inputs = [mx.array(chunk)[None] for chunk in chunks]
        cache = self._prefilling[plan.seq].layers
        forward_began = time.perf_counter()
        for x in inputs:
            logits = self.model(x, cache=cache)
        if not plan.sample:
            mx.eval([(kv.keys, kv.values) for kv in cache])
            return (), forward_began, time.perf_counter()
        params, seed = self._sampling[plan.seq]
        sampled = sample(logits[:, -1, :], [SampleRow(params, seed, plan.start + len(ids))])
        mx.eval(sampled)
        forward_done = time.perf_counter()
        return tuple(cast(list[int], sampled.tolist())), forward_began, forward_done

    def _decode(self, plan: DecodePlan) -> tuple[tuple[int, ...], float, float]:
        joiners = [seq for seq in plan.seqs if seq in self._prefilling]
        if joiners:
            rows = [(self._batch, s) for s in self._batch.seqs] if self._batch is not None else []
            rows += [(self._prefilling[seq], seq) for seq in joiners]
            self._batch = BatchKV.pack(rows)
            for seq in joiners:
                del self._prefilling[seq]
        batch = self._batch
        if batch is None:
            raise AssertionError("a valid decode names at least one sequence")
        fed = dict(zip(plan.seqs, plan.token_ids, strict=True))
        x = mx.array([[fed[seq]] for seq in batch.seqs])
        settings = [
            SampleRow(*self._sampling[seq], self._ledger.held(seq) + 1) for seq in batch.seqs
        ]
        forward_began = time.perf_counter()
        logits = self.model(x, cache=batch.layers)
        sampled = sample(logits[:, -1, :], settings)
        mx.eval(sampled)
        forward_done = time.perf_counter()
        by_row = cast(list[int], sampled.tolist())
        return tuple(by_row[batch.row(seq)] for seq in plan.seqs), forward_began, forward_done
