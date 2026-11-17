"""The runner for the real model: mlx-lm's model modules, our own loop and KV (E1, decision 0008).

Only the engine thread calls ``MLXRunner`` (rule 2). ``MLXTokenizer`` holds no
MLX arrays, so other threads may call it.
"""

import secrets
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import mlx.core as mx
from mlx_lm import load
from tokenizers import decoders

from ullm.models import ModelSpec
from ullm.runner.base import (
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
from ullm.runner.kv.contiguous import ContiguousKV
from ullm.runner.ledger import SeqLedger
from ullm.runner.sampler import SampleRow, sample


def load_mlx(spec: ModelSpec) -> tuple["MLXRunner", "MLXTokenizer"]:
    """Load weights, tokenizer and chat template once, at engine startup."""
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
        self._caches: dict[SeqId, list[ContiguousKV]] = {}
        self._sampling: dict[SeqId, tuple[SamplingParams, int]] = {}

    def info(self) -> RunnerInfo:
        return self._info

    def add(self, seq: SeqId, sampling: SamplingParams) -> None:
        self._ledger.add(seq)
        self._caches[seq] = [ContiguousKV() for _ in self.model.layers]
        seed = secrets.randbits(64) if sampling.seed is None else sampling.seed
        self._sampling[seq] = (sampling, seed)

    def release(self, seq: SeqId) -> None:
        self._ledger.release(seq)
        del self._caches[seq]
        del self._sampling[seq]

    def execute(self, plan: StepPlan) -> StepResult:
        if isinstance(plan, DecodePlan) and len(plan.seqs) > 1:
            raise ValueError("batched decode needs one batch KV cache (E7, M2-T1)")
        with self._ledger.step(plan):
            began = time.perf_counter()
            seq, chunks, sampled_at = _forwards(plan, self._ledger.held)
            inputs = [mx.array(chunk)[None] for chunk in chunks]
            cache = self._caches[seq]
            forward_began = time.perf_counter()
            for x in inputs:
                logits = self.model(x, cache=cache)
            if sampled_at is not None:
                params, seed = self._sampling[seq]
                sampled = sample(logits[:, -1, :], [SampleRow(params, seed, sampled_at)])
                mx.eval(sampled)
            else:
                mx.eval([(kv.keys, kv.values) for kv in cache])
            forward_done = time.perf_counter()
            tokens = tuple(cast(list[int], sampled.tolist())) if sampled_at is not None else ()
        done = time.perf_counter()
        return StepResult(
            tokens=tokens,
            forward_ms=(forward_done - forward_began) * 1000,
            total_ms=(done - began) * 1000,
        )


def _forwards(
    plan: StepPlan, held: Callable[[SeqId], int]
) -> tuple[SeqId, list[tuple[int, ...]], int | None]:
    """The sequence, the token chunks to run in order, and the position of the token
    sampled after the last chunk, or None when the step samples nothing.

    A sampled prefill runs all but its last token, then the last token alone, as
    mlx-lm's ``generate_step`` does: one ``[1, L]`` forward picks different greedy
    tokens at near-ties (decision 0008).
    """
    match plan:
        case PrefillPlan(seq=seq, token_ids=ids, start=start, sample=True) if len(ids) > 1:
            return seq, [ids[:-1], ids[-1:]], start + len(ids)
        case PrefillPlan(seq=seq, token_ids=ids, start=start, sample=sample):
            return seq, [ids], start + len(ids) if sample else None
        case DecodePlan(seqs=seqs, token_ids=ids):
            return seqs[0], [ids], held(seqs[0]) + 1
