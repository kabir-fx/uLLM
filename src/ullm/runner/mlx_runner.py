"""The runner for the real model: mlx-lm's model modules, our own loop and KV (E1, decision 0008).

Only the engine thread calls ``MLXRunner`` (rule 2). ``MLXTokenizer`` holds no
MLX arrays, so other threads may call it.
"""

import time
from collections.abc import Mapping, Sequence
from typing import Any, cast

import mlx.core as mx
from mlx_lm import load

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
from ullm.runner.kv.contiguous import ContiguousKV
from ullm.runner.ledger import SeqLedger


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
    return MLXRunner(model, info), MLXTokenizer(wrapped)


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
    def __init__(self, wrapped: Any) -> None:
        self.wrapped = wrapped  # mlx-lm's TokenizerWrapper

    def render(
        self, messages: Sequence[TemplateMessage], template_kwargs: Mapping[str, object]
    ) -> tuple[int, ...]:
        fixed = FIXED_TEMPLATE_OPTIONS & template_kwargs.keys()
        if fixed:
            raise ValueError(f"render fixes these chat template options: {sorted(fixed)}")
        ids = self.wrapped.apply_chat_template(
            list(messages), add_generation_prompt=True, **template_kwargs
        )
        if not all(type(t) is int for t in ids):
            raise ValueError("render takes one conversation, not a batch")
        return tuple(ids)


class MLXRunner:
    def __init__(self, model: Any, info: RunnerInfo) -> None:
        self.model = model  # an mlx-lm model module: model(inputs, cache=[one per layer])
        self._info = info
        self._ledger = SeqLedger(info)
        self._caches: dict[SeqId, list[ContiguousKV]] = {}

    def info(self) -> RunnerInfo:
        return self._info

    def add(self, seq: SeqId, sampling: SamplingParams) -> None:
        if sampling.temperature != 0:
            raise ValueError("only greedy decoding until the sampler lands (M1-T2)")
        self._ledger.add(seq)
        self._caches[seq] = [ContiguousKV() for _ in self.model.layers]

    def release(self, seq: SeqId) -> None:
        self._ledger.release(seq)
        del self._caches[seq]

    def execute(self, plan: StepPlan) -> StepResult:
        if isinstance(plan, DecodePlan) and len(plan.seqs) > 1:
            raise ValueError("batched decode needs one batch KV cache (E7, M2-T1)")
        with self._ledger.step(plan):
            began = time.perf_counter()
            seq, chunks, sample = _forwards(plan)
            inputs = [mx.array(chunk)[None] for chunk in chunks]
            cache = self._caches[seq]
            forward_began = time.perf_counter()
            for x in inputs:
                logits = self.model(x, cache=cache)
            if sample:
                last = logits[:, -1, :]
                sampled = mx.argmax(last - mx.logsumexp(last, axis=-1, keepdims=True), axis=-1)
                mx.eval(sampled)
            else:
                mx.eval([(kv.keys, kv.values) for kv in cache])
            forward_done = time.perf_counter()
            tokens = tuple(cast(list[int], sampled.tolist())) if sample else ()
        done = time.perf_counter()
        return StepResult(
            tokens=tokens,
            forward_ms=(forward_done - forward_began) * 1000,
            total_ms=(done - began) * 1000,
        )


def _forwards(plan: StepPlan) -> tuple[SeqId, list[tuple[int, ...]], bool]:
    """The sequence, the token chunks to run in order, and whether to sample after the last.

    A sampled prefill runs all but its last token, then the last token alone, as
    mlx-lm's ``generate_step`` does: one ``[1, L]`` forward picks different greedy
    tokens at near-ties (decision 0008).
    """
    match plan:
        case PrefillPlan(seq=seq, token_ids=ids, sample=True) if len(ids) > 1:
            return seq, [ids[:-1], ids[-1:]], True
        case PrefillPlan(seq=seq, token_ids=ids, sample=sample):
            return seq, [ids], sample
        case DecodePlan(seqs=seqs, token_ids=ids):
            return seqs[0], [ids], True
