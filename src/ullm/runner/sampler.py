"""The sampler (E2, decision 0009): one token per row of ``[B, vocab]`` logits.

Each row carries its own settings. A sampled row draws with a key derived from
its sequence's seed and the position the token takes, so a seed reproduces a
reply whatever shares the batch. A sequence rebuilt after preemption draws with
the same keys, but its tokens match only while its logits do: a prefill computes
bf16 logits that differ from decode's (decision 0008).
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx

from ullm.runner.base import SamplingParams


@dataclass(frozen=True, slots=True)
class SampleRow:
    params: SamplingParams
    seed: int  # the request's seed, or one the runner drew for it
    position: int  # the index the sampled token takes in its sequence


def sample(logits: mx.array, rows: Sequence[SampleRow]) -> mx.array:
    """``[B]`` token ids, lazy: the caller evaluates them (rule 4)."""
    if logits.shape[0] != len(rows):
        raise ValueError(f"one sampling row per logits row: {logits.shape[0]} and {len(rows)}")
    return mx.stack([_one(logits[i], row) for i, row in enumerate(rows)])


def _one(logits: mx.array, row: SampleRow) -> mx.array:
    params = row.params
    if params.temperature == 0:
        # The argmax of log-probabilities in the model's dtype, as mlx-lm's greedy sampler
        # takes it: in bf16 the subtraction can tie two close logits (decision 0008).
        return mx.argmax(logits - mx.logsumexp(logits, keepdims=True))
    logprobs = logits.astype(mx.float32)
    logprobs = logprobs - mx.logsumexp(logprobs, keepdims=True)
    # The masks see the untempered distribution, as in mlx-lm (decision 0009).
    if params.top_p < 1:
        logprobs = _top_p(logprobs, params.top_p)
    if 0 < params.top_k < logprobs.shape[-1]:
        logprobs = _top_k(logprobs, params.top_k)
    return mx.random.categorical(logprobs / params.temperature, key=_key(row))


def _top_p(logprobs: mx.array, top_p: float) -> mx.array:
    """Keep the most likely tokens until their mass reaches ``top_p``. Ties at the edge stay."""
    ordered = -mx.sort(-logprobs)
    probs = mx.exp(ordered)
    kept = ((mx.cumsum(probs) - probs) < top_p).sum()  # the most likely always has 0 before it
    return mx.where(logprobs < ordered[kept - 1], -mx.inf, logprobs)


def _top_k(logprobs: mx.array, top_k: int) -> mx.array:
    """Keep the ``top_k`` most likely tokens. Ties with the last one stay."""
    return mx.where(logprobs < mx.topk(logprobs, top_k).min(), -mx.inf, logprobs)


def _key(row: SampleRow) -> mx.array:
    # mx.random.key takes a uint64; a hash maps any int seed into that range.
    digest = hashlib.blake2b(f"{row.seed}:{row.position}".encode(), digest_size=8).digest()
    return mx.random.key(int.from_bytes(digest))
