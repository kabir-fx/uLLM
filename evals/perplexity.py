"""Perplexity over a fixed text (Q5, decision 0035). ``ullm serve`` returns no logprobs, so
this runs the model's forward pass directly: it measures the weights, not the server."""

import math
from collections.abc import Sequence
from typing import Any, cast

from evals.schema import Perplexity
from scripts.fetch_models import local_path

from ullm.models import ModelSpec

WINDOW = 1024
BATCH = 8


def windows(tokens: Sequence[int], size: int) -> list[list[int]]:
    return [list(tokens[i : i + size]) for i in range(0, len(tokens) - size + 1, size)]


def load(spec: ModelSpec) -> tuple[Any, Any]:
    """The pinned snapshot's model and tokenizer. Refuses to download one."""
    path = local_path(spec)
    if path is None:
        raise SystemExit(
            f"{spec.repo} is not downloaded: run `uv run python scripts/fetch_models.py {spec.key}`"
        )
    # MLX is absent on Linux, where the unit tests still import this module.
    from mlx_lm import load as mlx_lm_load

    loaded = mlx_lm_load(str(path))
    return loaded[0], loaded[1]


def measure(model: Any, chunks: list[list[int]], batch: int = BATCH) -> Perplexity:
    import mlx.core as mx
    import mlx.nn as nn

    losses = []
    for start in range(0, len(chunks), batch):
        x = mx.array(chunks[start : start + batch])
        logits = model(x[:, :-1]).astype(mx.float32)
        loss = nn.losses.cross_entropy(logits, x[:, 1:], reduction="none")
        mx.eval(loss)
        losses.append(loss.flatten())
    every = mx.concatenate(losses)
    mean = cast(float, every.mean().item())
    sd = cast(float, mx.sqrt(mx.var(every, ddof=1)).item())
    ppl = math.exp(mean)
    return Perplexity(
        window=len(chunks[0]),
        windows=len(chunks),
        tokens=every.size,
        mean_nll=mean,
        perplexity=ppl,
        standard_error=ppl * sd / math.sqrt(every.size),
    )
