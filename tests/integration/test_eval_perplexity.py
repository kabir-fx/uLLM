"""Q5: uLLM's perplexity equals mlx-lm's own ``eval_ppl`` on the same windows of the same
text, on the real model (decision 0035)."""

import math
import os
from importlib import import_module

import pytest
from evals import data
from evals.perplexity import load, measure, windows
from scripts.fetch_models import local_path

from ullm.models import DEFAULT_MODEL

pytestmark = pytest.mark.mlx


def test_perplexity_equals_mlx_lm_eval_ppl() -> None:
    if local_path(DEFAULT_MODEL) is None:
        if os.environ.get("ULLM_REQUIRE_MODEL") == "1":
            pytest.fail(f"{DEFAULT_MODEL.repo} is not downloaded and ULLM_REQUIRE_MODEL=1")
        pytest.skip(f"{DEFAULT_MODEL.repo} is not downloaded: run `make models`")
    import mlx.core as mx

    eval_ppl = import_module("mlx_lm.perplexity").eval_ppl

    model, tokenizer = load(DEFAULT_MODEL)
    chunks = windows(tokenizer.encode(data.wikitext()[:20_000]), 256)
    assert len(chunks) >= 9  # more than one batch of 8

    ours = measure(model, chunks)
    ppl, se = eval_ppl(model, mx.array(chunks), batch_size=8)

    assert ours.windows == len(chunks)
    assert ours.tokens == len(chunks) * 255
    assert math.isclose(ours.perplexity, ppl, rel_tol=1e-5)
    assert math.isclose(ours.standard_error, se, rel_tol=1e-4)
    assert math.isclose(ours.perplexity, math.exp(ours.mean_nll), rel_tol=1e-9)
