"""Rule 1 for one request: ``MLXRunner``'s greedy tokens equal mlx-lm's (E1, decision 0008).

The reference is mlx-lm's own generation loop, run here on the same machine,
in the same session and on the same weights, not the committed goldens: bf16
near-ties flip across machines (decision 0006), but not between two runs on
one GPU. So the match must be exact. Decision 0008 measured that a single
``[1, L]`` prefill differs from mlx-lm at margins of 0.0 and 0.125, which a
near-tie allowance would hide.

Also E1's chat-template golden: ``MLXTokenizer.render`` gives each golden's
prompt token IDs.
"""

import json
import os
from typing import Any

import pytest

pytestmark = pytest.mark.mlx

pytest.importorskip("mlx.core")

from scripts.make_goldens import PROMPTS, TEMPLATE_KWARGS, golden_path, reference  # noqa: E402

from ullm.models import DEFAULT_MODEL  # noqa: E402
from ullm.runner.base import DecodePlan, PrefillPlan, SamplingParams  # noqa: E402
from ullm.runner.mlx_runner import MLXRunner, MLXTokenizer  # noqa: E402

TOKENS = int(os.environ.get("ULLM_GOLDEN_TOKENS", "32"))  # as test_goldens.py; 128 is the full run
GREEDY = SamplingParams(temperature=0.0, top_p=1.0, top_k=0, seed=None)


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(golden_path(DEFAULT_MODEL).read_text())
    return data


def _generate(runner: MLXRunner, seq: int, prompt: tuple[int, ...], max_tokens: int) -> list[int]:
    eos = runner.info().eos_token_ids
    runner.add(seq, GREEDY)
    tokens = list(runner.execute(PrefillPlan(seq=seq, token_ids=prompt, start=0)).tokens)
    while tokens[-1] not in eos and len(tokens) < max_tokens:
        tokens += runner.execute(DecodePlan(seqs=(seq,), token_ids=(tokens[-1],))).tokens
    runner.release(seq)
    return tokens


def test_render_gives_the_golden_prompt_ids(
    golden: dict[str, Any], mlx_loaded: tuple[MLXRunner, MLXTokenizer]
) -> None:
    _, tokenizer = mlx_loaded
    prompts = {c["id"]: c["messages"] for c in json.loads(PROMPTS.read_text())["cases"]}
    for case in golden["cases"]:
        got = tokenizer.render(prompts[case["id"]], TEMPLATE_KWARGS)
        assert got == tuple(case["prompt_ids"]), f"{case['id']}: chat template renders differently"


def test_greedy_tokens_equal_mlx_lm_on_this_machine(
    golden: dict[str, Any], mlx_loaded: tuple[MLXRunner, MLXTokenizer]
) -> None:
    runner, tokenizer = mlx_loaded
    failures = []
    for seq, case in enumerate(golden["cases"]):
        want, margins, _ = reference(runner.model, tokenizer.wrapped, case["prompt_ids"], TOKENS)
        got = _generate(runner, seq, tuple(case["prompt_ids"]), TOKENS)
        if got != want:
            i = next(
                (n for n, (a, b) in enumerate(zip(got, want, strict=False)) if a != b),
                min(len(got), len(want)),
            )
            margin = margins[i] if i < len(margins) else "-"
            failures.append(f"{case['id']}: first difference at token {i}, mlx-lm margin {margin}")
    assert not failures, "MLXRunner differs from mlx-lm:\n" + "\n".join(failures)
