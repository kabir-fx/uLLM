"""The goldens are trustworthy here (F4, decision 0006).

These checks establish that mlx-lm itself still produces the goldens: the
installed versions match, every prompt renders to the same token IDs on any
machine, and greedy generation gives the same tokens.

Token reproduction is exact on the kind of machine the goldens were made on.
On other hardware (the CI runner's paravirtualized GPU, another Apple chip),
bf16 arithmetic can flip a near-tied greedy pick, and everything after it
differs. There, a divergence is accepted only where the golden machine itself
saw a near-tie, and it is reported as a warning. A divergence at a confident
position fails everywhere. This is about mlx-lm against its own goldens; uLLM
is compared against mlx-lm run on the same machine (M1-T5, rule 1).

Needs the model: ``make models``. Without it these tests skip, unless
``ULLM_REQUIRE_MODEL=1`` (set in CI), where a missing model is a failure.
"""

import json
import os
import warnings
from importlib.metadata import version
from typing import Any

import pytest

pytestmark = pytest.mark.mlx

pytest.importorskip("mlx.core")

from scripts.fetch_models import local_path  # noqa: E402
from scripts.make_goldens import (  # noqa: E402
    PROMPTS,
    TEMPLATE_KWARGS,
    current_chip,
    golden_path,
    load_model,
    reference,
    render,
)

from ullm.models import DEFAULT_MODEL  # noqa: E402

# CI generates this many tokens per prompt (about 32, catalog F2); a full-length
# comparison is `make goldens-check`.
TOKENS = int(os.environ.get("ULLM_GOLDEN_TOKENS", "32"))

# Provisional (decision 0006): about two bf16 steps at the magnitude of
# Qwen3's logits. M1-T5 settles the documented threshold for rule 1.
NEAR_TIE_MARGIN = 0.25


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(golden_path(DEFAULT_MODEL).read_text())
    return data


@pytest.fixture(scope="module")
def model_and_tokenizer() -> tuple[Any, Any]:
    if local_path(DEFAULT_MODEL) is None:
        if os.environ.get("ULLM_REQUIRE_MODEL") == "1":
            pytest.fail(f"{DEFAULT_MODEL.repo} is not downloaded and ULLM_REQUIRE_MODEL=1")
        pytest.skip(f"{DEFAULT_MODEL.repo} is not downloaded: run `make models`")
    return load_model(DEFAULT_MODEL)


def test_goldens_were_made_with_the_installed_versions(golden: dict[str, Any]) -> None:
    installed = {k: version(k.replace("_", "-")) for k in ("mlx", "mlx_lm", "transformers")}
    made_with = {k: golden["versions"][k] for k in installed}
    assert made_with == installed, "pinned libraries changed: run `make goldens`"
    assert golden["model"]["revision"] == DEFAULT_MODEL.revision
    assert golden["settings"]["chat_template_kwargs"] == TEMPLATE_KWARGS


def test_prompts_render_to_the_golden_token_ids(
    golden: dict[str, Any], model_and_tokenizer: tuple[Any, Any]
) -> None:
    _, tokenizer = model_and_tokenizer
    prompts = {c["id"]: c for c in json.loads(PROMPTS.read_text())["cases"]}
    for case in golden["cases"]:
        text, ids = render(tokenizer, prompts[case["id"]]["messages"])
        assert text == case["prompt_text"], f"{case['id']}: chat template renders differently"
        assert ids == case["prompt_ids"], f"{case['id']}: tokenizer gives different IDs"


def test_mlx_lm_reproduces_the_goldens_here(
    golden: dict[str, Any], model_and_tokenizer: tuple[Any, Any]
) -> None:
    model, tokenizer = model_and_tokenizer
    same_machine = golden["made_on"]["chip"] == current_chip()
    near_ties, failures = [], []
    for case in golden["cases"]:
        want = case["tokens"][:TOKENS]
        got, margins, _ = reference(model, tokenizer, case["prompt_ids"], max_tokens=len(want))
        if got == want:
            continue
        i = next((n for n, (a, b) in enumerate(zip(got, want, strict=False)) if a != b), len(got))
        golden_margin = case["top2_margins"][i]
        note = (
            f"{case['id']}: first difference at token {i}; top-2 margin there "
            f"{golden_margin} on the golden machine, {margins[i] if i < len(margins) else '-'} here"
        )
        if not same_machine and golden_margin <= NEAR_TIE_MARGIN:
            near_ties.append(note)
        else:
            failures.append(note)
    if near_ties:
        origin = golden["made_on"]["chip"]
        warnings.warn(
            f"near-tie divergences on {current_chip()} (goldens from {origin}):\n"
            + "\n".join(near_ties),
            stacklevel=1,
        )
    where = "the golden machine" if same_machine else "a confident position"
    assert not failures, f"mlx-lm differs from the goldens at {where}:\n" + "\n".join(failures)
