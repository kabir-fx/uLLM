"""The goldens are trustworthy on this machine (F4).

Before uLLM is compared against the goldens (M1-T5 on), these checks establish
that mlx-lm itself reproduces them here: the installed versions match, every
prompt renders to the same token IDs, and greedy generation gives the same
tokens. On CI this is what tells us whether the hosted runner's GPU agrees with
the machine the goldens were made on (decision 0005).

Needs the model: ``make models``. Without it these tests skip, unless
``ULLM_REQUIRE_MODEL=1`` (set in CI), where a missing model is a failure.
"""

import json
import os
from importlib.metadata import version
from typing import Any

import pytest

pytestmark = pytest.mark.mlx

pytest.importorskip("mlx.core")

from scripts.fetch_models import local_path  # noqa: E402
from scripts.make_goldens import (  # noqa: E402
    PROMPTS,
    TEMPLATE_KWARGS,
    golden_path,
    load_model,
    reference,
    render,
)

from ullm.models import DEFAULT_MODEL  # noqa: E402

# CI generates this many tokens per prompt (about 32, catalog F2); a full-length
# comparison is `make goldens-check`.
TOKENS = int(os.environ.get("ULLM_GOLDEN_TOKENS", "32"))


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
    diverged = []
    for case in golden["cases"]:
        want = case["tokens"][:TOKENS]
        got, _, _ = reference(model, tokenizer, case["prompt_ids"], max_tokens=len(want))
        if got != want:
            i = next(
                (n for n, (a, b) in enumerate(zip(got, want, strict=False)) if a != b), len(got)
            )
            diverged.append(
                f"{case['id']}: first difference at token {i}, "
                f"golden top-2 margin there {case['top2_margins'][i]}"
            )
    assert not diverged, "mlx-lm output differs from the goldens on this machine:\n" + "\n".join(
        diverged
    )
