"""Golden fixtures for the correctness anchor (rule 1, F4, Q3).

mlx-lm's own greedy generation over the fixed prompts in
``tests/equivalence/prompts.json``, with thinking off, stored with the exact
library versions, model revision and machine that produced them. uLLM's output
must match these token for token.

    uv run python scripts/make_goldens.py           # write the goldens
    uv run python scripts/make_goldens.py --check   # regenerate in memory, compare in full

mlx-lm's ``generate_step`` is used here on purpose: it is the reference that
uLLM is compared against, so it must never appear in engine code (rule 11).
For each generated token we also store the top-1 minus top-2 log-probability
margin the sampler saw, so a later divergence can be classified as a near-tie
or a real bug.
"""

import argparse
import json
import platform
import subprocess
import sys
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import load

from ullm.models import DEFAULT_MODEL, ModelSpec

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "tests" / "equivalence" / "prompts.json"
GOLDENS_DIR = ROOT / "tests" / "equivalence" / "goldens"
MAX_TOKENS = 128
TEMPLATE_KWARGS: dict[str, Any] = {"enable_thinking": False}  # thinking is off everywhere (F4)


def golden_path(spec: ModelSpec) -> Path:
    return GOLDENS_DIR / f"{spec.key}.json"


def load_model(spec: ModelSpec = DEFAULT_MODEL) -> tuple[Any, Any]:
    return load(spec.repo, revision=spec.revision)  # type: ignore[return-value]


def render(tokenizer: Any, messages: list[dict[str, str]]) -> tuple[str, list[int]]:
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, **TEMPLATE_KWARGS
    )
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, **TEMPLATE_KWARGS)
    return text, list(ids)


def reference(
    model: Any, tokenizer: Any, prompt_ids: list[int], max_tokens: int
) -> tuple[list[int], list[float], str]:
    """Greedy tokens, per-token top-2 margins, and the finish reason, from mlx-lm."""
    generate_step = import_module("mlx_lm.generate").generate_step
    eos = set(tokenizer.eos_token_ids)
    tokens: list[int] = []
    margins: list[float] = []
    for token, logprobs in generate_step(mx.array(prompt_ids), model, max_tokens=max_tokens):
        top2 = mx.topk(logprobs.astype(mx.float32), 2)
        margins.append(round(float(top2.max() - top2.min()), 4))
        tokens.append(int(token))
        if tokens[-1] in eos:
            return tokens, margins, "stop"
    return tokens, margins, "length"


def build(spec: ModelSpec = DEFAULT_MODEL, max_tokens: int = MAX_TOKENS) -> dict[str, Any]:
    model, tokenizer = load_model(spec)
    cases = []
    for case in json.loads(PROMPTS.read_text())["cases"]:
        text, ids = render(tokenizer, case["messages"])
        tokens, margins, finish = reference(model, tokenizer, ids, max_tokens)
        cases.append(
            {
                "id": case["id"],
                "prompt_text": text,
                "prompt_ids": ids,
                "tokens": tokens,
                "finish": finish,
                "top2_margins": margins,
                "text": tokenizer.decode(tokens),  # for people reading diffs; not compared
            }
        )
    return {
        "about": "mlx-lm greedy reference output; regenerate with `make goldens` after any "
        "change to the model revision, the prompts or the pinned libraries.",
        "model": {"key": spec.key, "repo": spec.repo, "revision": spec.revision},
        "versions": {
            "mlx": version("mlx"),
            "mlx_lm": version("mlx-lm"),
            "transformers": version("transformers"),
            "python": platform.python_version(),
        },
        "made_on": {
            "chip": _sysctl("machdep.cpu.brand_string"),
            "device": str(mx.default_device()),
        },
        "settings": {
            "sampler": "greedy (argmax of log-probabilities)",
            "max_tokens": max_tokens,
            "chat_template_kwargs": TEMPLATE_KWARGS,
            "eos_token_ids": sorted(tokenizer.eos_token_ids),
        },
        "cases": cases,
    }


def _sysctl(name: str) -> str | None:
    try:
        return subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def compare(golden: dict[str, Any], fresh: dict[str, Any]) -> list[str]:
    """Differences that matter: versions, prompts and tokens. Machine and text are ignored."""
    problems = []
    for key in ("model", "versions", "settings"):
        if golden[key] != fresh[key]:
            problems.append(f"{key}: golden {golden[key]} != now {fresh[key]}")
    for g, f in zip(golden["cases"], fresh["cases"], strict=True):
        if g["prompt_ids"] != f["prompt_ids"]:
            problems.append(f"{g['id']}: prompt renders differently")
        elif g["tokens"] != f["tokens"]:
            i = next(
                (
                    n
                    for n, (a, b) in enumerate(zip(g["tokens"], f["tokens"], strict=False))
                    if a != b
                ),
                min(len(g["tokens"]), len(f["tokens"])),
            )
            margin = g["top2_margins"][i] if i < len(g["top2_margins"]) else None
            problems.append(f"{g['id']}: tokens differ from position {i} (golden margin {margin})")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Write or check the golden fixtures.")
    parser.add_argument("--check", action="store_true", help="exit 1 if the goldens differ")
    args = parser.parse_args()

    fresh = build()
    path = golden_path(DEFAULT_MODEL)
    if args.check:
        problems = compare(json.loads(path.read_text()), fresh)
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1 if problems else 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fresh, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
