"""Rule 1 over HTTP: greedy replies from the server, on the real model, are the text of
mlx-lm's tokens, on this machine (decision 0011)."""

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from starlette.testclient import TestClient

from ullm.models import DEFAULT_MODEL
from ullm.server.app import ServeConfig, create_app

if TYPE_CHECKING:
    from ullm.runner.mlx_runner import MLXRunner, MLXTokenizer

pytestmark = pytest.mark.mlx

pytest.importorskip("mlx.core")

TOKENS = int(os.environ.get("ULLM_GOLDEN_TOKENS", "32"))  # as test_runner_equivalence.py
PROMPTS = Path(__file__).parent / "prompts.json"


def _streamed_text(http: TestClient, request: dict[str, Any]) -> tuple[str, bool]:
    with http.stream("POST", "/v1/chat/completions", json={**request, "stream": True}) as r:
        payloads = [line[6:] for line in r.iter_lines() if line.startswith("data: ")]
    chunks = [json.loads(p) for p in payloads if p != "[DONE]"]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    return text, payloads[-1] == "[DONE]"


def test_greedy_replies_over_http_are_mlx_lms_text(
    mlx_loaded: "tuple[MLXRunner, MLXTokenizer]",
) -> None:
    from scripts.make_goldens import TEMPLATE_KWARGS, golden_path, reference

    runner, tokenizer = mlx_loaded
    golden: list[dict[str, Any]] = json.loads(golden_path(DEFAULT_MODEL).read_text())["cases"]
    messages = {c["id"]: c["messages"] for c in json.loads(PROMPTS.read_text())["cases"]}
    eos = runner.info().eos_token_ids
    failures = []
    app = create_app(runner, tokenizer, ServeConfig(served_model=DEFAULT_MODEL.repo))
    with TestClient(app, base_url="http://127.0.0.1") as http:
        for case in golden:
            want, _, finish = reference(runner.model, tokenizer.wrapped, case["prompt_ids"], TOKENS)
            detokenizer = tokenizer.detokenizer()
            want_text = "".join(detokenizer.add(t) for t in want if t not in eos)
            want_text += detokenizer.flush()
            request = {
                "model": DEFAULT_MODEL.repo,
                "messages": messages[case["id"]],
                "temperature": 0,
                "max_tokens": TOKENS,
                "chat_template_kwargs": TEMPLATE_KWARGS,
            }
            reply = http.post("/v1/chat/completions", json=request).json()
            choice, usage = reply["choices"][0], reply["usage"]
            got = (
                choice["message"]["content"],
                choice["finish_reason"],
                usage["prompt_tokens"],
                usage["completion_tokens"],
            )
            if got != (want_text, finish, len(case["prompt_ids"]), len(want)):
                failures.append(f"{case['id']}: {got!r}")
            if _streamed_text(http, request) != (want_text, True):
                failures.append(f"{case['id']} streamed differs")
    assert not failures, "the server differs from mlx-lm:\n" + "\n".join(failures)
