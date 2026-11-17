"""B2: the committed workload A has the shape the catalog sets, and on the pinned tokenizer every
prompt renders to its stored length, sanitized or not (decision 0013)."""

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from bench.make_workload_a import COUNT, same_unsanitized
from bench.schema import load_workload

from ullm.runner.base import TemplateMessage

if TYPE_CHECKING:
    from ullm.runner.mlx_runner import MLXRunner, MLXTokenizer

WORKLOAD = load_workload(Path(__file__).resolve().parents[2] / "bench" / "workloads" / "a.json")


def test_workload_a_has_b2s_shape() -> None:
    requests = WORKLOAD.requests
    assert len(requests) == COUNT
    assert len({r.id for r in requests}) == COUNT
    assert all(100 <= r.prompt_tokens <= 1000 for r in requests)
    assert all(50 <= r.max_tokens <= 300 for r in requests)
    assert WORKLOAD.chat_template_kwargs == {"enable_thinking": False}
    for r in requests:
        roles = [m.role for m in r.messages]
        assert roles == ["user", "assistant"] * (len(roles) // 2) + ["user"], r.id


@pytest.mark.mlx
def test_every_prompt_renders_to_its_stored_length_sanitized_or_not(
    mlx_loaded: "tuple[MLXRunner, MLXTokenizer]",
) -> None:
    _, tokenizer = mlx_loaded
    for r in WORKLOAD.requests:
        messages: list[TemplateMessage] = [
            {"role": m.role, "content": m.content} for m in r.messages
        ]
        assert len(tokenizer.render(messages, WORKLOAD.chat_template_kwargs)) == r.prompt_tokens, (
            r.id
        )
        assert same_unsanitized(tokenizer, messages), r.id
