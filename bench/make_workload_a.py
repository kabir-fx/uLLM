"""Workload A, chat (B2): 256 dialogue prefixes from UltraChat, sized with uLLM's own tokenizer.

    uv run python -m bench.make_workload_a    # writes bench/workloads/a.json

Reads ``stingning/ultrachat`` at a pinned revision in file order, streaming only the rows it
needs. Each request is the longest of the first 1, 3 or 5 turns that renders to at most 1,000
prompt tokens, keeping prompts of at least 100. Decision 0013 has the reasons.
"""

import json
import random
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bench.schema import Message, Source, Workload, WorkloadRequest
from huggingface_hub import HfFileSystem

from ullm.models import DEFAULT_MODEL
from ullm.runner.base import TemplateMessage

if TYPE_CHECKING:
    from ullm.runner.mlx_runner import MLXTokenizer

OUT = Path(__file__).resolve().parents[1] / "bench" / "workloads" / "a.json"
SOURCE = Source(
    dataset="stingning/ultrachat",
    revision="f220fe796ce3ed62fbe1681b45ce6cbc9c6cabe0",
    file="train_0.jsonl",
    licence="MIT",
)
TEMPLATE_KWARGS = {"enable_thinking": False}
SEED = 0
COUNT = 256
MIN_PROMPT, MAX_PROMPT = 100, 1000
MIN_OUTPUT, MAX_OUTPUT = 50, 300
TURNS = (5, 3, 1)  # odd, so a prefix ends with a user turn


def dialogues() -> Iterator[list[str]]:
    path = f"datasets/{SOURCE.dataset}@{SOURCE.revision}/{SOURCE.file}"
    with HfFileSystem().open(path, "r", block_size=1 << 20) as f:
        for line in f:
            yield json.loads(line)["data"]


def _messages(turns: list[str]) -> list[TemplateMessage]:
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": text}
        for i, text in enumerate(turns)
    ]


def sized(tokenizer: "MLXTokenizer", turns: list[str]) -> tuple[list[TemplateMessage], int] | None:
    """The longest allowed prefix and its prompt length, or None if none fits."""
    for k in TURNS:
        if k > len(turns):
            continue
        messages = _messages(turns[:k])
        n = len(tokenizer.render(messages, TEMPLATE_KWARGS))
        if n <= MAX_PROMPT:
            return (messages, n) if n >= MIN_PROMPT else None
    return None


def same_unsanitized(tokenizer: "MLXTokenizer", messages: list[TemplateMessage]) -> bool:
    """True when no message holds control-token text, so a server that does not sanitize it
    (mlx_lm.server) renders the same ids as uLLM (decision 0011)."""
    plain: Any = tokenizer.wrapped.apply_chat_template(
        list(messages), add_generation_prompt=True, **TEMPLATE_KWARGS
    )
    return tuple(plain) == tokenizer.render(messages, TEMPLATE_KWARGS)


def build() -> Workload:
    from ullm.runner.mlx_runner import load_mlx  # MLX exists only on Apple Silicon

    _, tokenizer = load_mlx(DEFAULT_MODEL)
    rng = random.Random(SEED)
    requests: list[WorkloadRequest] = []
    for turns in dialogues():
        found = sized(tokenizer, turns)
        if found is None or not same_unsanitized(tokenizer, found[0]):
            continue
        messages, prompt_tokens = found
        requests.append(
            WorkloadRequest(
                id=f"a-{len(requests):03d}",
                messages=tuple(Message.model_validate(m) for m in messages),
                max_tokens=rng.randint(MIN_OUTPUT, MAX_OUTPUT),
                prompt_tokens=prompt_tokens,
            )
        )
        if len(requests) == COUNT:
            break
    return Workload(
        name="A",
        source=SOURCE,
        tokenizer=f"{DEFAULT_MODEL.repo}@{DEFAULT_MODEL.revision}",
        chat_template_kwargs=TEMPLATE_KWARGS,
        seed=SEED,
        requests=tuple(requests),
    )


def main() -> int:
    workload = build()
    OUT.write_text(workload.model_dump_json(indent=1) + "\n")
    lengths = sorted(r.prompt_tokens for r in workload.requests)
    print(f"wrote {len(lengths)} requests to {OUT}")
    print(
        f"prompt tokens: min {lengths[0]}, median {lengths[len(lengths) // 2]}, max {lengths[-1]}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
