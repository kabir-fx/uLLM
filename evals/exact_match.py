"""Exact match on GSM8K through ``ullm serve`` (Q5, decision 0035): one request at a time,
greedy, thinking off, so a second run on the same machine gives the same replies."""

import asyncio
import time
from collections.abc import Sequence
from pathlib import Path

import httpx2
from bench.conditions import footprint
from bench.target import Target, ready
from evals.data import Item
from evals.schema import ExactMatch, ItemResult
from evals.score import answer

PROMPT_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
MAX_TOKENS = 512
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
READY_TIMEOUT_S = 300.0


async def ask(client: httpx2.AsyncClient, served: str, item: Item) -> ItemResult:
    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": served,
            "messages": [{"role": "user", "content": item.question + PROMPT_SUFFIX}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
        },
    )
    response.raise_for_status()
    reply = response.json()
    choice = reply["choices"][0]
    text = choice["message"]["content"] or ""
    got = answer(text)
    return ItemResult(
        index=item.index,
        gold=str(item.gold),
        answer=None if got is None else str(got),
        correct=got == item.gold,
        finish_reason=choice["finish_reason"],
        completion_tokens=reply["usage"]["completion_tokens"],
        reply=text,
    )


async def _run(
    launch: str, port: int, items: Sequence[Item], log: Path
) -> tuple[ExactMatch, list[ItemResult]]:
    with log.open("ab") as out:
        target = Target(launch, out)
        try:
            async with httpx2.AsyncClient(
                base_url=f"http://127.0.0.1:{port}", timeout=None
            ) as client:
                served = await ready(client, target, READY_TIMEOUT_S, None)
                start = time.perf_counter()
                results = [await ask(client, served, item) for item in items]
                wall_s = time.perf_counter() - start
            memory = footprint(target.proc.pid)
        finally:
            target.stop()
    return ExactMatch.of(results, wall_s, None if memory is None else memory[1]), results


def run(
    launch: str, port: int, items: Sequence[Item], log: Path
) -> tuple[ExactMatch, list[ItemResult]]:
    """Start the server from ``launch``, ask it every item in order, read its peak memory, and
    stop it. Any failed request stops the run: a score over fewer items would mislead."""
    return asyncio.run(_run(launch, port, items, log))
