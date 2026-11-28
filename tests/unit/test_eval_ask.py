"""Q5: what the exact-match part sends for one GSM8K item, how it scores the reply, and how
the run's counts follow from the items (decision 0035)."""

import asyncio
import json
from decimal import Decimal

import httpx2
import pytest
from evals.data import Item
from evals.exact_match import ask
from evals.schema import ExactMatch, FinishReason, ItemResult

ITEM = Item(index=7, question="What is 2 + 2?", gold=Decimal(4))


def _reply(content: str, finish_reason: str = "stop", completion_tokens: int = 11) -> str:
    return json.dumps(
        {
            "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 30, "completion_tokens": completion_tokens},
        }
    )


def _ask(status: int, body: str) -> tuple[ItemResult, list[dict[str, object]]]:
    sent: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return httpx2.Response(status, text=body)

    async def main() -> ItemResult:
        transport = httpx2.MockTransport(handler)
        async with httpx2.AsyncClient(base_url="http://target", transport=transport) as client:
            return await ask(client, "served-model", ITEM)

    return asyncio.run(main()), sent


def test_one_item_is_one_greedy_request_with_thinking_off() -> None:
    _, sent = _ask(200, _reply("\\boxed{4}"))
    assert sent == [
        {
            "model": "served-model",
            "messages": [
                {
                    "role": "user",
                    "content": "What is 2 + 2?\nPlease reason step by step,"
                    " and put your final answer within \\boxed{}.",
                }
            ],
            "max_tokens": 512,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ]


def test_the_reply_is_scored_and_kept() -> None:
    result, _ = _ask(200, _reply("2 + 2 = 4, so \\boxed{4.0}"))
    assert result == ItemResult(
        index=7,
        gold="4",
        answer="4.0",
        correct=True,
        finish_reason="stop",
        completion_tokens=11,
        reply="2 + 2 = 4, so \\boxed{4.0}",
    )


def test_a_reply_with_no_box_has_no_answer_and_is_wrong() -> None:
    result, _ = _ask(200, _reply("four", finish_reason="length", completion_tokens=512))
    assert (result.answer, result.correct, result.finish_reason) == (None, False, "length")


def test_a_failed_request_stops_the_run() -> None:
    with pytest.raises(httpx2.HTTPStatusError):
        _ask(503, '{"error": {"message": "full"}}')


def _item(correct: bool, answer: str | None, finish: FinishReason, tokens: int) -> ItemResult:
    return ItemResult(
        index=0,
        gold="4",
        answer=answer,
        correct=correct,
        finish_reason=finish,
        completion_tokens=tokens,
        reply="",
    )


def test_the_run_counts_follow_from_the_items() -> None:
    items = [
        _item(True, "4", "stop", 100),
        _item(False, "5", "stop", 200),
        _item(False, None, "stop", 300),
        _item(False, None, "length", 512),
    ]
    run = ExactMatch.of(items, wall_s=4.0, server_peak_footprint_bytes=2**30)
    assert run == ExactMatch(
        items=4,
        correct=1,
        score=0.25,
        no_answer=2,
        truncated=1,
        output_tokens=1112,
        wall_s=4.0,
        output_tok_s=278.0,
        server_peak_footprint_bytes=2**30,
    )
