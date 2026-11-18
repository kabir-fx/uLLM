"""B1 and rule 7: what the load generator sends, and how it records a reply it cannot read
(decision 0013)."""

import asyncio
import json

import httpx2
import pytest
from bench.loadgen import Reply, send
from bench.schema import COMMON_FIELDS, Message, WorkloadRequest

REQUEST = WorkloadRequest(
    id="a-000",
    messages=(Message(role="user", content="hi"),),
    max_tokens=7,
    prompt_tokens=9,
)
KWARGS = {"enable_thinking": False}
GOOD = (
    'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
    'data: {"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":1}}\n\n'
    "data: [DONE]\n\n"
)


def _send(
    stream: str, extra_body: dict[str, object] | None = None
) -> tuple[Reply, list[dict[str, object]]]:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, text=stream)

    async def main() -> Reply:
        transport = httpx2.MockTransport(handler)
        async with httpx2.AsyncClient(base_url="http://target", transport=transport) as client:
            reply, _ = await send(
                client, "served-model", REQUEST, KWARGS, extra_body=extra_body or {}
            )
            return reply

    return asyncio.run(main()), bodies


def test_every_target_gets_one_request_body() -> None:
    reply, bodies = _send(GOOD)
    assert reply.outcome_error() is None
    assert bodies == [
        {
            "model": "served-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 7,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ]


def test_extra_body_fields_join_the_one_request_body() -> None:
    _, bodies = _send(GOOD, {"seed": 0})
    _, plain = _send(GOOD)
    assert bodies == [{**plain[0], "seed": 0}]


@pytest.mark.parametrize("field", COMMON_FIELDS)
def test_extra_body_cannot_replace_a_field_every_target_gets(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _send(GOOD, {field: 1})


def test_common_fields_name_the_whole_request_body() -> None:
    _, bodies = _send(GOOD)
    assert set(bodies[0]) == set(COMMON_FIELDS)


def test_a_payload_that_is_not_json_fails_the_request() -> None:
    reply, _ = _send('data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: {not json\n\n')
    assert reply.outcome_error() is not None
    assert "not json" in (reply.outcome_error() or "")


def test_usage_without_completion_tokens_fails_the_request() -> None:
    reply, _ = _send('data: {"choices":[],"usage":{"prompt_tokens":9}}\n\ndata: [DONE]\n\n')
    assert reply.usage is None
    assert reply.outcome_error() is not None


def test_a_json_payload_of_the_wrong_shape_fails_the_request() -> None:
    reply, _ = _send('data: {"choices":[{"delta":null}]}\n\ndata: [DONE]\n\n')
    assert reply.outcome_error() is not None
