"""C1 against the official OpenAI Python SDK, in process.

An ``httpx2.MockTransport`` stands in for the server: the SDK's real request
body is parsed with our request model, and our reply models are serialised
back for the SDK to parse. This pins the contract before the server exists;
M1-T5 repeats it against the running server.
"""

import json
from collections.abc import Callable

import httpx2
import openai
import pytest
from pydantic import ValidationError

from ullm.server.schemas import (
    ApiError,
    AssistantMessage,
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    Choice,
    ChunkChoice,
    Delta,
    Usage,
)

MODEL = "mlx-community/Qwen3-0.6B-bf16"
ID = "chatcmpl-r1-000001"
CREATED = 1_700_000_000


def client_for(handler: Callable[[ChatCompletionRequest], httpx2.Response]) -> openai.OpenAI:
    def transport(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/v1/chat/completions"
        parsed = ChatCompletionRequest.model_validate_json(request.content)
        return handler(parsed)

    return openai.OpenAI(
        api_key="unused",
        base_url="http://127.0.0.1:8000/v1",
        http_client=httpx2.Client(transport=httpx2.MockTransport(transport)),
        max_retries=0,
    )


def sse(*chunks: ChatCompletionChunk, done: bool = True, error: ApiError | None = None) -> bytes:
    lines = [f"data: {c.model_dump_json(exclude_none=True)}\n\n" for c in chunks]
    if error is not None:
        lines.append(f"data: {error.body().model_dump_json()}\n\n")
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def chunk(delta: Delta, finish: str | None = None) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id=ID,
        created=CREATED,
        model=MODEL,
        choices=[ChunkChoice(delta=delta, finish_reason=finish)],  # type: ignore[arg-type]
    )


def test_non_streaming_round_trip() -> None:
    seen: list[ChatCompletionRequest] = []

    def handler(req: ChatCompletionRequest) -> httpx2.Response:
        seen.append(req)
        reply = ChatCompletion(
            id=ID,
            created=CREATED,
            model=req.model,
            choices=[Choice(message=AssistantMessage(content="2 + 2 = 4."), finish_reason="stop")],
            usage=Usage.of(prompt_tokens=14, completion_tokens=8, cached_tokens=0),
        )
        return httpx2.Response(200, json=json.loads(reply.model_dump_json()))

    out = client_for(handler).chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": "terse"}, {"role": "user", "content": "2+2?"}],
        max_completion_tokens=32,
        temperature=0,
        seed=7,
        stop=["\n\n"],
        extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
    )

    req = seen[0]
    assert [m.template_role for m in req.messages] == ["system", "user"]
    assert (req.output_limit, req.temperature, req.seed, req.stop) == (32, 0.0, 7, ("\n\n",))
    assert req.top_k == 20
    assert req.chat_template_kwargs == {"enable_thinking": False}
    assert req.ignored_fields == []

    assert out.id == ID
    assert out.choices[0].message.content == "2 + 2 = 4."
    assert out.choices[0].finish_reason == "stop"
    assert out.usage is not None
    assert (out.usage.prompt_tokens, out.usage.completion_tokens, out.usage.total_tokens) == (
        14,
        8,
        22,
    )
    assert out.usage.prompt_tokens_details is not None
    assert out.usage.prompt_tokens_details.cached_tokens == 0


def test_streaming_round_trip_in_contract_order() -> None:
    def handler(req: ChatCompletionRequest) -> httpx2.Response:
        assert req.stream
        assert req.include_usage
        body = sse(
            chunk(Delta(role="assistant", content="")),
            chunk(Delta(content="2")),
            chunk(Delta(content=" + 2")),
            chunk(Delta(), finish="length"),
            ChatCompletionChunk(
                id=ID, created=CREATED, model=MODEL, choices=[], usage=Usage.of(14, 2)
            ),
        )
        return httpx2.Response(200, content=body, headers={"content-type": "text/event-stream"})

    stream = client_for(handler).chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "2+2?"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    events = list(stream)

    assert {e.id for e in events} == {ID}
    assert events[0].choices[0].delta.role == "assistant"
    text = "".join(c.delta.content or "" for e in events for c in e.choices)
    assert text == "2 + 2"
    assert events[-2].choices[0].finish_reason == "length"
    assert events[-1].choices == []
    assert events[-1].usage is not None
    assert events[-1].usage.completion_tokens == 2


def test_mid_stream_error_raises_in_the_sdk() -> None:
    def handler(req: ChatCompletionRequest) -> httpx2.Response:
        body = sse(
            chunk(Delta(role="assistant", content="")),
            chunk(Delta(content="par")),
            error=ApiError(503, "engine_unavailable", "engine stopped"),
            done=False,
        )
        return httpx2.Response(200, content=body, headers={"content-type": "text/event-stream"})

    stream = client_for(handler).chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "x"}], stream=True
    )
    with pytest.raises(openai.APIError, match="engine stopped"):
        list(stream)


@pytest.mark.parametrize(
    ("status", "sdk_error"),
    [
        (400, openai.BadRequestError),
        (401, openai.AuthenticationError),
        (404, openai.NotFoundError),
        (429, openai.RateLimitError),
        (503, openai.InternalServerError),
    ],
)
def test_error_statuses_map_to_sdk_exceptions(status: int, sdk_error: type[Exception]) -> None:
    err = ApiError(status, "invalid_value", "nope", param="temperature")

    def handler(req: ChatCompletionRequest) -> httpx2.Response:
        return httpx2.Response(status, json=err.body().model_dump())

    with pytest.raises(sdk_error) as info:
        client_for(handler).chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": "x"}]
        )
    assert isinstance(info.value, openai.APIStatusError)
    assert info.value.code == "invalid_value"
    assert info.value.param == "temperature"


def test_a_rejected_request_reaches_the_sdk_as_bad_request() -> None:
    def transport(request: httpx2.Request) -> httpx2.Response:
        try:
            ChatCompletionRequest.model_validate_json(request.content)
        except ValidationError as exc:
            return httpx2.Response(400, json=ApiError.from_validation(exc).body().model_dump())
        raise AssertionError("expected the request to be rejected")

    client = openai.OpenAI(
        api_key="unused",
        base_url="http://127.0.0.1:8000/v1",
        http_client=httpx2.Client(transport=httpx2.MockTransport(transport)),
        max_retries=0,
    )
    with pytest.raises(openai.BadRequestError) as info:
        client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": "x"}], n=2
        )
    assert (info.value.code, info.value.param) == ("unsupported_parameter", "n")
