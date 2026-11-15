"""C1 request validation and error mapping (decision 0003)."""

from typing import Any

import pytest
from pydantic import ValidationError

from ullm.server.schemas import ApiError, ChatCompletionRequest, Usage

MODEL = "mlx-community/Qwen3-0.6B-bf16"


def request(**fields: Any) -> dict[str, Any]:
    return {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], **fields}


def rejected(body: dict[str, Any]) -> ApiError:
    with pytest.raises(ValidationError) as info:
        ChatCompletionRequest.model_validate(body)
    return ApiError.from_validation(info.value)


def test_minimal_request_gets_openai_defaults() -> None:
    req = ChatCompletionRequest.model_validate(request())
    assert (req.stream, req.temperature, req.top_p, req.top_k) == (False, 1.0, 1.0, 0)
    assert req.stop == ()
    assert req.output_limit is None
    assert req.include_usage is False
    assert req.ignored_fields == []


def test_null_means_default_for_sampling_fields() -> None:
    req = ChatCompletionRequest.model_validate(
        request(temperature=None, top_p=None, top_k=None, stream=None)
    )
    assert (req.stream, req.temperature, req.top_p, req.top_k) == (False, 1.0, 1.0, 0)


def test_developer_role_reaches_the_template_as_system() -> None:
    req = ChatCompletionRequest.model_validate(
        request(messages=[{"role": "developer", "content": "be brief"}])
    )
    assert req.messages[0].template_role == "system"


def test_text_parts_are_joined_with_newlines() -> None:
    parts = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    req = ChatCompletionRequest.model_validate(
        request(messages=[{"role": "user", "content": parts}])
    )
    assert req.messages[0].text == "a\nb"


def test_image_part_is_unsupported_and_named() -> None:
    parts = [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {"url": "x"}}]
    err = rejected(request(messages=[{"role": "user", "content": parts}]))
    assert (err.status, err.code, err.param) == (400, "unsupported_value", "messages.0.content")
    assert "image_url" in err.message


def test_tool_messages_are_unsupported_until_tool_calling() -> None:
    err = rejected(request(messages=[{"role": "tool", "content": "42"}]))
    assert (err.code, err.param) == ("unsupported_value", "messages.0.role")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n", 2),
        ("logprobs", True),
        ("top_logprobs", 3),
        ("presence_penalty", 0.5),
        ("frequency_penalty", -1.0),
        ("logit_bias", {"42": 5}),
        ("tools", [{"type": "function", "function": {"name": "f"}}]),
        ("tool_choice", "required"),
        ("response_format", {"type": "json_object"}),
    ],
)
def test_output_changing_fields_are_rejected_not_ignored(field: str, value: Any) -> None:
    err = rejected(request(**{field: value}))
    assert (err.status, err.code, err.param) == (400, "unsupported_parameter", field)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n", 1),
        ("logprobs", False),
        ("presence_penalty", 0),
        ("frequency_penalty", 0.0),
        ("logit_bias", {}),
        ("tools", []),
        ("tool_choice", "none"),
        ("response_format", {"type": "text"}),
        ("n", None),
    ],
)
def test_no_op_values_of_unsupported_fields_are_accepted(field: str, value: Any) -> None:
    ChatCompletionRequest.model_validate(request(**{field: value}))


def test_unknown_fields_are_ignored_and_reported() -> None:
    req = ChatCompletionRequest.model_validate(request(metadata={"k": "v"}, user="u1"))
    assert req.ignored_fields == ["metadata", "user"]


def test_stream_options_need_stream() -> None:
    err = rejected(request(stream_options={"include_usage": True}))
    assert (err.code, err.param) == ("invalid_value", "stream_options")
    req = ChatCompletionRequest.model_validate(
        request(stream=True, stream_options={"include_usage": True})
    )
    assert req.include_usage


def test_output_limit_from_either_field_and_conflicts_rejected() -> None:
    assert ChatCompletionRequest.model_validate(request(max_tokens=7)).output_limit == 7
    assert ChatCompletionRequest.model_validate(request(max_completion_tokens=9)).output_limit == 9
    both = request(max_tokens=5, max_completion_tokens=5)
    assert ChatCompletionRequest.model_validate(both).output_limit == 5
    err = rejected(request(max_tokens=5, max_completion_tokens=6))
    assert (err.code, err.param) == ("invalid_value", "max_tokens")
    assert rejected(request(max_tokens=0)).param == "max_tokens"


def test_stop_normalised_and_bounded() -> None:
    assert ChatCompletionRequest.model_validate(request(stop="\n\n")).stop == ("\n\n",)
    assert ChatCompletionRequest.model_validate(request(stop=["a", "b"])).stop == ("a", "b")
    assert rejected(request(stop=["a", "b", "c", "d", "e"])).param == "stop"
    assert rejected(request(stop=[""])).param == "stop"


@pytest.mark.parametrize(
    ("field", "value"),
    [("temperature", 2.5), ("temperature", -0.1), ("top_p", 0.0), ("top_p", 1.5), ("top_k", -1)],
)
def test_sampling_ranges(field: str, value: float) -> None:
    err = rejected(request(**{field: value}))
    assert (err.code, err.param) == ("invalid_value", field)


def test_messages_required_and_non_empty() -> None:
    assert rejected(request(messages=[])).param == "messages"
    assert rejected({"model": MODEL}).param == "messages"


def test_user_text_cannot_break_error_formatting() -> None:
    err = rejected(request(messages=[{"role": "user", "content": [{"type": "{oops}"}]}]))
    assert "{oops}" in err.message


@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (400, "invalid_request_error"),
        (401, "authentication_error"),
        (404, "invalid_request_error"),
        (429, "rate_limit_error"),
        (503, "server_error"),
    ],
)
def test_error_body_shape(status: int, error_type: str) -> None:
    body = ApiError(status, "queue_full", "busy", retry_after_s=1).body().model_dump()
    assert body == {
        "error": {"message": "busy", "type": error_type, "param": None, "code": "queue_full"}
    }


def test_unknown_status_is_a_programming_error() -> None:
    with pytest.raises(ValueError, match="418"):
        ApiError(418, "invalid_value", "teapot")


def test_usage_totals_are_derived() -> None:
    usage = Usage.of(prompt_tokens=14, completion_tokens=8, cached_tokens=4)
    assert usage.total_tokens == 22
    assert usage.prompt_tokens_details.cached_tokens == 4
