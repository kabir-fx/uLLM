"""C1: the OpenAI-compatible chat completions contract (decision 0003).

Request models validate what a client sends; reply models define exactly what
uLLM sends back. A validation failure maps to an OpenAI-format error through
``ApiError.from_validation``. Validators raise ``PydanticCustomError`` whose
type is the OpenAI error code, so the code survives pydantic's error wrapping.
"""

from typing import Any, Final, Literal, Self, cast, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

# --- errors --------------------------------------------------------------------

ErrorType = Literal[
    "invalid_request_error", "authentication_error", "rate_limit_error", "server_error"
]
ErrorCode = Literal[
    "invalid_value",
    "unsupported_parameter",
    "unsupported_value",
    "context_length_exceeded",
    "model_not_found",
    "invalid_api_key",
    "rate_limit_exceeded",
    "queue_full",
    "engine_unavailable",
]
_ERROR_CODES: Final = frozenset(get_args(ErrorCode))
_ERROR_TYPE_BY_STATUS: Final[dict[int, ErrorType]] = {
    400: "invalid_request_error",
    401: "authentication_error",
    404: "invalid_request_error",
    429: "rate_limit_error",
    503: "server_error",
}


class ErrorBody(BaseModel):
    message: str
    type: ErrorType
    param: str | None
    code: ErrorCode


class ErrorResponse(BaseModel):
    """The body of every error reply, and of a mid-stream error event."""

    error: ErrorBody


class ApiError(Exception):
    """An error the API returns to the client in OpenAI's format."""

    def __init__(
        self,
        status: int,
        code: ErrorCode,
        message: str,
        *,
        param: str | None = None,
        retry_after_s: int | None = None,
    ) -> None:
        if status not in _ERROR_TYPE_BY_STATUS:
            raise ValueError(f"no OpenAI error type for HTTP status {status}")
        super().__init__(message)
        self.status = status
        self.code: ErrorCode = code
        self.message = message
        self.param = param
        self.retry_after_s = retry_after_s

    def body(self) -> ErrorResponse:
        return ErrorResponse(
            error=ErrorBody(
                message=self.message,
                type=_ERROR_TYPE_BY_STATUS[self.status],
                param=self.param,
                code=self.code,
            )
        )

    @classmethod
    def from_validation(cls, exc: ValidationError) -> "ApiError":
        """Report one validation problem, as OpenAI does.

        Errors raised by our own validators carry an OpenAI code and win over
        pydantic's generic ones, which can come first when a union fails.
        """
        errors = exc.errors()
        err = next((e for e in errors if e["type"] in _ERROR_CODES), errors[0])
        ctx = err.get("ctx") or {}
        param = ctx.get("param") or ".".join(str(part) for part in err["loc"]) or None
        code = cast(ErrorCode, err["type"]) if err["type"] in _ERROR_CODES else "invalid_value"
        message = err["msg"] if param is None else f"{param}: {err['msg']}"
        return cls(400, code, message, param=param)


def _reject(
    code: ErrorCode, template: str, param: str | None = None, **values: object
) -> PydanticCustomError:
    """A validation error carrying an OpenAI error code as its type.

    Client-supplied text goes in ``values`` and is referenced as ``{name}`` in
    the template, never formatted into the template itself.
    """
    context: dict[str, object] = dict(values)
    if param is not None:
        context["param"] = param
    return PydanticCustomError(code, template, context or None)


# --- request -------------------------------------------------------------------

MAX_STOP_STRINGS: Final = 4

# OpenAI fields that change the output. uLLM does not implement them yet, so a
# value other than the listed no-op values is rejected rather than ignored.
_UNSUPPORTED_UNLESS: Final[dict[str, tuple[Any, ...]]] = {
    "n": (1,),
    "logprobs": (False,),
    "top_logprobs": (0,),
    "presence_penalty": (0,),
    "frequency_penalty": (0,),
    "logit_bias": ({},),
    "tools": ([],),  # tool calling arrives with A2
    "tool_choice": ("none", "auto"),
    "response_format": ({"type": "text"},),  # structured output arrives with A3
}

# Fields where an explicit null means "use the default", as in OpenAI's API.
_NULL_MEANS_DEFAULT: Final = ("stream", "temperature", "top_p", "top_k")


class ContentPart(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["text"]
    text: str


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "developer", "user", "assistant"]
    content: str | list[ContentPart]
    name: str | None = None

    @field_validator("role", mode="before")
    @classmethod
    def _known_role(cls, role: object) -> object:
        if role == "tool":
            raise _reject("unsupported_value", "tool messages are not supported yet")
        return role

    @field_validator("content", mode="before")
    @classmethod
    def _text_parts_only(cls, content: object) -> object:
        # Checked here, before the str-or-list union, so the error names the
        # real problem instead of "input should be a valid string".
        if isinstance(content, list):
            for part in content:
                kind = part.get("type") if isinstance(part, dict) else None
                if kind != "text":
                    raise _reject(
                        "unsupported_value",
                        "content parts of type {kind} are not supported",
                        kind=repr(kind),
                    )
        return content

    @property
    def template_role(self) -> Literal["system", "user", "assistant"]:
        """The role the chat template sees: ``developer`` is OpenAI's newer name for ``system``."""
        return "system" if self.role == "developer" else self.role

    @property
    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "\n".join(part.text for part in self.content)


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")

    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    """``POST /v1/chat/completions``. Unknown fields are kept in ``ignored_fields``."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stop: tuple[str, ...] = ()
    seed: int | None = None
    # Extensions beyond OpenAI's API.
    top_k: int = Field(default=0, ge=0)  # 0 means off
    chat_template_kwargs: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def _screen_fields(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        for field, allowed in _UNSUPPORTED_UNLESS.items():
            value = data.get(field)
            if value is not None and value not in allowed:
                raise _reject(
                    "unsupported_parameter", "{field} is not supported", field, field=field
                )
        return {k: v for k, v in data.items() if not (k in _NULL_MEANS_DEFAULT and v is None)}

    @field_validator("stop", mode="before")
    @classmethod
    def _normalise_stop(cls, stop: object) -> object:
        if stop is None:
            return ()
        stops = [stop] if isinstance(stop, str) else stop
        if not isinstance(stops, list):
            return stops  # let pydantic report the type error
        if len(stops) > MAX_STOP_STRINGS:
            raise _reject("invalid_value", "at most {n} stop strings", "stop", n=MAX_STOP_STRINGS)
        if any(s == "" for s in stops):
            raise _reject("invalid_value", "stop strings must not be empty", "stop")
        return tuple(stops)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.stream_options is not None and not self.stream:
            raise _reject(
                "invalid_value",
                "stream_options is only allowed when stream is true",
                "stream_options",
            )
        both = self.max_tokens is not None and self.max_completion_tokens is not None
        if both and self.max_tokens != self.max_completion_tokens:
            raise _reject(
                "invalid_value", "max_tokens and max_completion_tokens disagree", "max_tokens"
            )
        return self

    @property
    def output_limit(self) -> int | None:
        """The requested output limit; ``None`` means the server default applies."""
        return self.max_completion_tokens or self.max_tokens

    @property
    def include_usage(self) -> bool:
        return self.stream_options is not None and self.stream_options.include_usage

    @property
    def ignored_fields(self) -> list[str]:
        return sorted(self.model_extra or {})


# --- replies -------------------------------------------------------------------

FinishReason = Literal["stop", "length"]  # "tool_calls" arrives with A2


class PromptTokensDetails(BaseModel):
    cached_tokens: int = Field(default=0, ge=0)


class Usage(BaseModel):
    """Exact token counts, taken from the engine, never estimated."""

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    prompt_tokens_details: PromptTokensDetails

    @classmethod
    def of(cls, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> "Usage":
        return cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=cached_tokens),
        )


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None


class Choice(BaseModel):
    index: int = 0
    message: AssistantMessage
    finish_reason: FinishReason
    logprobs: None = None


class ChatCompletion(BaseModel):
    """The non-streaming reply."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage
    system_fingerprint: str | None = None


class Delta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: Delta
    finish_reason: FinishReason | None = None
    logprobs: None = None


class ChatCompletionChunk(BaseModel):
    """One streaming event. The usage chunk has ``choices == []``."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]
    usage: Usage | None = None
    system_fingerprint: str | None = None


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "ullm"


class ModelList(BaseModel):
    """``GET /v1/models``."""

    object: Literal["list"] = "list"
    data: list[ModelCard]
