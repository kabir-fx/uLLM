"""A1 over HTTP, in process: C1's replies and errors, and C4's health (decision 0011)."""

import json
import logging
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient
from tests.recording import RecordingRunner, tokens_by_rid

from ullm.core.client import EngineClient, RequestStream
from ullm.core.engine import EngineConfig
from ullm.runner.base import RunnerInfo, SamplingParams
from ullm.runner.fake_runner import FakeRunner, FakeTokenizer, Latency
from ullm.server.app import MAX_LOGGED_FIELDS, ServeConfig, create_app
from ullm.server.schemas import ChatCompletion, ChatCompletionChunk, ErrorResponse
from ullm.telemetry import events as ev

SERVED = "fake-model"
INFO = RunnerInfo(model_id="fake", vocab_size=300, eos_token_ids=frozenset({0}), max_model_len=64)
FAST = Latency(
    prefill_base_ms=0.2, prefill_per_token_ms=0.0, decode_base_ms=0.2, decode_per_row_ms=0.0
)
SLOW = Latency(
    prefill_base_ms=0.2, prefill_per_token_ms=0.0, decode_base_ms=50.0, decode_per_row_ms=0.0
)
PROMPT_TOKENS = len("user\nhi\n")  # FakeTokenizer: one token per byte
TERMINAL = "http://localhost:5173"
LOCAL = "http://127.0.0.1:8000"


def body(**fields: Any) -> dict[str, Any]:
    return {
        "model": SERVED,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5,
        **fields,
    }


class Server:
    def __init__(
        self,
        runner: RecordingRunner | None = None,
        engine: EngineConfig | None = None,
        default_max_tokens: int = 4,
    ) -> None:
        self.runner = runner or RecordingRunner(FakeRunner(INFO, FAST))
        self.events: list[ev.Event] = []
        config = ServeConfig(
            served_model=SERVED,
            default_max_tokens=default_max_tokens,
            engine=engine or EngineConfig(),
        )
        self.app: FastAPI = create_app(self.runner, FakeTokenizer(), config, self.events.append)

    def sampled_text(self, completion_id: str) -> str:
        rid = completion_id.removeprefix("chatcmpl-")
        return "".join(f" t{t}" for t in tokens_by_rid(self.events, self.runner.steps)[rid])


@pytest.fixture
def server() -> Server:
    return Server()


@pytest.fixture
def http(server: Server) -> Iterator[TestClient]:
    with TestClient(server.app, base_url=LOCAL) as client:
        yield client


def sse_events(lines: Iterator[str]) -> list[str]:
    """The ``data:`` payloads, in order."""
    return [line.removeprefix("data: ") for line in lines if line.startswith("data: ")]


def error_of(response: Any) -> tuple[int, str, str | None]:
    err = ErrorResponse.model_validate(response.json()).error
    return response.status_code, err.code, err.param


def test_models_lists_the_served_id(http: TestClient) -> None:
    reply = http.get("/v1/models").json()
    assert [m["id"] for m in reply["data"]] == [SERVED]


def test_a_reply_is_the_engines_tokens_with_exact_usage(server: Server, http: TestClient) -> None:
    response = http.post("/v1/chat/completions", json=body())
    assert response.status_code == 200
    reply = ChatCompletion.model_validate(response.json())
    assert reply.model == SERVED
    assert reply.choices[0].finish_reason == "length"
    assert reply.choices[0].message.content == server.sampled_text(reply.id)
    usage = reply.usage
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (
        PROMPT_TOKENS,
        5,
        PROMPT_TOKENS + 5,
    )


def test_the_default_output_limit_applies_without_max_tokens(http: TestClient) -> None:
    reply = http.post("/v1/chat/completions", json={**body(), "max_tokens": None}).json()
    assert reply["usage"]["completion_tokens"] == 4


@pytest.mark.parametrize("include_usage", [True, False])
def test_a_stream_follows_c1s_order(server: Server, http: TestClient, include_usage: bool) -> None:
    request = body(stream=True, stream_options={"include_usage": include_usage})
    with http.stream("POST", "/v1/chat/completions", json=request) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        payloads = sse_events(response.iter_lines())
    assert payloads[-1] == "[DONE]"
    chunks = [ChatCompletionChunk.model_validate_json(p) for p in payloads[:-1]]
    assert len({c.id for c in chunks}) == 1
    assert chunks[0].choices[0].delta.role == "assistant"
    if include_usage:
        *chunks, usage = chunks
        assert usage.choices == []
        assert usage.usage is not None
        assert (usage.usage.prompt_tokens, usage.usage.completion_tokens) == (PROMPT_TOKENS, 5)
    assert all(c.usage is None for c in chunks)
    assert [c.choices[0].finish_reason for c in chunks] == [None] * (len(chunks) - 1) + ["length"]
    text = "".join(c.choices[0].delta.content or "" for c in chunks)
    assert text == server.sampled_text(chunks[0].id)


def test_a_stop_string_ends_the_reply_and_is_never_sent(http: TestClient) -> None:
    # FakeDetokenizer spells every token " t<id>", so the first token holds the stop string.
    reply = http.post("/v1/chat/completions", json=body(stop=" t")).json()
    assert reply["choices"][0]["finish_reason"] == "stop"
    assert reply["choices"][0]["message"]["content"] == ""
    request = body(stop=" t", stream=True)
    with http.stream("POST", "/v1/chat/completions", json=request) as response:
        payloads = sse_events(response.iter_lines())
    last = json.loads(payloads[-2])["choices"][0]
    assert last["finish_reason"] == "stop"
    assert "content" not in last["delta"]


@pytest.mark.parametrize(
    ("request_body", "status", "code", "param"),
    [
        ("{not json", 400, "invalid_value", None),
        ({"model": SERVED}, 400, "invalid_value", "messages"),
        (body(n=2), 400, "unsupported_parameter", "n"),
        (body(chat_template_kwargs={"tools": []}), 400, "unsupported_parameter", None),
        (body(temperature=1e-40), 400, "invalid_value", "temperature"),
        (body(max_tokens=64 - PROMPT_TOKENS + 1), 400, "context_length_exceeded", "messages"),
        (body(model="gpt-4o"), 404, "model_not_found", "model"),
    ],
    ids=["bad-json", "no-messages", "n", "template-kwargs", "temperature", "context", "model"],
)
def test_errors_use_openais_format(
    http: TestClient, request_body: Any, status: int, code: str, param: str | None
) -> None:
    content = request_body if isinstance(request_body, str) else json.dumps(request_body)
    response = http.post(
        "/v1/chat/completions", content=content, headers={"content-type": "application/json"}
    )
    got_status, got_code, got_param = error_of(response)
    assert (got_status, got_code) == (status, code)
    if param is not None:
        assert got_param == param


def test_an_error_never_echoes_the_prompt(http: TestClient) -> None:
    secret = "my-secret-prompt"
    request = body(messages=[{"role": "user", "content": secret}], max_tokens=64)
    response = http.post("/v1/chat/completions", json=request)
    assert error_of(response)[1] == "context_length_exceeded"
    assert secret not in response.text


def test_a_full_engine_replies_503_queue_full_with_retry_after() -> None:
    server = Server(RecordingRunner(FakeRunner(INFO, SLOW)), EngineConfig(max_in_flight=1))
    with (
        TestClient(server.app, base_url=LOCAL) as http,
        http.stream("POST", "/v1/chat/completions", json=body(stream=True)) as first,
    ):
        assert first.status_code == 200
        refused = http.post("/v1/chat/completions", json=body())
        assert error_of(refused)[:2] == (503, "queue_full")
        assert refused.headers["retry-after"] == "1"


class _FailsSecondStep(RecordingRunner):
    def execute(self, plan: Any) -> Any:
        if len(self.steps) == 1:
            raise RuntimeError("a runner failure")
        return super().execute(plan)


def test_a_failure_mid_stream_sends_an_error_event_and_no_done() -> None:
    server = Server(_FailsSecondStep(FakeRunner(INFO, FAST)))
    with (
        TestClient(server.app, base_url=LOCAL) as http,
        http.stream("POST", "/v1/chat/completions", json=body(stream=True)) as response,
    ):
        payloads = sse_events(response.iter_lines())
        assert response.status_code == 200
        assert http.get("/health").json()["alive"]
    assert "[DONE]" not in payloads
    err = ErrorResponse.model_validate_json(payloads[-1]).error
    assert (err.type, err.code) == ("server_error", "engine_unavailable")
    assert json.loads(payloads[1])["choices"][0]["delta"]["content"]  # the prefill's token


def test_a_failure_before_a_plain_reply_is_503() -> None:
    server = Server(_FailsSecondStep(FakeRunner(INFO, FAST)))
    with TestClient(server.app, base_url=LOCAL) as http:
        response = http.post("/v1/chat/completions", json=body())
    assert error_of(response)[:2] == (503, "engine_unavailable")
    assert response.headers["retry-after"] == "1"


class _DiesOnAdd(RecordingRunner):
    def add(self, seq: int, sampling: SamplingParams) -> None:
        raise RuntimeError("a bug in the core")


def test_health_reports_a_dead_engine_and_requests_get_503() -> None:
    server = Server(_DiesOnAdd(FakeRunner(INFO, FAST)))
    with TestClient(server.app, base_url=LOCAL) as http:
        assert http.get("/health").status_code == 200
        assert error_of(http.post("/v1/chat/completions", json=body()))[:2] == (
            503,
            "engine_unavailable",
        )
        health = http.get("/health")
        assert health.status_code == 503
        assert health.json()["alive"] is False
        refused = http.post("/v1/chat/completions", json=body())
        assert error_of(refused)[:2] == (503, "engine_unavailable")
        assert refused.headers["retry-after"] == "1"


def test_health_reports_the_load(http: TestClient) -> None:
    response = http.get("/health")
    assert response.status_code == 200
    assert response.json() == {"alive": True, "in_flight": 0, "waiting": 0, "running": 0}


def test_x_request_id_is_echoed_only_when_sent(http: TestClient) -> None:
    sent = http.post("/v1/chat/completions", json=body(), headers={"x-request-id": "abc-1"})
    assert sent.headers["x-request-id"] == "abc-1"
    assert "x-request-id" not in http.post("/v1/chat/completions", json=body()).headers
    assert http.get("/health", headers={"x-request-id": "h"}).headers["x-request-id"] == "h"


def test_traceparent_rides_on_the_ticket(server: Server, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str | None] = []
    real: Callable[..., RequestStream] = EngineClient.submit

    def spy(self: EngineClient, *args: Any, **kwargs: Any) -> RequestStream:
        seen.append(kwargs.get("trace"))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(EngineClient, "submit", spy)
    traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    with TestClient(server.app, base_url=LOCAL) as http:
        http.post("/v1/chat/completions", json=body(), headers={"traceparent": traceparent})
        http.post("/v1/chat/completions", json=body())
    assert seen == [traceparent, None]


@pytest.mark.parametrize(("origin", "allowed"), [(TERMINAL, True), ("http://evil.test", False)])
def test_cors_allows_only_the_terminal(http: TestClient, origin: str, allowed: bool) -> None:
    preflight = http.options(
        "/v1/chat/completions",
        headers={
            "origin": origin,
            "access-control-request-method": "POST",
            "access-control-request-headers": "content-type",
        },
    )
    assert (preflight.headers.get("access-control-allow-origin") == origin) is allowed


def test_a_cross_origin_simple_post_runs_nothing(server: Server, http: TestClient) -> None:
    # A page on any origin may send text/plain with no preflight; only JSON is a request.
    response = http.post(
        "/v1/chat/completions",
        content=json.dumps(body()),
        headers={"content-type": "text/plain", "origin": "http://evil.test"},
    )
    assert error_of(response)[:2] == (400, "invalid_value")
    assert server.runner.adds == 0


def test_a_foreign_host_is_refused(server: Server, http: TestClient) -> None:
    # A DNS-rebound page reaches 127.0.0.1 under its own host name.
    response = http.post("/v1/chat/completions", json=body(), headers={"host": "evil.test"})
    assert response.status_code == 400
    assert server.runner.adds == 0
    assert http.get("/health", headers={"host": "localhost:8000"}).status_code == 200


def test_a_body_too_large_for_the_context_is_refused_before_parsing(
    server: Server, http: TestClient
) -> None:
    huge = "x" * 2_000_000  # FakeTokenizer: one token per byte, against a 64-token context
    response = http.post(
        "/v1/chat/completions", json=body(messages=[{"role": "user", "content": huge}])
    )
    assert error_of(response)[:2] == (400, "invalid_value")
    assert huge not in response.text
    assert server.runner.adds == 0


def test_unknown_field_logging_is_bounded(
    http: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    extras = {f"x{i}": 1 for i in range(100)}
    with caplog.at_level(logging.WARNING, logger="ullm.server"):
        for _ in range(3):
            assert http.post("/v1/chat/completions", json=body(**extras)).is_success
    logged = [r for r in caplog.records if r.name.startswith("ullm.server")]
    assert len(logged) == MAX_LOGGED_FIELDS


def test_each_unknown_field_is_logged_once(
    http: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="ullm.server"):
        for _ in range(2):
            assert http.post("/v1/chat/completions", json=body(user="u", metadata={})).is_success
    logged = [r.getMessage() for r in caplog.records if r.name.startswith("ullm.server")]
    assert len(logged) == 2
    assert any("metadata" in m for m in logged)
    assert any("user" in m for m in logged)
