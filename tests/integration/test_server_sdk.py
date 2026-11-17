"""A1's proof: the official OpenAI Python SDK against the server, streaming and not."""

from collections.abc import Iterator
from contextlib import contextmanager

import openai
import pytest
from starlette.testclient import TestClient
from tests.recording import RecordingRunner

from ullm.core.engine import EngineConfig
from ullm.runner.base import RunnerInfo
from ullm.runner.fake_runner import FakeRunner, FakeTokenizer, Latency
from ullm.server.app import ServeConfig, create_app

SERVED = "fake-model"
INFO = RunnerInfo(model_id="fake", vocab_size=300, eos_token_ids=frozenset({0}), max_model_len=64)
FAST = Latency(
    prefill_base_ms=0.2, prefill_per_token_ms=0.0, decode_base_ms=0.2, decode_per_row_ms=0.0
)
SLOW = Latency(
    prefill_base_ms=0.2, prefill_per_token_ms=0.0, decode_base_ms=50.0, decode_per_row_ms=0.0
)
MESSAGES: list[openai.types.chat.ChatCompletionMessageParam] = [{"role": "user", "content": "hi"}]


@contextmanager
def sdk_for(latency: Latency, engine: EngineConfig | None = None) -> Iterator[openai.OpenAI]:
    runner = RecordingRunner(FakeRunner(INFO, latency))
    config = ServeConfig(served_model=SERVED, engine=engine or EngineConfig())
    with TestClient(
        create_app(runner, FakeTokenizer(), config), base_url="http://127.0.0.1"
    ) as http:
        yield openai.OpenAI(
            base_url="http://127.0.0.1/v1", api_key="unused", http_client=http, max_retries=0
        )


@pytest.fixture
def sdk() -> Iterator[openai.OpenAI]:
    with sdk_for(FAST) as client:
        yield client


def test_the_sdk_gets_a_reply(sdk: openai.OpenAI) -> None:
    reply = sdk.chat.completions.create(model=SERVED, messages=MESSAGES, max_tokens=5, seed=1)
    assert reply.choices[0].finish_reason == "length"
    assert reply.choices[0].message.content
    assert reply.usage is not None
    assert reply.usage.completion_tokens == 5


def test_the_sdk_streams_a_reply_with_usage(sdk: openai.OpenAI) -> None:
    stream = sdk.chat.completions.create(
        model=SERVED,
        messages=MESSAGES,
        max_tokens=5,
        stream=True,
        stream_options={"include_usage": True},
    )
    chunks = list(stream)
    assert chunks[0].choices[0].delta.role == "assistant"
    assert [c.choices[0].finish_reason for c in chunks if c.choices][-1] == "length"
    assert chunks[-1].choices == []
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.completion_tokens == 5


def test_the_sdk_lists_the_model(sdk: openai.OpenAI) -> None:
    assert [m.id for m in sdk.models.list()] == [SERVED]


def test_the_sdk_raises_its_own_errors(sdk: openai.OpenAI) -> None:
    with pytest.raises(openai.BadRequestError) as bad:
        sdk.chat.completions.create(model=SERVED, messages=MESSAGES, n=2)
    assert bad.value.code == "unsupported_parameter"
    with pytest.raises(openai.NotFoundError) as missing:
        sdk.chat.completions.create(model="gpt-4o", messages=MESSAGES)
    assert missing.value.code == "model_not_found"


def test_the_sdk_sees_503_when_the_engine_is_full() -> None:
    with sdk_for(SLOW, EngineConfig(max_in_flight=1)) as sdk:
        with sdk.chat.completions.create(
            model=SERVED, messages=MESSAGES, max_tokens=5, stream=True
        ) as first:
            next(iter(first))
            with pytest.raises(openai.InternalServerError) as full:
                sdk.chat.completions.create(model=SERVED, messages=MESSAGES, max_tokens=5)
        assert full.value.status_code == 503
        assert full.value.code == "queue_full"
