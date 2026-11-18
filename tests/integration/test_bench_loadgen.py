"""B1: the closed-loop load generator against a real socket and the real server on FakeRunner
(decision 0013)."""

import asyncio

import pytest
from bench.loadgen import Reply, closed_loop
from bench.schema import Message, Sent, WorkloadRequest
from tests.recording import RecordingRunner
from tests.serving import SERVED, serving

from ullm.runner.base import RunnerInfo
from ullm.runner.fake_runner import FakeRunner, FakeTokenizer, Latency
from ullm.telemetry import events as ev

INFO = RunnerInfo(model_id="fake", vocab_size=300, eos_token_ids=frozenset({0}), max_model_len=512)
LATENCY = Latency(
    prefill_base_ms=0.2, prefill_per_token_ms=0.0, decode_base_ms=2.0, decode_per_row_ms=0.0
)
KWARGS = {"enable_thinking": False}


def _request(i: int, content: str | None = None, max_tokens: int = 8) -> WorkloadRequest:
    messages = (Message(role="user", content=content or "x" * (i + 1)),)
    prompt = len(FakeTokenizer().render([{"role": "user", "content": messages[0].content}], {}))
    return WorkloadRequest(
        id=f"r{i}", messages=messages, max_tokens=max_tokens, prompt_tokens=prompt
    )


def _peak_open(records: list[Sent]) -> int:
    edges = sorted(
        [(r.sent_unix, 1) for r in records]
        + [(r.sent_unix + r.end_ms / 1000, -1) for r in records],
        key=lambda e: (e[0], e[1]),
    )
    open_, peak = 0, 0
    for _, step in edges:
        open_ += step
        peak = max(peak, open_)
    return peak


def _run(requests: list[WorkloadRequest], concurrency: int) -> list[Sent]:
    async def main() -> list[Sent]:
        events: list[ev.Event] = []
        async with serving(RecordingRunner(FakeRunner(INFO, LATENCY)), events) as client:
            return await closed_loop(
                client,
                SERVED,
                requests,
                concurrency,
                KWARGS,
                extra_body={},
            )

    return asyncio.run(asyncio.wait_for(main(), timeout=30))


def test_every_reply_is_timed_with_exact_usage() -> None:
    requests = [_request(i, max_tokens=1 + i % 8) for i in range(12)]
    records = _run(requests, concurrency=3)
    assert [r.id for r in records] == [q.id for q in requests]
    for record, request in zip(records, requests, strict=True):
        assert record.error is None
        assert record.usage is not None
        assert record.usage.prompt_tokens == request.prompt_tokens
        assert record.usage.completion_tokens == request.max_tokens
        assert record.finish_reason == "length"
        assert len(record.chunk_ms) >= 1
        assert list(record.chunk_ms) == sorted(record.chunk_ms)
        assert 0 < record.chunk_ms[0] <= record.chunk_ms[-1] <= record.end_ms


@pytest.mark.parametrize("concurrency", [1, 4])
def test_exactly_concurrency_requests_are_open_at_once(concurrency: int) -> None:
    records = _run([_request(i) for i in range(12)], concurrency)
    assert _peak_open(records) == concurrency


def test_requests_are_sent_in_workload_order() -> None:
    records = _run([_request(i) for i in range(12)], concurrency=4)
    sent = [r.sent_unix for r in records]
    assert sent == sorted(sent)


def test_a_refused_request_is_recorded_and_the_loop_goes_on() -> None:
    too_long = _request(1, content="y" * 600)
    records = _run([_request(0), too_long, _request(2)], concurrency=1)
    assert [r.error is None for r in records] == [True, False, True]
    refused = records[1]
    assert refused.error is not None
    assert refused.error.startswith("HTTP 400")
    assert refused.usage is None
    assert refused.chunk_ms == ()


def test_a_role_chunk_is_not_content_and_usage_comes_from_its_own_chunk() -> None:
    reply = Reply()
    reply.take('{"choices":[{"index":0,"delta":{"role":"assistant"}}]}', 1.0)
    reply.take('{"choices":[{"index":0,"delta":{"content":"hi"}}]}', 2.0)
    reply.take('{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}', 3.0)
    usage = '{"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}'
    reply.take(usage, 4.0)
    reply.take("[DONE]", 5.0)
    assert reply.chunk_ms == [2.0]
    assert reply.finish_reason == "stop"
    assert reply.usage is not None
    assert reply.usage.completion_tokens == 2
    assert reply.outcome_error() is None


def test_a_stream_that_fails_or_stops_early_is_an_error() -> None:
    failed = Reply()
    failed.take('{"choices":[{"index":0,"delta":{"content":"a"}}]}', 1.0)
    failed.take('{"error":{"message":"engine failed","type":"server_error"}}', 2.0)
    assert "engine failed" in (failed.outcome_error() or "")

    cut = Reply()
    cut.take('{"choices":[{"index":0,"delta":{"content":"a"}}]}', 1.0)
    assert cut.outcome_error() == "the stream ended without [DONE]"

    no_usage = Reply()
    no_usage.take("[DONE]", 1.0)
    assert no_usage.outcome_error() == "no usage chunk"
