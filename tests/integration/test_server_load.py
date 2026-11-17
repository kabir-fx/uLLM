"""Q4: 50 OpenAI SDK clients at once, over a real socket, each get their own reply
(decision 0012)."""

import asyncio
from dataclasses import dataclass

import openai
from tests.recording import RecordingRunner, tokens_by_rid
from tests.serving import SERVED, serving

from ullm.runner.base import RunnerInfo
from ullm.runner.fake_runner import FakeRunner, Latency
from ullm.telemetry import events as ev
from ullm.telemetry.check import check_trace
from ullm.telemetry.trace import frames_from

CLIENTS = 50
INFO = RunnerInfo(model_id="fake", vocab_size=300, eos_token_ids=frozenset({0}), max_model_len=4096)
LATENCY = Latency(
    prefill_base_ms=0.2, prefill_per_token_ms=0.0, decode_base_ms=5.0, decode_per_row_ms=0.0
)


@dataclass(frozen=True)
class Reply:
    ids: frozenset[str]
    text: str
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int


def _content(i: int) -> str:
    return f"client {i}"


def _max_tokens(i: int) -> int:
    return 1 + i % 8


async def _ask(sdk: openai.AsyncOpenAI, i: int) -> Reply:
    messages: list[openai.types.chat.ChatCompletionMessageParam] = [
        {"role": "user", "content": _content(i)}
    ]
    if i % 2 == 0:
        reply = await sdk.chat.completions.create(
            model=SERVED, messages=messages, max_tokens=_max_tokens(i)
        )
        assert reply.usage is not None
        return Reply(
            frozenset({reply.id}),
            reply.choices[0].message.content or "",
            reply.choices[0].finish_reason,
            reply.usage.prompt_tokens,
            reply.usage.completion_tokens,
        )
    stream = await sdk.chat.completions.create(
        model=SERVED,
        messages=messages,
        max_tokens=_max_tokens(i),
        stream=True,
        stream_options={"include_usage": True},
    )
    ids, pieces, finish, usage = set(), [], None, None
    async for chunk in stream:
        ids.add(chunk.id)
        if chunk.choices:
            pieces.append(chunk.choices[0].delta.content or "")
            finish = chunk.choices[0].finish_reason or finish
        if chunk.usage is not None:
            usage = chunk.usage
    assert usage is not None
    return Reply(
        frozenset(ids), "".join(pieces), finish, usage.prompt_tokens, usage.completion_tokens
    )


def _peak_open(events: list[ev.Event]) -> int:
    """The most requests the engine held at once: arrived and not yet ended."""
    open_now = peak = 0
    for event in events:
        if isinstance(event, ev.ReqArrived):
            open_now += 1
            peak = max(peak, open_now)
        elif isinstance(event, ev.ReqFinished | ev.ReqAborted):
            open_now -= 1
    return peak


def test_fifty_concurrent_sdk_clients_each_get_their_own_reply() -> None:
    runner = RecordingRunner(FakeRunner(INFO, LATENCY))
    events: list[ev.Event] = []

    async def main() -> list[Reply]:
        async with serving(runner, events) as http:
            sdk = openai.AsyncOpenAI(
                base_url=f"{http.base_url}/v1", api_key="unused", max_retries=0
            )
            replies = await asyncio.gather(*(_ask(sdk, i) for i in range(CLIENTS)))
            assert (await http.get("/health")).json()["in_flight"] == 0
            await sdk.close()
            return replies

    replies = asyncio.run(asyncio.wait_for(main(), timeout=30))

    sampled = tokens_by_rid(events, runner.steps)
    assert all(len(r.ids) == 1 for r in replies)
    rids = [next(iter(r.ids)).removeprefix("chatcmpl-") for r in replies]
    assert len(set(rids)) == CLIENTS
    for i, (reply, rid) in enumerate(zip(replies, rids, strict=True)):
        assert reply.text == "".join(f" t{t}" for t in sampled[rid]), f"client {i}"
        assert reply.finish_reason == "length"
        assert reply.prompt_tokens == len(f"user\n{_content(i)}\n".encode())
        assert reply.completion_tokens == len(sampled[rid]) == _max_tokens(i)
    assert _peak_open(events) >= CLIENTS // 2
    assert runner.adds == runner.releases == CLIENTS
    assert check_trace(frames_from(events, "r1")).problems == []
