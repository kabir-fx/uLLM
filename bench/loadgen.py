"""The closed-loop load generator (B1): N clients take requests from one queue in workload
order, stream each reply over raw SSE, and time every content chunk (decision 0013)."""

import asyncio
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import httpx2
from bench.schema import RequestRecord, Usage, WorkloadRequest

TEMPERATURE = 0.0
COMMON_FIELDS = (
    "model",
    "messages",
    "max_tokens",
    "temperature",
    "stream",
    "stream_options",
    "chat_template_kwargs",
)


@dataclass
class Reply:
    """What one SSE stream said, built one ``data:`` payload at a time."""

    chunk_ms: list[float] = field(default_factory=list)
    usage: Usage | None = None
    finish_reason: str | None = None
    error: str | None = None
    done: bool = False

    def take(self, data: str, at_ms: float) -> None:
        if data == "[DONE]":
            self.done = True
            return
        try:
            event = json.loads(data)
            if "error" in event:
                self.error = json.dumps(event["error"])
                return
            if usage := event.get("usage"):
                self.usage = Usage(
                    prompt_tokens=usage["prompt_tokens"],
                    completion_tokens=usage["completion_tokens"],
                )
            for choice in event.get("choices", []):
                if choice.get("delta", {}).get("content"):
                    self.chunk_ms.append(at_ms)
                if choice.get("finish_reason"):
                    self.finish_reason = choice["finish_reason"]
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            self.error = f"unreadable payload {data[:200]!r}: {type(e).__name__}: {e}"

    def outcome_error(self) -> str | None:
        if self.error is not None:
            return self.error
        if not self.done:
            return "the stream ended without [DONE]"
        if self.usage is None:
            return "no usage chunk"
        return None


async def send(
    client: httpx2.AsyncClient,
    model: str,
    request: WorkloadRequest,
    template_kwargs: dict[str, bool],
    *,
    extra_body: Mapping[str, object],
) -> tuple[Reply, float]:
    """Stream one request. Returns the reply and its end time, both in ms after the send.
    ``extra_body`` adds fields for one target and may not replace one that every target gets
    (rule 7)."""
    if clash := sorted(extra_body.keys() & set(COMMON_FIELDS)):
        raise ValueError(f"extra body fields replace fields every target gets: {clash}")
    body = {
        "model": model,
        "messages": [m.model_dump() for m in request.messages],
        "max_tokens": request.max_tokens,
        "temperature": TEMPERATURE,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": template_kwargs,
        **extra_body,
    }
    reply = Reply()
    start = time.perf_counter()
    try:
        async with client.stream("POST", "/v1/chat/completions", json=body) as response:
            if response.status_code != 200:
                text = (await response.aread()).decode(errors="replace")
                reply.error = f"HTTP {response.status_code}: {text}"
            else:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        reply.take(line[len("data: ") :], (time.perf_counter() - start) * 1000)
    except httpx2.HTTPError as e:
        reply.error = f"{type(e).__name__}: {e}"
    return reply, (time.perf_counter() - start) * 1000


async def closed_loop(
    client: httpx2.AsyncClient,
    model: str,
    requests: Sequence[WorkloadRequest],
    concurrency: int,
    template_kwargs: dict[str, bool],
    phase: Literal["warmup", "measure"],
    repeat: int,
    run_start: float,
    *,
    extra_body: Mapping[str, object],
) -> list[RequestRecord]:
    """Send every request with at most ``concurrency`` open, in order. ``run_start`` is a
    ``time.perf_counter`` value; records are returned in workload order."""
    queue = list(enumerate(requests))
    queue.reverse()
    records: list[RequestRecord | None] = [None] * len(requests)

    async def client_loop() -> None:
        while queue:
            i, request = queue.pop()
            sent_s = time.perf_counter() - run_start
            reply, end_ms = await send(
                client, model, request, template_kwargs, extra_body=extra_body
            )
            records[i] = RequestRecord(
                phase=phase,
                repeat=repeat,
                id=request.id,
                sent_s=sent_s,
                chunk_ms=tuple(reply.chunk_ms),
                end_ms=end_ms,
                usage=reply.usage,
                finish_reason=reply.finish_reason,
                error=reply.outcome_error(),
            )

    await asyncio.gather(*(client_loop() for _ in range(concurrency)))
    return [r for r in records if r is not None]
