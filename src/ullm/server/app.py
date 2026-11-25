"""A1: the OpenAI-compatible HTTP server over ``EngineClient`` (decision 0011), O1's
``/telemetry/stream`` (decision 0023) and O2's ``/metrics`` (decision 0024)."""

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import anyio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST
from pydantic import ValidationError
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ullm.core.client import EngineClient, RequestStream, Unavailable
from ullm.core.engine import Delta as EngineDelta
from ullm.core.engine import EngineConfig, Failed, Finished
from ullm.core.request import Emit
from ullm.runner.base import Runner, SamplingParams, TemplateMessage, Tokenizer
from ullm.server.schemas import (
    ApiError,
    AssistantMessage,
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    Choice,
    ChunkChoice,
    Delta,
    ModelCard,
    ModelList,
    Usage,
)
from ullm.telemetry import trace
from ullm.telemetry.bus import Bus, Hub, TelemetryClock
from ullm.telemetry.metrics import Metrics

_log = logging.getLogger(__name__)

RETRY_AFTER_S = 1
# Room in a request body for everything but the messages: model, stop strings, options.
BODY_SLACK_BYTES = 64 * 1024
MAX_LOGGED_FIELDS = 32  # unknown field names logged per process; client-chosen, so bounded
LOCAL_HOSTS = ("127.0.0.1", "localhost")  # a DNS-rebound page arrives under its own name
# Docker Desktop reaches the Mac's 127.0.0.1 under this name, and Prometheus cannot override
# Host. The name resolves through the network's DNS, so it may only scrape (decision 0029).
DOCKER_SCRAPE_HOST = re.compile(r"host\.docker\.internal(:\d+)?", re.IGNORECASE)
DEFAULT_MAX_TOKENS = 512  # mlx_lm.server's default, so baselines match
TERMINAL_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")
# uvicorn waits for every open response before the lifespan's exit, and a telemetry stream
# never ends by itself. Past this, it cancels them, so the recording still gets closed.
GRACEFUL_SHUTDOWN_S = 5


@dataclass(frozen=True, slots=True)
class ServeConfig:
    served_model: str  # the id a client sends as `model`
    default_max_tokens: int = DEFAULT_MAX_TOKENS
    engine: EngineConfig = field(default_factory=EngineConfig)
    record: Path | None = None  # write the run's telemetry to this .jsonl.gz
    telemetry_keepalive_s: float = 10.0


def create_app(
    runner: Runner, tokenizer: Tokenizer, config: ServeConfig, emit: Emit | None = None
) -> FastAPI:
    """With ``emit``, the engine's events go there and nothing streams or counts them; without,
    the app runs its own bus and serves ``/telemetry/stream`` and ``/metrics``."""
    info = runner.info()
    max_body = max_body_bytes(tokenizer.token_bytes(), info.max_model_len)
    created = int(time.time())
    logged_fields: set[str] = set()
    engine: EngineClient | None = None

    clock, bus, metrics = TelemetryClock.start(), Bus(), Metrics()
    hub = (
        None
        if emit is not None
        else Hub(bus, clock, config.engine.replica, observe=metrics.observe)
    )
    if config.record is not None and hub is None:
        raise ValueError("record needs the app's own telemetry: pass no emit")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        nonlocal engine
        tasks: list[asyncio.Task[None]] = []
        if hub is not None and config.record is not None:
            config.record.parent.mkdir(parents=True, exist_ok=True)
            raw = config.record.open("wb")
            tasks.append(asyncio.create_task(trace.record(hub.subscribe(), raw)))
        if hub is not None:
            tasks.append(asyncio.create_task(hub.run()))
        engine = EngineClient.start(
            runner, tokenizer.detokenizer, config.engine, emit or bus.emit, clock
        )
        yield
        await engine.close()
        if hub is not None:
            hub.close()
        await asyncio.gather(*tasks)

    def running() -> EngineClient:
        if engine is None:
            raise RuntimeError("the engine starts with the app's lifespan")
        return engine

    app = FastAPI(title="uLLM", lifespan=lifespan)
    app.add_middleware(_EchoRequestId)
    app.add_middleware(_LocalHosts)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=TERMINAL_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["content-type"],
    )

    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError) -> Response:
        return _error_response(exc)

    @app.get("/v1/models")
    async def models() -> ModelList:
        return ModelList(data=[ModelCard(id=config.served_model, created=created)])

    @app.get("/health")
    async def health() -> Response:
        state = running().health()
        return JSONResponse(asdict(state), status_code=200 if state.alive else 503)

    if hub is not None:
        app.state.hub = hub

        @app.get("/telemetry/stream")
        async def telemetry_stream() -> Response:
            return _TelemetryStream(hub, config.telemetry_keepalive_s)

        @app.get("/metrics")
        async def prometheus() -> Response:
            # The engine emits a request's events before it delivers the reply, so after this
            # flush the counters include every reply a client has already read.
            hub.flush()
            load = running().health()
            body = metrics.render(waiting=load.waiting, running=load.running)
            return Response(body, media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        # A page on any origin may POST text/plain with no CORS preflight; JSON needs one.
        if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
            raise ApiError(400, "invalid_value", "content-type must be application/json")
        try:
            req = ChatCompletionRequest.model_validate_json(await _read_body(request, max_body))
        except ValidationError as e:
            raise ApiError.from_validation(e) from None
        if req.model != config.served_model:
            raise ApiError(
                404, "model_not_found", f"model {req.model!r} is not served here", param="model"
            )
        for name in req.ignored_fields:
            if name not in logged_fields and len(logged_fields) < MAX_LOGGED_FIELDS:
                logged_fields.add(name)
                _log.warning("ignoring unknown request field %r", name)

        messages = [TemplateMessage(role=m.template_role, content=m.text) for m in req.messages]
        # About 2.8 us a token on the real tokenizer: 115 ms for a full context (decision 0011).
        prompt = await anyio.to_thread.run_sync(
            tokenizer.render, messages, req.chat_template_kwargs or {}
        )
        max_tokens = req.output_limit or config.default_max_tokens
        if len(prompt) + max_tokens > info.max_model_len:
            raise ApiError(
                400,
                "context_length_exceeded",
                f"the prompt ({len(prompt)} tokens) plus max_tokens ({max_tokens}) exceeds "
                f"the context of {info.max_model_len} tokens",
                param="messages",
            )
        try:
            sampling = SamplingParams(
                temperature=req.temperature, top_p=req.top_p, top_k=req.top_k, seed=req.seed
            )
        except ValueError as e:  # the schema checks every other field's range
            raise ApiError(400, "invalid_value", str(e), param="temperature") from None
        try:
            stream = running().submit(
                prompt,
                sampling,
                max_tokens=max_tokens,
                stop=req.stop,
                trace=request.headers.get("traceparent"),
            )
        except Unavailable as e:
            raise _unavailable(e.why) from None

        return _Reply(stream, req.model, streamed=req.stream, include_usage=req.include_usage)

    return app


def max_body_bytes(token_bytes: Sequence[bytes], max_model_len: int) -> int:
    """The costliest token, as ``ensure_ascii`` JSON spells it, at every context position.
    It covers string contents whose text the chat template keeps. It does not cover the
    overhead of many content parts or text the template drops, such as an earlier turn's
    reasoning, so such a request can be refused before the context check (decision 0011)."""
    return max(map(_json_cost, token_bytes)) * max_model_len + BODY_SLACK_BYTES


def _json_cost(token: bytes) -> int:
    # These bytes as ensure_ascii JSON spells them: control characters escape to 2 or 6 bytes,
    # and a multi-byte character costs at most 3 a byte. A client may spell any character as
    # a 6-byte escape; that is not covered.
    return sum(len(json.dumps(chr(b))) - 2 if b < 0x80 else 3 for b in token)


async def _read_body(request: Request, limit: int) -> bytes:
    too_large = ApiError(400, "invalid_value", f"the request body is over {limit} bytes")
    if int(request.headers.get("content-length") or 0) > limit:
        raise too_large
    body = bytearray()
    async for piece in request.stream():
        body += piece
        if len(body) > limit:
            raise too_large
    return bytes(body)


def _unavailable(why: Literal["full", "stopped"]) -> ApiError:
    if why == "full":
        return ApiError(503, "queue_full", "too many open requests", retry_after_s=RETRY_AFTER_S)
    return ApiError(
        503, "engine_unavailable", "the engine is not running", retry_after_s=RETRY_AFTER_S
    )


def _failed() -> ApiError:
    return ApiError(
        503, "engine_unavailable", "the engine failed this request", retry_after_s=RETRY_AFTER_S
    )


def _error_response(exc: ApiError) -> JSONResponse:
    headers = {} if exc.retry_after_s is None else {"retry-after": str(exc.retry_after_s)}
    return JSONResponse(exc.body().model_dump(), status_code=exc.status, headers=headers)


class _Reply(Response):
    """One request's reply, SSE or one JSON. It owns the stream, so a reply that ends early,
    the client leaving included, aborts the request. It watches for the client leaving itself:
    uvicorn reports a disconnect only to a ``receive`` call, and Starlette's replies do not all
    make one (decision 0011)."""

    def __init__(
        self, stream: RequestStream, model: str, *, streamed: bool, include_usage: bool
    ) -> None:
        super().__init__()
        self._stream = stream
        self._write = self._write_events if streamed else self._write_json
        self._id = f"chatcmpl-{stream.rid}"
        self._created = int(time.time())
        self._model = model
        self._include_usage = include_usage

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async with self._stream, anyio.create_task_group() as tasks:

            async def watch() -> None:
                while (await receive())["type"] != "http.disconnect":
                    pass
                tasks.cancel_scope.cancel()

            tasks.start_soon(watch)
            await self._write(scope, receive, send)
            tasks.cancel_scope.cancel()

    async def _write_json(self, scope: Scope, receive: Receive, send: Send) -> None:
        text: list[str] = []
        async for out in self._stream:
            match out:
                case EngineDelta(text=piece):
                    text.append(piece)
                case Finished(text=piece, reason=reason) as done:
                    text.append(piece)
                    reply = ChatCompletion(
                        id=self._id,
                        created=self._created,
                        model=self._model,
                        choices=[
                            Choice(
                                message=AssistantMessage(content="".join(text)),
                                finish_reason=reason,
                            )
                        ],
                        usage=_usage(done),
                    )
                    await JSONResponse(reply.model_dump(mode="json"))(scope, receive, send)
                case Failed():
                    await _error_response(_failed())(scope, receive, send)

    async def _write_events(self, scope: Scope, receive: Receive, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"text/event-stream; charset=utf-8"),
                    (b"cache-control", b"no-cache"),
                ],
            }
        )

        async def event(data: str) -> None:
            body = f"data: {data}\n\n".encode()
            await send({"type": "http.response.body", "body": body, "more_body": True})

        await event(self._chunk(Delta(role="assistant")).model_dump_json(exclude_none=True))
        async for out in self._stream:
            match out:
                case EngineDelta(text=piece):
                    chunk = self._chunk(Delta(content=piece))
                    await event(chunk.model_dump_json(exclude_none=True))
                case Finished(text=piece, reason=reason) as done:
                    last = self._chunk(Delta(content=piece or None), reason)
                    await event(last.model_dump_json(exclude_none=True))
                    if self._include_usage:
                        usage = ChatCompletionChunk(
                            id=self._id,
                            created=self._created,
                            model=self._model,
                            choices=[],
                            usage=_usage(done),
                        )
                        await event(usage.model_dump_json(exclude_none=True))
                    await event("[DONE]")
                case Failed():
                    await event(_failed().body().model_dump_json())
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    def _chunk(
        self, delta: Delta, finish_reason: Literal["stop", "length"] | None = None
    ) -> ChatCompletionChunk:
        return ChatCompletionChunk(
            id=self._id,
            created=self._created,
            model=self._model,
            choices=[ChunkChoice(delta=delta, finish_reason=finish_reason)],
        )


def _usage(done: Finished) -> Usage:
    return Usage.of(prompt_tokens=done.prompt_tokens, completion_tokens=done.completion_tokens)


class _TelemetryStream(Response):
    """One tab's frames as SSE, until the tab leaves or the hub closes. Its subscription
    lives exactly as long as the reply."""

    def __init__(self, hub: Hub, keepalive_s: float) -> None:
        super().__init__()
        self._hub = hub
        self._keepalive_s = keepalive_s

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"text/event-stream; charset=utf-8"),
                    (b"cache-control", b"no-cache"),
                ],
            }
        )
        async with self._hub.subscribe() as frames, anyio.create_task_group() as tasks:

            async def watch() -> None:
                while (await receive())["type"] != "http.disconnect":
                    pass
                tasks.cancel_scope.cancel()

            tasks.start_soon(watch)
            while True:
                body = b": keepalive\n\n"
                with anyio.move_on_after(self._keepalive_s):
                    frame = await anext(frames, None)
                    if frame is None:
                        break
                    body = f"data: {frame.model_dump_json()}\n\n".encode()
                await send({"type": "http.response.body", "body": body, "more_body": True})
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            tasks.cancel_scope.cancel()


class _LocalHosts:
    """Starlette's host check, except Prometheus in Docker on ``GET /metrics``."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app
        self._checked = TrustedHostMiddleware(app, allowed_hosts=LOCAL_HOSTS)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        scrape = (
            scope["type"] == "http"
            and scope["method"] == "GET"
            and scope["path"] == "/metrics"
            and DOCKER_SCRAPE_HOST.fullmatch(Headers(scope=scope).get("host", ""))
        )
        await (self._app if scrape else self._checked)(scope, receive, send)


class _EchoRequestId:
    """C4: a client's ``X-Request-Id`` comes back on the reply. It is never the engine's rid."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        sent = [v for k, v in scope.get("headers", []) if k == b"x-request-id"]
        if scope["type"] != "http" or not sent:
            await self._app(scope, receive, send)
            return

        async def echo(message: Message) -> None:
            if message["type"] == "http.response.start":
                message["headers"] = [*message.get("headers", []), (b"x-request-id", sent[0])]
            await send(message)

        await self._app(scope, receive, echo)
