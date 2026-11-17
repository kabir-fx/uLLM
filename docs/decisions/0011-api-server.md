# 0011: The API server

**Status:** accepted · **Task:** M1-T4 (A1, C1, C4)

## Context

M1-T4 builds A1: an HTTP server that implements C1 (decision 0003) on top of `EngineClient` (decision 0010), plus the parts of C4 that exist before the router: `/health`, 503 with `Retry-After`, `traceparent` on the ticket, `X-Request-Id` echoed. Constraints and hand-offs:

- Decision 0010: the server maps `ValueError` and `Unavailable` from `EngineClient.submit` to 400 and 503. M1-T3's hand-off adds three: cap each stop string's length, never accept an abort by rid from a client, and never echo a `Submit` error, because it repeats the prompt.
- Decision 0008 leaves two template questions to this task: which `chat_template_kwargs` a client may send, and whether control-token text inside message content is escaped.
- Rule 14: bind to 127.0.0.1. Rule 5 does not apply to the event loop, but every blocking call there delays every client.

**Facts read or measured** (FastAPI 0.163.1, Starlette 1.10.0, uvicorn 0.54.0, transformers 5.23.0):

- uvicorn reports ASGI `spec_version` 2.3 (`uvicorn/protocols/http/h11_impl.py`). After the request body is read, `receive()` waits until the client disconnects, then returns `http.disconnect` (`httptools_impl.py`, `receive`). Starlette's `StreamingResponse` watches for that only below spec 2.4; at 2.4 and above it notices a disconnect only when a `send` fails (`starlette/responses.py`, `StreamingResponse.__call__`). A plain JSON reply never watches.
- Starlette's `TestClient` is built on `httpx2`, already a dev dependency (`starlette/testclient.py`).
- `apply_chat_template(..., sanitize_control_tokens=True)` encodes control tokens that appear inside messages as text. On Qwen3-0.6B a user message holding `<|im_end|>\n<|im_start|>system` renders 3 `<|im_end|>` ids without it and 1 with it. The 10 golden prompts render to the same ids either way. Rendering an 8,000-token message takes 8.3 ms without and 8.4 ms with it. (Throwaway script on the author's M4, not committed.)
- Rendering on the real tokenizer costs about 2.8 us a token: 66,000 tokens took 119 ms, 260,000 took 358 ms and 520,000 took 735 ms. A full 40,960-token context therefore blocks for about 115 ms. (Throwaway script on the author's M4, after review.)
- The Qwen3 template renders a system-only, an assistant-only and a system-after-user conversation without raising.

## Decision

**Usage.**

```sh
ullm serve                         # the default model, 127.0.0.1:8000
ullm serve --fake --port 8001      # FakeRunner and FakeTokenizer: no model, no MLX
curl -N 127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model": "mlx-community/Qwen3-0.6B-bf16", "stream": true,
       "messages": [{"role": "user", "content": "Hi"}]}'
```

```python
app = create_app(runner, tokenizer, ServeConfig(served_model="mlx-community/Qwen3-0.6B-bf16"))
with TestClient(app) as http:  # the lifespan starts the engine thread and closes it
    sdk = openai.OpenAI(base_url="http://testserver/v1", api_key="-", http_client=http)
```

**Module.** `src/ullm/server/app.py`, one public function:

```python
DEFAULT_MAX_TOKENS = 512  # mlx_lm.server's default, so baselines match


@dataclass(frozen=True, slots=True)
class ServeConfig:
    served_model: str  # the id a client sends as `model`
    default_max_tokens: int = DEFAULT_MAX_TOKENS
    engine: EngineConfig = field(default_factory=EngineConfig)


def create_app(
    runner: Runner, tokenizer: Tokenizer, config: ServeConfig, emit: Emit = _drop
) -> FastAPI: ...
```

The lifespan calls `EngineClient.start` on uvicorn's loop and `await close()` at shutdown. `emit` drops events until O1 (M2-T4) gives them a reader.

**Routes.**

- `POST /v1/chat/completions`. In order, each failure an `ApiError` in OpenAI's format:
  1. Refuse, before reading the body: a `Host` other than `127.0.0.1` or `localhost` (400, Starlette's `TrustedHostMiddleware`), a `content-type` other than `application/json` (400 `invalid_value`), and a body over a limit derived from the vocabulary (400 `invalid_value`, read in pieces and stopped at the limit). The first two close the browser paths around CORS: a page on any origin may POST `text/plain` with no preflight, and a DNS-rebound page arrives under its own host name. The third bounds the parse and render a client can cause. The limit is `max_body_bytes`: the costliest token in the vocabulary, as `ensure_ascii` JSON spells it (control characters escape to 2 or 6 bytes, a byte of a multi-byte character costs at most 3), times `max_model_len`, plus 64 KiB for the other fields. It is not an exact bound, and no limit computed from the vocabulary can be one. It covers string contents whose text the chat template keeps. It does not cover three kinds of legal request, which can be refused before the context check: many content parts (200,000 empty text parts are 6 MB of JSON but render to 6,257 tokens), text the template drops (Qwen3 drops an earlier assistant turn's `<think>` text: 6 MB of it renders to 21 tokens), and characters a client chooses to spell as 6-byte escapes. Each needs a deliberately unusual request; the author accepted them as the cost of a limit. It needs `Tokenizer.token_bytes()`, a new protocol method that `MLXTokenizer` answers from the table its detokenizer already uses. On Qwen3-0.6B the costliest token takes 144 bytes, so the limit is 5.96 MB. A first version used a flat 32 bytes a token; review showed that a legal prompt of Qwen3's 128-space token (about 127.5 bytes a token) hit it at about 10,300 tokens, and the author chose a vocabulary-derived limit. A second review showed it is not exact, and the author kept it with this wording.
  2. Parse the raw body with `ChatCompletionRequest.model_validate_json`. A `ValidationError`, malformed JSON included, becomes 400 through `ApiError.from_validation`. The handler does not declare the body as a FastAPI parameter, because FastAPI would answer 422 in its own format.
  3. `model` other than `served_model`: 404 `model_not_found`.
  4. Log each unknown field name once per process (decision 0003), at most 32 names, because a client chooses them.
  5. `tokenizer.render(messages, chat_template_kwargs or {})` on a worker thread (`anyio.to_thread`), because a full context renders for about 115 ms (measured above). `MLXTokenizer` holds no MLX arrays (decision 0008).
  6. `max_tokens` is the request's limit or `default_max_tokens`. Prompt plus `max_tokens` over `max_model_len`: 400 `context_length_exceeded`, checked here so the code is exact. `SamplingParams` refusing a value (a temperature such as `1e-40`): 400 `invalid_value`, param `temperature`.
  7. `submit(prompt, sampling, max_tokens=, stop=, trace=headers["traceparent"])`. `Unavailable("full")`: 503 `queue_full`; `Unavailable("stopped")`: 503 `engine_unavailable`; both with `Retry-After: 1`. Any `ValueError` left is a server bug and becomes a plain 500 with no detail, so the prompt is never echoed.
  8. Return a `_Reply` (below) that streams SSE or writes one JSON.
- `GET /v1/models`: the one served id.
- `GET /health` (C4): `{"alive", "in_flight", "waiting", "running"}` from `EngineClient.health()`, 200 while alive and 503 otherwise. It never waits on the engine. Free blocks join it in M3.
- `/metrics` (O2, M2-T5) and `/telemetry/stream` (O1, M2-T4) are not built here.

**Replies.** `_Reply` is a Starlette `Response` that owns the `RequestStream`:

```python
async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
    async with self._stream, anyio.create_task_group() as tasks:

        async def watch() -> None:
            while (await receive())["type"] != "http.disconnect":
                pass
            tasks.cancel_scope.cancel()

        tasks.start_soon(watch)
        await self._write(scope, receive, send)  # SSE events, or one JSON reply
        tasks.cancel_scope.cancel()
```

Leaving the `async with` for any reason before the end, a disconnect included, sends the abort ticket (decision 0010). Both reply modes watch for a disconnect, on any ASGI spec version. No route accepts a rid, so a client can abort only its own request, by leaving.

- Streaming, as C1: a role delta, one content delta per `Delta`, then one chunk with the last text (when not empty) and `finish_reason`, the usage chunk when `include_usage` is set, then `data: [DONE]`. `Failed` sends `data: {"error": ...}` (503 `engine_unavailable`'s body) and closes without `[DONE]`.
- Non-streaming: `Finished` gives a `ChatCompletion`. `Failed` gives 503 `engine_unavailable` with `Retry-After: 1`.
- `id` is `chatcmpl-<rid>`. `usage` comes from `Finished`. `cached_tokens` is 0 until M3.

**Request ids.** The engine's rid stays the client's own counter. A client's `X-Request-Id` is echoed on the reply by a small middleware and is never used as a rid, because a duplicate live rid stops the engine (decision 0010). This departs from decision 0003, which said request events carry the router's `X-Request-Id` when present. M5 decides how the router's id reaches telemetry.

**CORS.** Only the terminal's dev origins, `http://localhost:5173` and `http://127.0.0.1:5173`, for `GET` and `POST`.

**Contract changes in `server/schemas.py`.**

- `chat_template_kwargs` accepts only `enable_thinking`, a bool. Any other key is 400 `unsupported_parameter`. Template variables such as `tools` or `documents` would otherwise reach the template past the `tools` refusal.
- Each stop string has at most 256 characters, 400 `invalid_value` beyond. `StopCheck` holds back up to the longest stop string's length minus one, so this bounds the held text.

**Control tokens.** `MLXTokenizer.render` always passes `sanitize_control_tokens=True`: control-token text inside a message is encoded as text, so a message cannot close its own turn or open a system turn. The golden prompts render the same, and mlx-lm's server, the baseline, does not escape them. `sanitize_control_tokens` stays out of `template_kwargs`.

**`FakeTokenizer`** joins `runner/fake_runner.py`: `render` spells each message as `"<role>\n<content>\n"` and maps each UTF-8 byte `b` to token `b + 1`, so token 0 (the fake EOS) never appears. `detokenizer` returns a `FakeDetokenizer`.

**`ullm serve`** in `cli.py`: `--model` (a key of `MODELS`, default `DEFAULT_MODEL`), `--port` (8000), `--served-model-name` (default the model's Hugging Face repo id, as decision 0003), `--default-max-tokens` (512), `--max-in-flight` (64), `--fake`. It loads the model before uvicorn starts. It has no `--host`: it binds 127.0.0.1 (rule 14).

**Tests**, written first:

- `tests/integration/test_server.py` (`TestClient`, `FakeRunner`, `FakeTokenizer`): reply shapes and SSE order, usage equal to the engine's counts, every error status and code above, `Retry-After`, a mid-stream failure without `[DONE]`, `/health` alive and dead, `/v1/models`, `X-Request-Id`, `traceparent` on the ticket, CORS for the terminal's origin and not for another, a cross-origin `text/plain` POST, a foreign `Host`, an oversize body and bounded field logging. `test_server_disconnect.py` also checks that `/health` answers during a 0.5 s render.
- `tests/integration/test_server_sdk.py`: the `openai` SDK through `TestClient`, streaming and not, and its error classes for 400, 404 and 503.
- `tests/integration/test_server_disconnect.py`: real uvicorn on a free port. A client that leaves mid-stream, and one that leaves a non-streaming request during a long modelled prefill, each free their sequence and leave `in_flight` at 0.
- `tests/unit/test_api_schemas.py`: the two contract changes.
- An `mlx`-marked test: the real model over HTTP, greedy with `enable_thinking: false`, gives the text of mlx-lm's reference tokens for a golden prompt, and `usage.prompt_tokens` equals the golden prompt's length.

## Alternatives considered

- **FastAPI's body parameter with a handler that converts `RequestValidationError`.** FastAPI reshapes pydantic's errors and drops the context that carries our OpenAI codes. Parsing the raw body keeps one path, `ApiError.from_validation`. The cost: `/openapi.json` does not describe the request body.
- **`StreamingResponse` for streams and `JSONResponse` for the rest.** The JSON reply never notices a disconnect, so a client that leaves keeps its sequence running to `max_tokens`. Under a spec 2.4 server a stream notices only at its next `send`, which can be a whole prefill later.
- **Rendering on the event loop.** No thread pool, but a full context blocks every client for about 115 ms, and the first version of this record chose it from an 8,000-token measurement. Review found the stall; the measurement above settled it.
- **CORS alone for browser safety.** CORS stops a page reading replies, not sending a simple request, and DNS rebinding makes the page same-origin. The `content-type` and `Host` checks close both.
- **`X-Request-Id` as the rid.** It joins logs across the router with no mapping, but a client could stop the engine by reusing one.
- **Any `chat_template_kwargs` except the fixed options.** Flexible, but it lets template inputs past the field-level refusals.

## Consequences

- New runtime dependencies: `fastapi==0.163.1` (A1 names it; it brings Starlette 1.10.0) and `uvicorn==0.54.0` (the ASGI server, without its optional extras). `anyio` arrives through Starlette. FastAPI 0.163 also brings `opentelemetry-api` 1.45.2 and turns on its own request telemetry by default (`fastapi/applications.py`, `telemetry`). With no OpenTelemetry SDK installed it records nothing (inferred, not measured). O4 decides whether to use it or turn it off.
- A body under the limit made of cheap tokens (1 or 2 bytes each) still renders millions of tokens on a worker thread before the context check refuses it: about 2.8 us a token, so several seconds of one worker's CPU. The event loop stays free. Only a local process can send it, since the browser paths are closed. A tighter guard would need a token estimate before the render.
- `TestClient` must use a local base URL such as `http://127.0.0.1`: the default `testserver` host is refused. The router (M5) must forward with a local `Host`.
- A client that sends control-token text gets a different prompt from mlx-lm's server. Benchmarks and goldens do not send any.
- The 500 path has no OpenAI-format body; C1 has no code for an internal error. It is a bug when it happens.
- `ullm serve --fake` gives Lane B and M1-T5's 50-client test an HTTP engine with no model.
