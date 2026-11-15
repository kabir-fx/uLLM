# 0003: Contracts v0 (C1 API, C2 telemetry, C3 runner)

**Status:** accepted (approved by the author before implementation) · **Task:** M0-T4 `[design]`

Contracts are what the lanes depend on, so they are fixed before the code that uses them. This record makes the open details of catalog entries C1–C3 concrete. C4 and C5 are designed in the tasks that build them (M1-T4, M1-T6).

## Where each contract lives

| Contract | Python source of truth | Exported artifacts | Checked by |
|---|---|---|---|
| C1 OpenAI API | `src/ullm/server/schemas.py` (pydantic) | none: FastAPI derives `/openapi.json` from these models in M1-T4 | unit tests, plus parsing our responses with the official `openai` SDK's own types |
| C2 Telemetry v1 | `src/ullm/telemetry/events.py` (pydantic) | `src/ullm/telemetry/schema/telemetry.v1.json` (JSON Schema), `terminal/src/contracts/telemetry.ts` (TypeScript) | round-trip property tests; `make check` fails if either artifact is stale |
| C3 Runner | `src/ullm/runner/base.py` (`Protocol` plus frozen dataclasses) | none | mypy; conformance tests once runners exist (M1-T1) |

`make contracts` regenerates the artifacts. `make check` re-exports them in memory and fails on any difference, so a schema change cannot merge without its generated files.

## C1: OpenAI-compatible API

**Request** (`POST /v1/chat/completions`), parsed into `ChatCompletionRequest`:

| Field | v0 behaviour |
|---|---|
| `model` | Must equal the served model id, else 404 `model_not_found`. The default served id is the Hugging Face repo id (for example `mlx-community/Qwen3-0.6B-bf16`), so the quantization is always visible; `--served-model-name` sets an alias |
| `messages` | At least one. Roles `system`, `developer` (treated as `system`), `user`, `assistant`. `content` is a string or a list of text parts, joined with a newline. Image or audio parts get 400 `unsupported_value`. `tool` messages arrive with A2 |
| `stream`, `stream_options.include_usage` | `stream_options` without `stream: true` is 400, as in OpenAI |
| `max_tokens`, `max_completion_tokens` | Either sets the output limit. Both, with different values: 400. Neither: the server's `--default-max-tokens`. Prompt plus limit over the context: 400 `context_length_exceeded` |
| `temperature`, `top_p` | OpenAI defaults (1.0 and 1.0) and ranges (0–2, and greater than 0 up to 1). Temperature 0 means greedy |
| `top_k`, `chat_template_kwargs` | Extensions. `top_k` 0 means off. `chat_template_kwargs` is passed to the chat template, for `enable_thinking`. Without it the model's own template default applies; benchmarks, goldens and evals always send `enable_thinking: false` explicitly |
| `stop` | A string or up to 4 non-empty strings |
| `seed` | Integer; seeds the request's own sampler state |

Unknown fields are ignored and logged once per field name, as OpenAI-compatible servers usually do, so a client sending harmless extras such as `metadata` or `user` keeps working. **Known OpenAI fields that would change the output are rejected** when set to a non-default value, with 400 `unsupported_parameter` naming the field: `n > 1`, `logprobs`, `top_logprobs`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `tools` and `tool_choice` (until A2), `response_format` (until A3). Ignoring those silently would return output the client did not ask for.

**Non-streaming reply:** `ChatCompletion` with `id: "chatcmpl-<request id>"`, `object: "chat.completion"`, `created` (Unix seconds), `model`, one choice (`message.role`, `message.content`, `finish_reason` = `stop` or `length`; `tool_calls` arrives with A2), and `usage`.

**Streaming reply:** `ChatCompletionChunk` events, all with the same `id`, in this order: a delta with `role: "assistant"`; one content delta per decoded text chunk; a delta with `finish_reason`; when `include_usage` is set, a chunk with `choices: []` and `usage`; then `data: [DONE]`. **`[DONE]` is sent only after a complete reply.** A failure mid-stream sends `data: {"error": {...}}` and closes without `[DONE]`, so a client can always tell a finished reply from a broken one.

**Usage:** `prompt_tokens`, `completion_tokens`, `total_tokens`, and `prompt_tokens_details.cached_tokens` (always present; 0 until prefix caching lands in M3). All counted by the engine.

**Errors:** `{"error": {"message", "type", "param", "code"}}`. The status code decides how the OpenAI SDK classifies an error; `code` is the machine-readable reason.

| Status | `type` | `code` values |
|---|---|---|
| 400 | `invalid_request_error` | `invalid_value`, `unsupported_parameter`, `unsupported_value`, `context_length_exceeded` |
| 401 (M8) | `authentication_error` | `invalid_api_key` |
| 404 | `invalid_request_error` | `model_not_found` |
| 429 (M8) | `rate_limit_error` | `rate_limit_exceeded`, with `Retry-After` |
| 503 | `server_error` | `queue_full`, `engine_unavailable`, with `Retry-After` |

`GET /v1/models` lists the one served id. `/health` and the load summary belong to C4 (M1-T4).

## C2: Telemetry protocol v1

**Frame**, one per flush (about 15 a second), sent as one SSE `data:` line; idle streams get a `: keepalive` comment so proxies do not close them:

```json
{"v": 1, "replica": "r1", "seq": 412, "t": 61234.5, "dropped": 0, "events": [ ... ]}
```

- `seq` increases by one per frame on a connection, so a reader can see gaps.
- `t` and every event's `t` are **milliseconds since the replica's telemetry epoch** (its start), from the monotonic clock. Small numbers that read well and survive replay. `hello` carries the wall-clock time of the epoch.
- `dropped` counts events the bounded buffer dropped since the previous frame, so the terminal can show that it missed something rather than hide it.

**Events.** Every event has `type` and `t`. Request events carry `rid`, the request id (the API's id, or the router's `X-Request-Id` when present). **Telemetry never contains prompt or output text**, only counts and timings, so recorded runs are safe to publish.

| `type` | Fields (besides `type`, `t`) | From |
|---|---|---|
| `hello` | `replica`, `model`, `source` (`engine`, `sim` or `fixture`), `version`, `commit`, `epoch_unix_ms`, `limits` {`max_num_seqs`, `max_prefill_tokens`, `max_model_len`}, `kv` {`block_size`, `num_blocks`, `capacity_tokens`: each null until known}, `machine` {`chip`, `memory_gb`} or null | M1 |
| `req.arrived` | `rid`, `prompt_tokens`, `max_tokens`, `tenant` (null until M8) | M1 |
| `req.admitted` | `rid`, `queue_ms`, `cached_tokens` | M1 |
| `req.first_token` | `rid`, `ttft_ms` (from arrival) | M1 |
| `req.finished` | `rid`, `reason` (`stop`, `length`, `tool_calls`), `usage` {`prompt_tokens`, `completion_tokens`, `cached_tokens`}, `timings` {`queue_ms`, `ttft_ms`, `tpot_ms` or null, `e2e_ms`} | M1 |
| `req.aborted` | `rid`, `reason` (`client_disconnect`, `shutdown`, `error`), `completion_tokens` | M1 |
| `req.preempted` | `rid`, `completion_tokens`, `blocks_freed` | M3 |
| `step` | `n` (step index), `kind` (`prefill` or `decode`), `rids` (rows in batch order), `tokens` (tokens processed), `ms` (after `mx.eval`; `t` is the step's end), `waiting`, `running`, `kv_tokens` (tokens held in KV) | M1 |
| `kv.snapshot` | `num_blocks`, `owners` (list of `rid`), `states` (base64, one byte per block: 0 free, 1 Cached, 2 shared with ref count above 1, 3 + i owned by `owners[i]`) | M3 |
| `router.decision` | `rid`, `replica`, `kind` (`prefix`, `least_load`, `round_robin`), `matched_messages`, `retry` (bool) | M5 |

`source` keeps the hosted replay honest: the terminal labels fixture and `sim` data as simulated, always. Terminal statistics (tokens per second, TTFT percentiles) are computed from these events in the browser, so there is one source for each number.

**Compatibility rule.** Within v1, only additive changes are allowed: new optional fields and new event types. Readers ignore event types and fields they do not know. Removing, renaming or retyping anything is v2. Producers are strict (pydantic `extra="forbid"`) so a typo fails a test instead of shipping.

**Recorded traces** are gzip-compressed JSONL (`.jsonl.gz`, which browsers can decompress natively), one frame per line, exactly as sent over SSE, starting with a frame that holds `hello`. Live and replay sources in the terminal read the same thing.

**Engine-side cost.** The models define the wire format. Whether the engine thread builds them directly or appends lighter tuples that the API thread converts is decided in M2-T4, from the measured overhead (rule 5).

**Generation.** pydantic exports the JSON Schema in serialization mode with defaults marked required, so `type` is a required literal and TypeScript gets a proper discriminated union. TypeScript is generated with `json-schema-to-typescript`, via pnpm in `terminal/`.

## C3: Runner protocol v0

A sketch of the boundary. Exact signatures are settled in M1-T1 (catalog C3) and recorded then.

```python
SeqId = int  # engine-internal; the core maps request ids to these


@dataclass(frozen=True, slots=True)
class SamplingParams:
    temperature: float
    top_p: float
    top_k: int
    seed: int | None


@dataclass(frozen=True, slots=True)
class PrefillPlan:  # one sequence, input [1, L], no padding (DEC-7)
    seq: SeqId
    token_ids: tuple[int, ...]  # the tokens to run this step
    start: int  # position of token_ids[0]: 0 now; >0 after a prefix hit or a chunk
    sample: bool  # sample after this step (false for a non-final chunk, E13)


@dataclass(frozen=True, slots=True)
class DecodePlan:  # input [B, 1]
    seqs: tuple[SeqId, ...]  # batch order
    token_ids: tuple[int, ...]  # the last token of each sequence, fed back


StepPlan = PrefillPlan | DecodePlan


@dataclass(frozen=True, slots=True)
class StepResult:
    tokens: tuple[int, ...]  # one per sampled row, in plan order (empty if sample is false)
    forward_ms: float  # all timings are taken after mx.eval (rule 4)
    total_ms: float


class Runner(Protocol):
    def info(self) -> RunnerInfo: ...  # model id, vocab size, EOS ids, max context
    def add(self, seq: SeqId, sampling: SamplingParams) -> None: ...  # create per-sequence state
    def execute(self, plan: StepPlan) -> StepResult: ...
    def release(self, seq: SeqId) -> None: ...  # free that sequence's tensors
```

The core owns every decision (which sequences, which tokens, when to release). The runner owns every tensor (weights, KV, logits, per-sequence random state) and samples, because sampling works on `[B, vocab]` arrays on the GPU. The tokenizer and streaming detokenizer reach the core through a separate MLX-free protocol, decided in M1-T1 and M1-T2, because mlx-lm's own detokenizer imports MLX. From M3 the plans also carry block tables.

## New dependencies

- `pydantic==2.14.3` (runtime): the contract models. It is already in the environment through mlx-lm on the Mac, but the core must have it on Linux too.
- `openai` (dev): the contract test parses our responses with the SDK's own types, and M1-T5 needs it for the SDK test anyway.
- `json-schema-to-typescript` and `typescript` (terminal dev): generation and the type check of the generated file.
- Added during implementation: `httpx2` (dev), the HTTP client the `openai` 3.x SDK is built on, imported directly by the in-process SDK test for its `MockTransport`; `jsonschema` and `types-jsonschema` (dev), so a property test can check that every emitted frame validates against the exported schema.

## Alternatives considered

- **Hand-written TypeScript types.** Two definitions drift. Generating from one source, with a staleness check, removes the drift.
- **Protocol Buffers or MessagePack for telemetry.** Smaller frames, but the volume is tiny (about 15 frames a second), and JSON can be read in a browser's network tab and replayed from a text file. Not worth the toolchain.
- **WebSocket for telemetry.** The data flows one way; SSE is plain HTTP and goes through the Go proxy unchanged (DEC-10).
- **A 400 for every unknown request field.** Strict, but it breaks clients that send harmless extras. Rejecting only the known fields that change the output keeps both safety and compatibility.
- **Integer slot ids in telemetry instead of request ids.** Smaller frames, but the terminal and logs need the same id the client and the router see.

## Implementation notes

- Schema rendering lives in `ullm.telemetry.export` so tests import it from the package; `scripts/export_contracts.py` only writes or checks the file.
- Field titles are dropped from the exported schema (model titles stay). Otherwise the TypeScript generator emits one alias per field (`T1`, `Rid4`, ...).
- Times round to microseconds only in JSON (`when_used="json"`); in-process values stay exact. The custom serialiser hides `minimum: 0` from the schema, so it is restated explicitly.
- `terminal/src/contracts/telemetry.typecheck.ts` fails the type check if the generated types stop forming a discriminated union on `type`, or if a new event type is not handled in its exhaustive switch. A new event type therefore needs a deliberate one-line change on the terminal side.
- `tests/unit/test_telemetry_contract.py` holds the list of every string field in the protocol. A new string field fails that test until it is added to the list, which is where the "no prompt or output text" rule is enforced.
- The C1 contract is already exercised through the real `openai` SDK in process (`tests/unit/test_api_schemas_sdk.py`): the SDK's own request body is parsed by our model, and our replies, streams, mid-stream errors and error statuses are parsed by the SDK.
