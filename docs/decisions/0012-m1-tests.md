# 0012: The M1 tests and rule 1's near-tie threshold

**Status:** accepted · **Task:** M1-T5 (Q1, Q3, Q4)

## Context

M1-T5 names three test groups: unit tests (Q1), single-request equivalence (Q3), and the SDK contract and 50 clients (Q4). Rule 8 ships tests with their behaviour, so M1-T1 to M1-T4 already wrote most of them:

| Catalog item | Already on `main` |
|---|---|
| Q1 sampler | `tests/unit/test_sampler.py` |
| Q1 stop check, detokenizer | `tests/property/test_stop_check.py`, `tests/property/test_detokenizer.py`, the real-tokenizer cases in `tests/unit/test_mlx_runner.py` |
| Q1 chat template | `test_render_gives_the_golden_prompt_ids` (`tests/equivalence/test_runner_equivalence.py`), `test_control_tokens_inside_a_message_render_as_text` |
| Q1 SSE framing | `test_a_stream_follows_c1s_order` checks event order and payloads, through a helper that keeps only `data:` lines |
| Q3 one request | `test_runner_equivalence.py`, `test_engine_equivalence.py`, `test_server_equivalence.py`, each against mlx-lm on the same machine in the same session |
| Q4 SDK contract | `tests/integration/test_server_sdk.py`, `tests/unit/test_api_schemas_sdk.py` |
| Q4 50 clients | `tests/integration/test_bridge.py`, but without HTTP |

Four gaps remained:

- No test held 50 clients open at once over HTTP.
- No test read the raw SSE bytes. A server that ended each event with one newline instead of two passed every test in `test_server.py`, whose `sse_events` helper keeps only `data:` lines. Only two SDK tests in `test_server_sdk.py` failed (mutant run, below).
- The engine equivalence test did not report mlx-lm's top-2 margin on a mismatch, as Q3 requires.
- `make equivalence` was a placeholder.

Decision 0006 also left one choice to this task: the documented near-tie threshold of rule 1. Decision 0006's evidence:

- Exact bf16 ties appear at 6 of 495 golden positions.
- On another GPU, mlx-lm diverged from its own goldens only at margins of 0.125 or less.
- Decision 0008 measured that a single `[1, L]` prefill, a real bug against mlx-lm, diverges at margins 0.0 and 0.125.

## Decision

**Near-tie threshold for rule 1.**

- **One request, and many requests that run one at a time (`max_num_seqs = 1`): no allowance.** uLLM and mlx-lm run the same arithmetic on the same machine, so any divergence is a bug, whatever its margin. The tests still report the margin at the first difference, so a failure can be classified without a rerun.
- **mlx-lm against its committed goldens on another chip: 0.25**, as decision 0006 proposed. It is now settled, no longer provisional. All five observed cross-machine divergences were at 0.125 or less. This threshold never applies to uLLM against mlx-lm.
- **Batch sizes above 1 (M2-T3) and later variants** set their own threshold, from a measurement of how their arithmetic moves the logits. Until then, they get none.

**Tests added.**

- `tests/integration/test_server_load.py`: 50 `openai.AsyncOpenAI` clients at once against a real uvicorn server on 127.0.0.1 with `FakeRunner`. Even-numbered clients ask for a plain reply, odd ones stream with usage. Client `i` sends `i + 1` characters, so its `prompt_tokens` identifies it, and `max_tokens` cycles from 1 to 8. Each reply must be the text of exactly its own request's sampled tokens, with exact usage and one completion id. Then: at least 25 requests were open at once (measured: 48 to 50 in 8 runs), every sequence was released, `/health` shows none in flight, and the trace checker finds no problem.
- `tests/serving.py`: the real-uvicorn helper, moved from `test_server_disconnect.py` so both files use it.
- `test_each_sse_event_is_one_data_line_then_a_blank_line` (`tests/integration/test_server.py`): splits the raw body on blank lines. Each event is one `data: ` line with no other line break, the last is `[DONE]`, and every other one parses as a chunk.
- `tests/equivalence/divergence.py`: `divergence()` formats the first differing position and mlx-lm's margin there. The runner and engine equivalence tests use it.
- `make equivalence` downloads the model if needed (`make models`), then runs `pytest -m mlx tests/equivalence` with `ULLM_REQUIRE_MODEL=1`, so a missing model fails instead of skipping.

**Evidence that each new test can fail** (uncommitted mutants, restored after each run):

| Mutant | Result |
|---|---|
| `_Reply._write_json` appends to one buffer shared by every reply | the load test fails: `client 0` got ` t13 t123 t265 t147 t133 t40` instead of ` t40` |
| The load test awaits its clients one at a time | the load test fails: at most 1 request open |
| Two clients' whole replies swap (10 and 26) | the load test fails on `prompt_tokens`: 33 against 17. Each prompt has its own length, because `FakeRunner`'s tokens ignore the prompt |
| Each SSE event ends with `\n` instead of `\n\n` | the framing test fails, and both `test_a_stream_follows_c1s_order` cases still pass. `test_the_sdk_streams_a_reply_with_usage` and `test_the_sdk_sees_503_when_the_engine_is_full` also fail |
| A single `[1, L]` prefill (decision 0008) | runner and engine equivalence fail: `palindrome` at token 20, margin 0.0, and `translate` at token 14, margin 0.125 |

## Alternatives considered

- **A 0.25 allowance for uLLM against mlx-lm.** Rejected. It would pass the `[1, L]` prefill bug, whose divergences are at 0.0 and 0.125.
- **The 50-client test through Starlette's `TestClient`.** Rejected. `TestClient` is synchronous, so it sends one request at a time, and it bypasses uvicorn and the socket.
- **The 50-client test on the real model.** Rejected for this task. On `FakeRunner` it runs in about 1.3 s in the core CI job with no model. What it checks, the routing of each reply to its client, does not depend on the model. Single-request equivalence over HTTP already covers the real model.
- **Margins in the server equivalence test.** Not added. That test compares text, and the engine test already compares token ids through the same engine loop.

## Consequences

- Gate G1's "equivalence and SDK contract tests green in CI" is met by the `mlx` job (equivalence) and the `core` job (SDK contract and 50 clients).
- M2-T3 must measure how batching moves logits and record its threshold before any batched equivalence test allows a divergence.
- `tests/serving.py` is the place for any later test that needs a real socket.
