# M1-T5: The M1 tests

**State:** DONE at `a6e0ad5`. You can now type `/ship`.
**Decision record:** [0012](../decisions/0012-m1-tests.md) (the full design, for agents) · **Spec:** roadmap M1-T5 (Q1, Q3, Q4)

## The goal

This slice completes the tests that gate G1 needs. You get one command for equivalence, and proof that 50 SDK clients share the server correctly.

## The problem

- Rule 8 says that tests ship with behaviour. So M1-T1 to M1-T4 already wrote most of Q1, Q3 and Q4.
- No test sends 50 clients at the same time over HTTP. `test_bridge.py` sends 50 clients, but with no HTTP server.
- No test reads the raw SSE bytes. The helper `sse_events` drops lines that are not `data:`, so a bad frame passes `test_server.py`. Only two SDK tests catch it.
- Q3 says that a mismatch must show the top-2 margin. The engine equivalence test does not show it.
- `make equivalence` is a stub that exits 2.
- Rule 1 says that the equivalence tests must never become less strict.
- Decision 0006 tells this slice to set the near-tie threshold of rule 1.

## How it works

1. **`tests/serving.py`.** The agent moves the `serving` helper here from `test_server_disconnect.py`. It starts a real uvicorn server on 127.0.0.1 on a free port.
2. **`tests/integration/test_server_load.py`.** Fifty `AsyncOpenAI` clients send requests at the same time. Each client sends a prompt of a different length, so its usage identifies it. Half of the clients stream. Each reply must be the text of its own tokens, with exact usage.
3. **The SSE framing test** (`test_server.py`). It reads the raw body. Each event must be one `data:` line and one blank line.
4. **The engine equivalence test.** At the first different token, the failure message shows the mlx-lm top-2 margin.
5. **The near-tie threshold.** uLLM against mlx-lm on one machine gets no allowance. Any different token fails. Only mlx-lm against goldens from another chip keeps the 0.25 allowance.
6. **`make equivalence`.** It downloads the model if necessary. Then it runs `tests/equivalence` with `ULLM_REQUIRE_MODEL=1`, so a missing model fails.

## What you get at the end

- `make equivalence` runs the four equivalence files on the real model.
- `uv run pytest tests/integration/test_server_load.py` proves that 50 concurrent SDK clients each get their own reply.
- `make ci-core` and the core CI job run the new SSE framing test and the load test with no model.

What you do not get yet: equivalence at batch sizes above 1 (M2-T3).

## Where this puts the project

```
M0  █████████
M1  ████▓░░░
```

The next slices are M1-T6 (the benchmark harness) and M1-T8 (the terminal shell).

## Choices the agent made

You can change these before `/ship`.

- uLLM against mlx-lm gets no near-tie allowance. A 0.25 allowance will pass the `[1, L]` prefill bug from decision 0008. That bug diverges at margins 0.0 and 0.125. M2-T3 will measure batches and set their threshold.
- The load test uses a real socket and the official SDK, not the in-process `TestClient`. A `TestClient` sends one request at a time.
- The load test uses `FakeRunner`, so it runs in the core CI job with no model.
- `make equivalence` depends on `make models`, as `ci-mlx` does.

## What can go wrong

- The load test runs 50 requests one after the other in the engine, because `max_num_seqs` is 1. It takes about 1.3 s on `FakeRunner`.
- A real-model load test is not part of this slice.

## Words

| Word | Meaning |
|---|---|
| SSE framing | How the server writes each stream event: `data: <payload>`, then a blank line. |
| top-2 margin | The gap between the two largest log probabilities at one position. A small gap is a near-tie. |

## Learn from this slice

A test helper that filters its input can hide the bug that the test must find. The framing test reads bytes, so it sees each byte the server sends.

Fifty clients over a real socket test more than the engine. They also test uvicorn, the disconnect watcher and the SDK parser at the same time.

Test yourself: why can a `TestClient` not prove that 50 clients run at the same time?

<details><summary>Answer</summary>

`TestClient` is a synchronous client. Each call waits for its reply before the next call starts.

</details>

## Result

**DONE.** The proof is VERIFIED and the review is READY at `9d9fb41`. The fix commits up to `a6e0ad5` are covered. One reviewer agent did the work. `make ci` passed at `a6e0ad5`: 344 core tests and 93 `mlx` tests.

Changes against the plan:

- Decision 0012 also sets the near-tie threshold of rule 1. Decision 0006 gave this slice that job.
- A new unit test checks `divergence()`. An off-by-one mutant makes it fail.

The review found three bugs, and the agent fixed each one:

- Two clients can swap whole replies, and the load test did not see it. Now each client sends a prompt of a different length. The swap fails on `prompt_tokens`, 33 against 17.
- `serving` did not stop its server when a test body failed. The `ullm-engine` thread stayed alive. Now a `finally` stops it.
- Two documentation sentences were false. One said that `make check` runs the new tests. The other said that no test caught the SSE framing mutant. Two SDK tests did catch it.

Not proven: the load test on a slow CI runner. The reviewer ran it 20 times with 20 CPU burners. It passed each time, in 2.3 s to 2.7 s.

For later slices: M2-T3 must measure how batches change the logits. Then it can set a threshold for batches.

Read these yourself:

- `tests/integration/test_server_load.py:112`: does each assertion tie a reply to its own client?
- `docs/decisions/0012-m1-tests.md`, the section "Near-tie threshold for rule 1": do you agree that one machine gets no allowance?
- `Makefile:87`: `make equivalence` downloads the model if necessary.
