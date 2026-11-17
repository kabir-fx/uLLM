# M1-T4: The API server

**State:** DONE at `d125bc0`. You can now type `/ship`.
**Decision record:** [0011](../decisions/0011-api-server.md) (the full design, for agents) · **Spec:** roadmap M1-T4 (A1, C1, C4)

## The goal

This slice lets any OpenAI client talk to uLLM over HTTP. You change only `base_url`, and `curl` streams text from the real model.

## The problem

- The engine runs, but only Python code in the same process can call it. No HTTP server exists.
- C1 fixes the request, the reply, the stream order and the error format. C4 fixes `/health` and the 503 replies.
- A client that leaves must stop its request. Otherwise the model works for nobody until `max_tokens`.
- A client must not stop the request of another client. Request ids are easy to guess.
- The server binds to 127.0.0.1 only (rule 14).

## How it works

```
client (curl, OpenAI SDK)
   |  POST /v1/chat/completions
   v
create_app (FastAPI on uvicorn, the event loop)
   1. parse and check the body      -> 400 or 404 on a bad request
   2. render the messages to tokens
   3. EngineClient.submit           -> 503 when full or stopped
   4. _Reply: stream SSE or one JSON, and watch for the client to leave
   |                                     ^
   v  tickets                            |  outputs
Engine thread (decision 0010, no change)
```

1. **`create_app`** (`server/app.py`). It makes the FastAPI app. At start, it starts the engine thread. At stop, it closes the engine.
2. **The chat route.** It checks the request in a fixed order. Each problem gives an error in the OpenAI format. It never puts prompt text in an error.
3. **`_Reply`.** It sends the reply and watches for a disconnect at the same time. If the client leaves early, the reply ends, and the stream sends an abort. This works for streams and for plain JSON replies.
4. **`/health`.** It shows if the engine thread is alive, and how many requests are open, waiting and running. It never waits for the engine.
5. **`FakeTokenizer`** (`runner/fake_runner.py`). It changes text to tokens with no model. With it, `ullm serve --fake` runs on any machine.
6. **`ullm serve`** (`cli.py`). It loads the model, then starts the server on 127.0.0.1.

**Facts that the agent read or measured first.**

- uvicorn tells the app about a disconnect only when the app asks for the next message. FastAPI's normal JSON reply never asks. Thus `_Reply` asks.
- On your M4, the agent rendered a message with fake control tokens. With the escape option, the tokens became text. The 10 golden prompts gave the same tokens with and without the option.
- Rendering 8,000 tokens took about 8 ms. Thus the server renders on the event loop, with no helper thread.

## What you get at the end

- `ullm serve` starts the server with the real model. `curl -N` streams the reply.
- `ullm serve --fake` starts the same server with the fake model, for the terminal and for tests.
- New tests:
  - every reply shape, the stream order, and every error code,
  - the OpenAI SDK against the server, with a stream and without one,
  - a client that leaves frees its request, with a real uvicorn server,
  - the real model over HTTP gives the same text as mlx-lm.

What you do not get yet:

- No `/metrics`. It comes in M2-T5.
- No `/telemetry/stream`. It comes in M2-T4.
- No 50 HTTP clients test. It comes in M1-T5.

## Where this puts the project

```
     done █   this slice ▓   not started ░

M0   █████████       9 of 9
M1   ███▓░░░░        3 of 8, then 4 of 8 after this slice
M2   ░░░░░░░░░░░░░   0 of 13    <- Gate A, tag v0.1: 13 of 30 tasks after this slice
M3   ░░░░░░░░░       0 of 9
M4   ░░░             0 of 3
M5   ░░░░░░          0 of 6
M6   ░░░░░           0 of 5
M7   ░░░░░           0 of 5     <- Gate B, tag v1.0
M8   ░░░░░░          0 of 6
M9   ░░░             0 of 3     All: 13 of 67 tasks after this slice
```

This slice opens the next tasks:

- M1-T5 tests 50 HTTP clients against `ullm serve --fake`.
- M1-T6 and M1-T7 measure uLLM over HTTP, the same way as `mlx_lm.server`.

## What you approve

You approved the first two choices. The other choices are open for review.

1. **Control tokens in a message become text.** The other option was to keep them as control tokens, as `mlx_lm.server` does. Then a user can write a fake system turn.
2. **`chat_template_kwargs` accepts only `enable_thinking`.** The other option was to accept all keys. Then a key such as `tools` gets past the refusal of tools.
3. **The rid stays internal.** The server echoes `X-Request-Id`, but it never uses it as the rid. A repeated id stops the engine. This changes decision 0003.
4. **Each stop string has 256 characters or fewer.** The engine holds back text up to the length of the longest stop string.
5. **The default `max_tokens` is 512.** This is the default of `mlx_lm.server`, so the baselines match.

## What can go wrong

- A render of a very long prompt blocks the event loop for some milliseconds. The agent measured 8 ms for 8,000 tokens.
- An unexpected error gives a plain 500 with no OpenAI body. C1 has no code for it.

## Words

| Word | Meaning |
|---|---|
| SSE | Server-sent events: lines that start with `data:`, sent over one HTTP reply. |
| Lifespan | The FastAPI code that runs at server start and stop. |
| Disconnect | The client closes the connection before the reply ends. |
| Control token | A special token that the chat template writes, such as `<|im_end|>`. |

## Learn from this slice

1. **Why the server parses the raw body.** FastAPI gives its own 422 error format. Our validators carry the OpenAI error code. Thus the route parses the body itself, and one function makes every 400.
2. **Why the reply owns the stream.** The `async with` block sends the abort when the reply ends early, for any reason. Thus no path can forget the abort.

**Test yourself.** A client sends a plain JSON request and closes the connection during a long prefill. What stops the work?

<details><summary>Answer</summary>

`_Reply` waits for the next message from uvicorn in a second task. uvicorn returns `http.disconnect`. The task cancels the reply, the `async with` block ends, and the stream sends an `Abort` ticket. The engine frees the sequence at its next turn.

</details>

## Result

**State.** DONE. The proof is VERIFIED at `1c55ab8`. The review is READY and covers the fix commits to `d125bc0`. `make ci` passed at `03f4ae3`, and the last commit changes only comments and the decision record.

**Changes against the plan.**

- The server renders the prompt on a worker thread, not on the event loop. A full context of 40,960 tokens takes about 115 ms to render.
- The server refuses a `Host` other than `127.0.0.1` or `localhost`. Tests must use a local base URL, and the router must send a local `Host`.
- The server refuses a body that is not JSON. A page on any web site can send `text/plain` with no CORS check.
- The server refuses a body over a limit before it reads the JSON. The limit comes from the costliest token in the vocabulary. On Qwen3 it is 5.96 MB.
- The server logs at most 32 unknown field names.
- The `Tokenizer` protocol has a new method, `token_bytes`.

**Bugs that proof and review found.** The agent fixed each one with a test first.

1. A page on any web site could start a generation with a `text/plain` POST.
2. A page that used DNS rebinding could read the replies.
3. A very large body stopped the event loop for 737 ms.
4. A flood of unknown fields wrote one log line for each field.
5. A flat limit of 32 bytes for each token refused some legal prompts.

**Limits you accepted.** The body limit is not exact. It can refuse a legal request with many empty content parts. It can also refuse text that the template drops, such as old `<think>` text.

**Not proven.** No real browser was used. The agent sent the browser headers with a script.

**For later slices.**

- A prompt of 29,928 tokens took 46.9 s to prefill on the real model. Look at this in M1-T6 or M2.
- FastAPI turns on its own request telemetry. O4 must use it or turn it off.
