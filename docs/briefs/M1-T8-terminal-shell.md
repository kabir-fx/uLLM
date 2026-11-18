# M1-T8: The terminal shell, ticker and feed

**State:** DONE at `61c8a1e`. Type `/ship` to open the pull request.
**Decision record:** [0017](../decisions/0017-terminal-shell.md) (the full design, for agents) · **Spec:** roadmap M1-T8 (T1, part of T2)

## The goal

You open the terminal in a browser and you see the engine work. A stats ticker and a request feed update from a recorded trace or from a live stream.

## The problem

- `terminal/` holds only the generated telemetry types. It has no app.
- The engine does not serve `/telemetry/stream` yet. That arrives in M2-T4. The terminal must work now from the fixtures.
- Telemetry must never pass through React state for each event (T1). React would render about 1,000 times a second.
- Fixture and `sim` data must show a SIMULATED label wherever the terminal shows it (decision 0003).
- Lane B may not change a contract (roadmap 3.5). The terminal reads C2 as it is.

## How it works

```
 trace file (.jsonl.gz) ──▶ replaySource ─┐
                                           ├─▶ TelemetrySource ─▶ TelemetryStore ─▶ ticker, feed
 /telemetry/stream (SSE) ─▶ liveSource ───┘     (one interface)    (ring buffers)    (React, once per frame)
```

1. **`TelemetrySource`** (`src/source.ts`). Both sources give frames to the same sink. The rest of the app does not know which source runs.
2. **`replaySource`** (`src/sources/replay.ts`). It reads a trace file and sends each frame at its own time `t`.
3. **`liveSource`** (`src/sources/live.ts`). It reads SSE from a URL with `fetch` and sends each `data:` line as a frame.
4. **`TelemetryStore`** (`src/store.ts`). It applies events to request rows and ring buffers. It tells React to render not more than once in each animation frame.
5. **Ticker and feed** (`src/ui/`). The ticker shows tokens per second, TTFT, TPOT, batch size, queue depth and KV use. The feed shows the newest 200 requests with a state pill.

## What you get at the end

- `pnpm -C terminal dev` opens the terminal on the `steady-chat` fixture at `http://127.0.0.1:5173`.
- `?trace=<url>` replays another trace. `?live=<url>` reads a live stream.
- `make check` also runs the terminal tests and the production build.

What you do not get yet:

- The step chart, batch strip and requests table. They arrive in M2-T7.
- Play, pause, speed and a scrubber. They arrive in M2-T9.
- A live stream from the engine. It arrives in M2-T4.

## Where this puts the project

```
M0  █████████
M1  ███████▓
M2  ░░░░░░░░░░░░░
```

This slice opens M2-T7, M2-T8 and M2-T9 for Lane B, and then gate G1.

## Choices the agent made

You can still change these before `/ship`. Decision 0017 gives the reasons.

- The ticker uses trace time, not the clock of your computer. A replay at four times speed then shows the same figures.
- Tokens per second counts sampled tokens in the last second. TTFT and TPOT use the requests of the last 60 seconds.
- The live source uses `fetch` and not `EventSource`. Node can then test it against a real SSE server.
- A `hello` event starts a new session in the store. The contract sends `hello` first on every connection.

## What can go wrong

- The live source has no test against the real engine. M2-T4 must prove it.
- The live source does not reconnect after an error. The header shows the error.
- With few requests in the window, p99 is the slowest request. The ticker shows the count beside each percentile.

## Words

| Word | Meaning |
|---|---|
| frame | One flush of telemetry: one SSE `data:` line, or one line of a trace file. |
| trace time | The `t` of the newest frame, in milliseconds from the start of the replica. |
| ring buffer | An array of fixed size that writes over its oldest item when it is full. |

## Learn from this slice

**An external store.** React does not hold the telemetry. The store holds it and counts a version. React reads the store once in each animation frame through `useSyncExternalStore`.

**One interface for two sources.** The replay source and the live source give the same frames to the same sink. A test can drive the whole terminal from a file, with no engine.

Test yourself: why does a replay at speed 4 show the same TTFT as at speed 1?

<details><summary>Answer</summary>
The ticker uses the `t` of each event, and the replay does not change `t`. Only the delay between frames changes.
</details>

## Result

**State:** DONE. The proof is VERIFIED at `c851227`. The review is READY at `c851227`, with the fix commits up to `61c8a1e`. `make ci` passes at `61c8a1e`. This change used 2 agent runs.

**Against the plan:** the build followed the plan. The UI tests render with `react-dom/server` under Node, so the change adds no DOM library.

**Bugs that the review found:**

- A frame without `events` froze the replay, and the header still showed "open". Now `parseFrame` rejects that frame. The replay also stops with an error status when the store throws.
- No test checked that the store keeps only 200 ended rows. A new test fails when the eviction line goes.
- Data before a `hello` showed no SIMULATED badge. Now the badge shows until a `hello` says `engine`.

**What the verifier saw:** the agent drove the built app in headless Chrome. Both fixtures ended with the counts of the trace. A `ullm sim` trace showed 50 rows, with 38 finished and 12 aborted. A local SSE server gave the same result as the replay. At speed 1 and speed 8, the final TTFT and TPOT were the same.

**What is not proven:**

- The live source against the real engine. M2-T4 brings `/telemetry/stream`.
- The lower edge of the 60 s window in a browser. The fixtures are shorter than 60 s.

**Read these yourself:**

- `terminal/src/store.ts:61-72`: the reset at `hello` and the count of missed frames.
- `terminal/src/sources/replay.ts:24-45`: the tick loop and its error path.
- `terminal/src/sources/live.ts:41-60`: the SSE parser and the CR LF case.
