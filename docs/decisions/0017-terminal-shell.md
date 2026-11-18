# 0017: The terminal shell, store and sources

**Status:** accepted · **Task:** M1-T8 (T1, part of T2)

## Context

`terminal/` held only the generated C2 types. T1 asks for the shell, an external store with ring buffers that keeps telemetry out of React state per event, and a `TelemetrySource` with a live SSE and a replay implementation. T2's first two panels are the stats ticker and the request feed. The engine does not serve `/telemetry/stream` until M2-T4, so the terminal is built against the fixtures (decision 0004) and `ullm sim --trace` recordings.

Facts this design rests on:

- `vite preview` and `vite dev` serve `.gz` files with `Content-Encoding: gzip`, so the browser receives the trace already decompressed (`curl -sI` on the preview server, vite 8.3.5). A hosted build on another server may not do that.
- Node 24 has `fetch`, `DecompressionStream` and `TextDecoderStream` as globals, but no `EventSource` without `--experimental-eventsource`.
- `ullm.server.app` already allows the terminal's origins through CORS (`TERMINAL_ORIGINS`), so a live source on port 5173 can read port 8000.
- Headless Chrome with `--virtual-time-budget` and `--dump-dom` showed only the first render of the app on macOS 27, with `CVDisplayLink` errors in its log; the cause is probably `requestAnimationFrame` not firing there (not proven). A real-time headless session over the DevTools protocol shows the app updating.

## Decision

**Layout.**

| File | Owns |
|---|---|
| `src/source.ts` | `TelemetrySource { label; start(sink): stop }` and `Sink { frame; status }` |
| `src/sources/replay.ts` | `replaySource(label, load, { speed })`: frame `i` is sent at `(t_i − t_0) / speed` after the start, from one start time, so timer lateness never accumulates |
| `src/sources/live.ts` | `liveSource(url)`: `fetch` plus a WHATWG event-stream parser (`SseParser`). No reconnect |
| `src/trace.ts` | `parseFrame` (rejects any `v` other than 1), `parseTrace` (gzip or plain JSONL, told apart by the gzip magic bytes), `loadTrace` |
| `src/ring.ts` | `Ring<T>`, fixed capacity, overwrites the oldest |
| `src/store.ts` | `TelemetryStore`: request rows, step and latency samples, seq-gap and drop counters, the source status |
| `src/stats.ts` | `tickerStats(store)`: the ticker's figures |
| `src/config.ts` | `?live=<url>`, `?trace=<url>` (default `telemetry/steady-chat.jsonl.gz`), `?speed=` |
| `src/ui/` | `App`, `Header`, `Ticker`, `Feed`, `tokens.css`, `app.css` |

**Store.** Frames go into the store, never into React state. Each `ingest` bumps `version`; subscribers hear at most once per `schedule` callback, which is `requestAnimationFrame` in the browser. React reads the store through `useSyncExternalStore(subscribe, getVersion)` and then reads the store's fields directly. A frame whose first event is `hello` resets the store: C2 sends `hello` first on every connection and `seq` restarts with it. A jump in `seq` adds the missing count to `missedFrames`. `dropped` adds to `droppedEvents`. The header shows both when they are not zero. Events for a request the store never saw arrive (a live connection opened mid-request) are ignored, and event types it does not know are ignored (C2 compatibility rule).

**Request states.** `req.arrived` → waiting, `req.admitted` → prefilling, a step that includes the request → running, `req.preempted` → preempted, then finished or aborted. Running starts at the prefill step, as in `ullm.telemetry.check`, so a preempted request that is prefilled again becomes running without a second first token. Tokens count one per step row, the checker's rule, and take the engine's `usage.completion_tokens` (or `completion_tokens` on abort) when the request ends.

**Feed retention.** Every request in flight stays. At most 200 ended requests stay, the oldest removed first (the T7 budget names 200 table rows). The feed shows the newest 200 arrivals, newest first.

**Ticker.** All windows end at the store's trace time, the `t` of the newest frame, so a replay at any speed shows the same figures as at speed 1.

| Figure | Definition |
|---|---|
| tok/s | Tokens sampled by steps with `t` in the last 1,000 ms (one per row), per second |
| TTFT p50, p99 | Nearest-rank percentiles of `req.first_token.ttft_ms` in the last 60 s, with the count |
| TPOT p50 | Nearest rank of non-null `req.finished.timings.tpot_ms` in the last 60 s, with the count |
| batch | The newest step's `running`, over `hello.limits.max_num_seqs` |
| queue | The newest step's `waiting` |
| KV | The newest step's `kv_tokens`, as a share of `hello.kv.capacity_tokens` when it is known |

These figures are for watching. They are never a claim: claims come from `bench/` results (rule 8).

**Simulated label.** The header shows a SIMULATED badge whenever `hello.source` is not `engine` (decision 0003).

**Checks.** `make check-ts`, and so `make ci-terminal` and the CI terminal job, now also run `vitest run` and `vite build`. The UI tests render with `react-dom/server` under Node, so no DOM library is needed. A browser test arrives with Playwright in M2-T10.

## Alternatives considered

- **`EventSource` for the live source.** It reconnects by itself, but Node cannot run it without a flag, so the live path could only be tested in a browser. Reconnecting also needs care: a new connection brings a new `hello` and a new `seq`, which the store already handles. Reconnect can be added on top of `fetch` when the engine serves the stream (M2-T4).
- **React state or a reducer per frame.** About 15 frames a second, each with up to dozens of events, would re-render on every frame and copy state on every event. T1 forbids it.
- **Wall-clock windows for the ticker.** They would make a replay at speed 4 show four times the tokens per second.
- **A time window for the feed instead of a row cap.** A row cap bounds memory and DOM size directly, which the T7 budget is about.
- **Copying the fixtures into `public/`.** `publicDir: "fixtures"` serves them where they already are, with one copy in the repo.

## Consequences

- M2-T7 adds the step chart and batch strip as canvases that read the same store on `requestAnimationFrame`, and the requests table reads `store.request()` and the feed rows.
- M2-T9 adds play, pause, speed and a scrubber to `replaySource`; the run picker replaces `?trace=`.
- M2-T4 must prove `liveSource` against the real `/telemetry/stream`, including the `: keepalive` comment and CORS.
- New dependencies, pinned exactly: `react` and `react-dom` 19.3.1 (the UI), `vite` 8.3.5 and `@vitejs/plugin-react` 6.1.2 (dev server and build), `vitest` 5.0.5 (tests), `@types/react` and `@types/react-dom` 19.3.0 (types). Each was released at least two weeks before it was pinned.
