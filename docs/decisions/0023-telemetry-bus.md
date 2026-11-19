# 0023: Telemetry bus, hub, stream and recorder

**Status:** accepted (the author approved the shape and chose the recommended answer to each question before any logic was written) · **Task:** M2-T4 (O1, C2) · **Answers:** the open item in decision 0004 on time order

## Context

The engine already emits every C2 event through one `Emit` callable (decision 0010). `ullm serve` passes `_drop`; `ullm sim` appends to a list and groups it into frames afterwards with `trace.frames_from`. Nothing streams or records a live run.

Constraints:

- Rule 5: the engine thread never blocks on telemetry. Rule 2: threads talk through queues, no locks in `core`.
- O1: a bounded buffer that drops the oldest on overflow and counts the drops; frames flushed about 15 times a second on `/telemetry/stream`; a closed tab costs nothing; a recorder writes any run to a trace file. Proof: overhead on and off, a schema round-trip, recordings pass `check_trace`.
- C2 and decision 0003: a frame's `seq` counts per connection; `dropped` counts events lost since the previous frame; every connection and every recording starts with `hello`; idle streams get a `: keepalive` comment. A frame's events are no later than the frame's `t` (`check_trace`).
- The terminal's live source (`terminal/src/sources/live.ts`) reads SSE with `fetch`, never reconnects, and expects `hello` first.
- `Engine` keeps its own epoch (`clock()` at construction) and the wall-clock `epoch_unix_ms`. A frame's `t` must be on the same clock, so the epoch cannot stay private to `Engine`.
- `collections.deque` appends and pops are thread-safe in CPython ("Deques support thread-safe, memory efficient appends and pops from either side", Python 3.12 library docs, `collections.deque`). `maxlen` discards from the other end on overflow, silently.

## Decision

### Shape

```
ENGINE THREAD                      API THREAD (asyncio loop)
Engine ──emit──▶ Bus ──drain, 15/s──▶ Hub ──put──▶ Subscription ─▶ /telemetry/stream (one per tab)
       (deque, maxlen,              (one task)       (deque, maxlen)  ─▶ record() ─▶ .jsonl.gz
        index per event)
```

1. **`TelemetryClock`** (`ullm/telemetry/bus.py`): a frozen pair of a monotonic epoch and its `epoch_unix_ms`, with `now_ms()`. `Engine` takes `clock: TelemetryClock | None` in place of today's `clock: Callable[[], float]` (no caller passes it). `EngineClient.start` gains `clock: TelemetryClock | None = None` and passes it on. One clock means `frame.t` and every event's `t` share an epoch.
2. **`Bus`**: the engine thread's sink. `emit(event)` appends `(index, event)` to a `deque(maxlen=capacity)` and increments `index`. Single producer: only the engine thread calls `emit`. `drain() -> (events, dropped)` pops everything present; `dropped` is the gap between the index it expected and the first index it popped, so the count is exact. Single consumer: only the `Hub` calls `drain`. Never blocks, takes no lock. Capacity 4,096 events (a guess, sized for a stalled loop of a few seconds at batch 16; `dropped` shows when it is wrong).
3. **`Hub`**: owns the subscriptions, on the loop thread only. `run()` is one task: every `FRAME_MS` it drains the bus, keeps the latest `hello`, and puts the events and the upstream `dropped` into every subscription. `subscribe()` returns a `Subscription` whose first frame starts with the kept `hello`. `close()` drains once more and ends every subscription, so a recording after `engine.close()` is complete.
4. **`Subscription`**: an async iterator of `Frame`, on the loop thread only. Its own bounded `deque` (4,096 events) drops the oldest when its reader is slow, and adds those to its `dropped`. It builds a frame when it is read: its own `seq` from 0, `t = clock.now_ms()`, `dropped` since its previous frame, every pending event. Leaving the iterator unsubscribes, so a closed tab costs nothing after its disconnect.
5. **`/telemetry/stream`** (`server/app.py`): SSE, one `data:` line per frame, and `: keepalive` after 10 s without a frame. It watches for the disconnect as `_Reply` does. With telemetry off it answers 404.
6. **Recorder**: `record(subscription, path)` (`ullm/telemetry/trace.py`) writes each frame as one line of the gzip trace format, with `mtime=0`. `ullm serve --record PATH` subscribes it before the engine starts, so the recording begins with `hello` and closes in the lifespan's exit. `ullm sim` runs on the same `Bus`, `Hub` and `Subscription` instead of `frames_from`, so `make sim` checks the real path. `frames_from` stays for tests that check a list of events the engine emitted on the test thread.
7. **`create_app(..., emit=None)`**: `None` means the app's own `Bus` and `Hub`, stream on. Tests that pass `events.append` keep working and get no stream. `ullm serve --telemetry off` passes a drop sink: today's behaviour, the baseline for the overhead.
8. **Time order.** The engine thread is the one producer, on one monotonic clock, and the bus keeps emit order. Event times therefore never go backwards in stream order, and `check_trace` keeps its strict rule. This closes decision 0004's open item.
9. **`kv.snapshot`** waits for blocks (M3-T4, M3-T8). Nothing here emits it.

### Overhead (the O1 proof)

A paired session (decision 0016) at concurrency 8, `--max-num-seqs 8`, servers per chunk: baseline `ullm-off` (`--telemetry off`), its twin, and `ullm-on` (`--record {out}/ullm-on.jsonl.gz`). A recorder serialises every frame as a terminal tab does, then compresses it, so it costs at least one tab. `bench.session` gains an `{out}` placeholder for the session directory. The result is the `ullm-on` / `ullm-off` ratio on output tokens per second and TPOT, with the twin as the noise floor, committed under `bench/results/m2/telemetry/`, with a `docs/CLAIMS.md` row. The agent builds the group and runs a smoke session; the author runs the full session (AC power, idle Mac, `sudo` for `powermetrics`), and the agent analyses and commits it in this slice. The `ullm-on` recording is committed beside it when it is under 5 MB.

"Off" still builds the event objects, as M1 did; the measurement covers the bus, the hub, serialisation and compression, not event construction.

### As built

- uvicorn waits for every open response before the lifespan's exit (`uvicorn/server.py`, `Server.shutdown` and `_wait_tasks_to_complete`, 0.54.0), and a telemetry stream never ends by itself. An open tab kept `ullm serve` from stopping on Ctrl+C, and a second Ctrl+C skips the lifespan's exit, so the recording lost its trailer. `ullm serve` now passes `timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_S` (5 s): past it uvicorn cancels open replies, then the lifespan closes the hub and the recording. An open chat reply is cut the same way after 5 s. `tests/integration/test_serve.py` checks it through the real process.
- `ullm serve` opens the recording before it reports ready, so a path it cannot write fails startup instead of losing the run's telemetry; `trace.record` takes the open file.
- The app keeps its hub on `app.state.hub`, so a test can see that a closed tab left (`Hub.readers`).
- `ServeConfig` gains `record` and `telemetry_keepalive_s`; `ullm serve` gains `--telemetry on|off` and `--record PATH`, and refuses both `--record` and `--telemetry off`.
- `bench/groups/m2-telemetry.json` and `make bench-telemetry` run the session. Servers start per chunk, so each chunk's `ullm-on` overwrites `{out}/ullm-on.jsonl.gz`: the kept recording is the last chunk's.

## Alternatives considered

- **Events ride `deliver`.** The engine buffers a turn's events and hands them over with the outputs; no shared structure at all. Lost because the bound would sit after the loop's unbounded `call_soon_threadsafe` queue, so a stalled loop grows memory, and events emitted on the engine's death path are never delivered.
- **One engine-side buffer per subscription.** No hub task, and no work at all with no reader. Lost because subscribing from the loop thread races the engine's `hello` and index reads; the fix is ordering arguments in comments, and engine cost grows with tabs.
- **Empty frames 15 times a second when idle.** Simpler reader, but decision 0003 already chose a keepalive comment.

## Consequences

- `core` changes by one parameter type in `Engine` and one in `EngineClient.start`. Telemetry code stays outside `core`.
- A process killed without its lifespan exit leaves a truncated gzip recording. `iter_frames` will raise at the end of it.
- A tab that joins mid-run sees steps for requests it never saw arrive. Recordings do not, because they subscribe first.
