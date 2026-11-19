# M2-T4: Telemetry bus, stream, recorder, measured overhead

**State:** NOT DONE. You run `make bench-telemetry`, then the agent records the overhead.
**Decision record:** [0023](../decisions/0023-telemetry-bus.md) (the full design, for agents) · **Spec:** roadmap O1 and C2, task M2-T4

## The goal

The terminal will show a live engine through `/telemetry/stream`. You will also record any run to a trace file and replay it later.

## The problem

- The engine emits every C2 event now, but `ullm serve` drops them. No tab can see a live run.
- `ullm sim` groups its events into frames after the run. Thus no code path frames events while the engine runs.
- Rule 5: the engine thread never waits for telemetry. Rule 2: threads talk through queues, and `core` has no locks.
- O1 asks for a buffer with a limit. It drops the oldest events and counts them.
- O1 also asks for proof: the cost of telemetry, measured on and off.

## How it works

```
ENGINE THREAD                      API THREAD (asyncio loop)
Engine ──emit──▶ Bus ──drain, 15/s──▶ Hub ──put──▶ Subscription ─▶ /telemetry/stream (one per tab)
                                                                ─▶ record() ─▶ .jsonl.gz
```

1. `TelemetryClock`, in `src/ullm/telemetry/bus.py`, gives one time base. The engine and the frames use it, so their times agree.
2. `Bus` is the buffer of the engine thread. `emit` adds an event with a number. It never waits. When the buffer is full, the oldest event goes. The gap in the numbers gives the exact drop count.
3. `Hub` runs as one task on the API thread. About 15 times a second, it empties the bus and gives the events to each subscription.
4. `Subscription` is one reader: a tab or the recorder. It has its own buffer with a limit. A slow tab loses its own oldest events, and other readers do not. Each new reader gets `hello` first.
5. `/telemetry/stream`, in `src/ullm/server/app.py`, sends one SSE frame per read. When a tab closes, its subscription goes.
6. `record`, in `src/ullm/telemetry/trace.py`, writes frames to a gzip trace file. `ullm serve --record PATH` uses it. `ullm sim` uses the same path, so `make sim` tests the real bus.
7. `ullm serve --telemetry off` drops every event, as M1 did. This is the base for the cost measurement.

## What you get at the end

- `ullm serve --fake` and `curl -N http://127.0.0.1:8000/telemetry/stream` show live frames.
- The terminal live source can connect to a real server.
- `ullm serve --record run.jsonl.gz` writes a trace that `check_trace` accepts.
- A committed result gives the cost of telemetry on output tokens per second and TPOT, with a `docs/CLAIMS.md` row.
- New tests prove these things:
  - the bus drops the oldest events and counts each drop exactly
  - `emit` never waits, also when the bus is full
  - each reader gets `hello` first and its own frame numbers
  - a slow reader loses only its own events
  - stream frames match the JSON Schema and parse back
  - recordings from `serve` and `sim` pass `check_trace`
  - a closed tab removes its subscription

**What you do not get yet:** `kv.snapshot` events come with blocks in M3-T4 and M3-T8. Prometheus metrics come in M2-T5.

## Where this puts the project

```
M2  T1 █  T2 █  T3 █  T4 ▓  T5 ░  T6 ░  T7 █  T8 ░  T9 ░  T10 ░  T11 ░  T12 ░  T13 ░
```

This slice opens M2-T9 (replay of real runs) and the live mode of M2-T8.

## What you approve

1. **A hub on the API thread, not one engine buffer per tab.** The other design needs no hub task. But each new tab then races the engine thread, and the engine work grows with each tab.
2. **One shared clock.** `Engine` takes a `TelemetryClock` in place of its private epoch. Without it, frame times and event times can disagree.
3. **A keepalive comment after 10 s of silence, not empty frames.** Decision 0003 already chose the comment.
4. **The recorder is the "on" case of the measurement.** It does the same JSON work as one tab, and it also compresses. Thus the result is an upper bound for one tab.

Questions, each with the recommended answer:

- **Q1. Who runs the paired session for the cost measurement?** It needs AC power, an idle Mac and `sudo` for `powermetrics`. Recommended: the agent builds the group and runs a smoke session. Then you run the full session, and the agent commits and analyses it in this slice.
- **Q2. Do we commit the `ullm-on` recording from that session?** Recommended: yes, if it is below 5 MB. It gives M2-T9 a real trace to replay.
- **Q3. Buffer sizes of 4,096 events for the bus and for each reader?** This number is a guess, not a measurement. Recommended: yes, because `dropped` will show it when it is wrong.

## What can go wrong

- If the API thread stops for a long time, the bus drops events. The terminal sees `dropped` above 0 and shows a gap.
- If you kill the server without a clean exit, the recording ends without its gzip trailer. `iter_frames` will raise at the end.
- A tab that joins during a run sees steps for requests that arrived earlier. Recordings start before the engine, so they are complete.
- The "off" case still builds event objects, as M1 did. Thus the result does not include the cost to build events.

## Words

| Word | Meaning |
|---|---|
| bus | The bounded buffer that the engine thread writes events into. |
| hub | The task on the API thread that moves events from the bus to each subscription. |
| subscription | One reader of the events, with its own buffer and its own frame numbers. |
| recorder | A subscription that writes frames to a `.jsonl.gz` trace file. |
| overhead | The change in throughput and TPOT when telemetry is on, against off. |

## Learn from this slice

A bounded buffer must say what it lost. Here each event has a number, so the reader counts the missing numbers. The count is exact, and the producer needs no lock.

A slow reader must not slow the producer. Each subscription has its own buffer, so a slow tab loses only its own events. The engine thread and the other tabs do not see it.

Test yourself: why can `check_trace` keep its rule that event times never go backwards?

<details><summary>Answer</summary>

Only the engine thread emits events, and it reads one monotonic clock. The bus, the hub and each subscription keep the order of the events. Thus the stream order is the emit order, and the times only go forward.

</details>

## Result

NOT DONE at `cf769f8`. The code is complete. The overhead result is still missing.

- Proof: VERIFIED at `cd2890d`. The verifier passed 10 of 10 claims on the real server, the sim and the real model.
- Review: two reviewers said NOT READY at `cd2890d`. A bad `--record` path failed silently, and the process still exited 0. The fix opens the file at startup. A scoped re-check said READY at `ade2bd6`.
- Full check: `make ci` passed at `ade2bd6`.
- Changes against the plan: `ullm serve` stops open replies after 5 s on Ctrl+C. Without this limit, an open tab kept the server alive.
- Not proven yet: the telemetry overhead. A smoke session ran, but it is not a claim.
- Not seen end to end: a stalled tab with `dropped` above 0. Unit tests cover it.

Read these yourself:

- `src/ullm/telemetry/bus.py`, `Bus.drain`: the drop count comes from gaps in the index.
- `src/ullm/server/app.py`, the lifespan: the shutdown order is engine, hub, then recorder.
- `src/ullm/cli.py`, `timeout_graceful_shutdown`: an open chat reply also stops after 5 s.
