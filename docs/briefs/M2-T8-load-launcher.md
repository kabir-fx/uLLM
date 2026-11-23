# M2-T8: Load launcher and playground

**State:** DONE at `7db5646`. Type `/ship` to open the pull request.
**Decision record:** [0024](../decisions/0024-load-launcher.md) (the full design, for agents) · **Spec:** roadmap M2-T8 (T3)

## The goal

You start a load run from the terminal and see the engine take it, live. You also send one prompt and see the reply stream in.

## The problem

- To put load on the engine now, you run `bench.run` in a shell. The terminal only watches.
- A browser cannot hold 16 streams to one origin (T3). So a Python process must send the load, not the page.
- The load generator has only a closed loop. T3 also asks for an arrival rate.
- Lane B may not change a contract (roadmap 3.5). The engine server must not change.
- The control endpoint starts work on your machine. It must refuse other origins and other hosts (rule 13).

## How it works

```
terminal (:5173)
  ├── GET  :8000/telemetry/stream        live frames, as now
  ├── POST :8000/v1/chat/completions     playground
  └── :8090/load, :8090/workloads        launcher
              bench.control ──load──▶ ullm serve (:8000)
```

1. `bench/control.py` is a small FastAPI app on 127.0.0.1:8090. You start it with `uv run python -m bench.control --target http://127.0.0.1:8000`. It accepts only a loopback target, set at startup.
2. `POST /load` starts one run. `GET /load` gives its state. `DELETE /load` stops it, and the server aborts the open requests.
3. `bench/loadgen.py` gets `open_loop`: requests arrive at random times at a mean rate, from a fixed seed. `closed_loop` stays the same.
4. `terminal/src/ui/Launcher.tsx` is the launcher panel. You pick the workload, closed or open, the concurrency or the rate, `max_tokens`, and the number of requests.
5. `terminal/src/ui/Playground.tsx` sends one prompt to the server directly and streams the text. It needs no server change.
6. The TypeScript types for the control app come from its JSON Schema, as the C2 types do.

## What you get at the end

Run `uv run ullm serve --fake`, `uv run python -m bench.control --target http://127.0.0.1:8000`, and `pnpm -C terminal dev`. Open `http://localhost:5173/?live=http://127.0.0.1:8000/telemetry/stream`. Click to fire load, and the ticker, chart, strip and table fill. Type a prompt, and the reply streams in.

New tests:

- The control app starts a run against a real `ullm serve --fake` app, and the totals match the requests sent.
- A second start while a run is open gets 409. A stop cancels the run, and the server sees the aborts.
- A wrong `Host`, a foreign `Origin`, a non-JSON body, and a bad spec are refused.
- `open_loop` sends at the right mean rate for a fixed seed, and gives the same times twice.
- The terminal tests build the right request, show 409 and other errors, and stream a reply through `streamChat`.

What you do not get yet:

- The run picker for replay mode, and the hosted build: M2-T9.
- Workloads B and C: M3-T8.
- `make up`, which starts all three processes: M5-T4.

## Where this puts the project

```
M2  T1 █  T2 █  T3 █  T4 █  T5 ░  T6 ░  T7 █  T8 █  T9 ░  T10 ░  T11 ░  T12 ░  T13 ░
```

After this slice, M2-T9 and M2-T10 are open in Lane B.

## What you approve

1. A separate control process beat a route on `ullm serve`. A route would share the server's event loop and change C1.
2. The terminal shows progress from the live telemetry. The control status gives only the state and the totals at the end. Live counts need a callback in `closed_loop`, which the benchmark harness shares.
3. Open-loop arrivals are random, with a fixed seed, at a mean rate. A fixed interval is the other option. Recommended: random, because real traffic comes in bursts.
4. The launcher and the playground show only in live mode. Recommended: yes, because T4 puts a run picker there in replay mode.
5. The limits are: concurrency 1 to 64, rate up to 50 a second, `max_tokens` 1 to 4096. Recommended: these limits, because they cover workload A and the M2 sweep.

## What can go wrong

- An open-loop rate above what the engine serves fills the queue. Then the server sends 503, and the errors total shows it.
- A launched run is not a benchmark. It writes no result directory, and no figure may come from it (rule 7).
- If the control process stops, the launcher shows that it cannot reach it.

## Words

| Word | Meaning |
|---|---|
| closed loop | N clients, and each sends its next request when its last reply ends. |
| open loop | Requests arrive at a rate, whatever the server does. |
| control process | `bench.control`, the local app that sends launched load. |

## Learn from this slice

A closed loop measures the server at a fixed load. When the server slows, the clients slow too, so the queue stays short. An open loop keeps the same rate, so a slow server builds a queue. You see queueing delay only in the open loop.

A browser page can send a request to any local port. The control app checks the `Host` header and the `Origin`, and it accepts only JSON. A JSON body forces a CORS preflight, so a foreign page cannot start a run.

Test yourself: why does a launched run not count as a benchmark?

<details><summary>Answer</summary>It does not record run conditions or write a C5 directory. A claim must come from a committed result (rule 7).</details>

## Result

**State:** DONE. The proof is VERIFIED at `7db5646`: 8 of 8 claims. The review was NOT READY at `9dc61f1`, and the fix commits up to `7db5646` close every Required finding. `make ci` passes at `7db5646`. This change used 3 agent runs.

**Against the plan:** the build followed the plan. Two things were added: a client with no cap on open connections, and a run that fails on any error.

**What proof and review found:**

- httpx's default pool holds 100 connections, so an open loop could never have more than 100 requests open. The reviewer saw a peak of 100 at the server. A test now holds 120 open, and `target_client` lifts the cap.
- An error outside the caught types left the run in `running` for good, so no new run could start. A test now makes the parser raise `RecursionError`. The run ends `failed`, and the next start gets 202.
- No test covered the 4,096-byte body limit. A test now sends a body one space over. With the limit removed, the test fails with 202.
- Decision 0024 named the parameter `client`, but the code says `target`. The `make contracts` help named only the telemetry schema. Both are fixed.

**What the verifier saw:**

- A closed run of 64 at concurrency 8, in headless Chrome, filled the feed live. It ended `done: 64 ok, 0 errors`, and the tap saw 64 finishes.
- A stop during an open run at 4 a second read `stopped`. The server saw 3 aborts 61 ms after the click, and no arrivals after it.
- A second start from another tab showed `409: a run is running; stop it first`. The first run still ended done.
- The playground reply grew in 249 DOM updates.
- Every bad target, Host, Origin, content type and out-of-limit spec was refused. Replay mode shows neither panel.
- With `task.cancel()` removed, the stop check failed: 13 arrivals came after the stop.

**What is not proven:**

- The real MLX model. Every run used `FakeRunner`.
- The panels' logic in unit tests: the polling, the disabled button, the error text. The verifier drove it in Chrome. M2-T10 brings Playwright.
- Port 5173 on 127.0.0.1, which another app held. The verifier used `[::1]:5173`.

**Read these yourself:**

- `bench/control.py:89-122`: the run task. Cancel sets `stopped` and re-raises, any other error sets `failed`, and the totals come only at the end.
- `bench/control.py:150-173`: the gates before a run starts, in order: content type, size, spec, workload, 409.
- `bench/loadgen.py:124-158`: the arrival times from the seed, and one task per request with no cap.

