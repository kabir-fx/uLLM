# 0024: The load launcher, its control process, and the playground

**Status:** accepted · **Task:** M2-T8 (T3)

## Context

T3 asks for a launcher that picks a workload, a concurrency or an arrival rate, and `max_tokens`, then calls a local-only control endpoint that starts the Python load generator. A browser cannot hold 16 streams to one origin. The playground sends one prompt and streams the reply.

Facts this design rests on:

- `bench/loadgen.py` has only `closed_loop(client, model, requests, concurrency, template_kwargs, extra_body=)`. It returns every `Sent` at the end, in workload order. There is no open-loop (arrival-rate) generator.
- `bench/workloads/` holds one workload, `a.json`: 256 requests, `chat_template_kwargs` `{"enable_thinking": false}`. B and C arrive with M3-T8.
- `ullm serve` already serves `/telemetry/stream` (decision 0023) and allows CORS from `TERMINAL_ORIGINS` for `GET` and `POST` with `content-type` (`ullm/server/app.py`). The playground can call `/v1/chat/completions` from the terminal with no server change.
- `ullm/server/app.py` guards itself with `TrustedHostMiddleware(LOCAL_HOSTS)` against DNS rebinding and refuses a non-JSON body so every cross-origin POST needs a preflight.
- The terminal picks its source with `?live=<stream url>` or `?trace=<url>` (`terminal/src/config.ts`). C2 types are generated from JSON Schema and CI fails when they are stale (`make check` runs `check:contracts`).
- `bench` sessions take ports from `FIRST_PORT = 8100` (`bench/schema.py`).

## Decision

**A separate control process, not a route on the engine server.**

```
browser (terminal, :5173)
  ├── GET  :8000/telemetry/stream              live frames, as now
  ├── POST :8000/v1/chat/completions           playground, streamed
  └── GET/POST/DELETE :8090/load, GET :8090/workloads
                        bench.control  ── closed_loop / open_loop ──▶ :8000
```

`uv run python -m bench.control --target http://127.0.0.1:8000 [--port 8090]`. The target is fixed at startup and must be a loopback URL. The browser never names it.

**Interface** (pydantic, `extra="forbid"`, in `bench/control.py`):

```python
class Closed(Frozen):
    kind: Literal["closed"]
    concurrency: int  # 1..64


class Open(Frozen):
    kind: Literal["open"]
    rate_per_s: float  # (0, 50]


class LoadSpec(Frozen):
    workload: str  # a name GET /workloads lists
    arrival: Closed | Open  # discriminated on kind
    max_tokens: int  # 1..4096; replaces every request's max_tokens
    requests: int  # 1..the workload's length; its first N, in order


class Workload(Frozen):
    name: str
    requests: int


class LoadStatus(Frozen):
    state: Literal["idle", "running", "done", "stopped", "failed"]
    spec: LoadSpec | None
    ok: int
    errors: int  # counted when the run ends; 0 while running
    first_error: str | None


def create_control_app(client: httpx2.AsyncClient, workloads: Path) -> FastAPI: ...
```

| Route | Does | Errors |
|---|---|---|
| `GET /workloads` | `list[Workload]` from `workloads/*.json` | |
| `GET /load` | the current `LoadStatus` | |
| `POST /load` | starts a run as one `asyncio.Task`; 202 `LoadStatus` | 409 while one runs; 400 bad spec, unknown workload, non-JSON body |
| `DELETE /load` | cancels the run; its streams close, so the server aborts them; 200 `LoadStatus` `stopped` | |

The control app reuses `LOCAL_HOSTS`, `TERMINAL_ORIGINS` and the JSON content-type check from `ullm/server/app.py`, and allows `DELETE`.

**Open loop.** `bench/loadgen.open_loop(client, model, requests, rate_per_s, seed, template_kwargs, extra_body=)`: Poisson arrivals from `random.Random(seed)`, each request its own task, no cap on open requests. A refusal (503) is recorded as an error, as in `closed_loop`. `closed_loop` does not change.

**Progress** comes from the live telemetry the terminal already shows. The control status reports only state and the totals at the end.

**Terminal.**

| File | Owns |
|---|---|
| `src/control.ts` | `controlClient(base)`: `workloads()`, `status()`, `start(spec)`, `stop()`; errors as `Error` with the server's message |
| `src/playground.ts` | `streamChat(api, model, prompt, maxTokens, signal)`: an async iterator of text pieces, then the usage; reuses `SseParser` |
| `src/contracts/control.ts` | generated from `bench/control.schema.json` by `gen-contracts.mjs`, checked for staleness like C2 |
| `src/ui/Launcher.tsx`, `src/ui/Playground.tsx` | the two panels; controls in React state, polling `GET /load` once a second while a run is `running` |

Both panels render only in live mode. The API base is the origin of `?live=`. The control base is `?control=`, default `http://127.0.0.1:8090`. In replay mode T4 puts a run picker in their place.

## Alternatives

- **Routes on `ullm serve`.** Lost: the load generator would share the server's event loop and distort the numbers the terminal shows, `ullm` would import `bench`, and it adds to C1, which Lane B may not change without Lane A.
- **A Vite dev-server middleware that spawns the load generator.** Lost: it works only under `pnpm dev`, and it turns an HTTP request into a shell command.
- **Live progress counts in `LoadStatus`.** Lost for now: it needs a callback in `closed_loop`, which the benchmark harness shares, and the telemetry already shows every request.

## Consequences

- Launched load is not a benchmark. It writes no C5 directory, and no claim may come from it (rule 7).
- With `open`, a rate above what the engine serves grows the queue until the server answers 503. The errors total shows it.
