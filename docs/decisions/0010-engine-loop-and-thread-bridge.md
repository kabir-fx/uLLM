# 0010: Request state machine, engine loop and thread bridge

**Status:** accepted · **Task:** M1-T3 `[design]`

## Context

M1-T3 builds E4 (request and state machine), E5 (engine loop and thread bridge) and the E6 scheduler interface at `max_num_seqs = 1`, plus `ullm sim`. Constraints:

- Rule 2: only the engine thread touches MLX, the scheduler, the requests and the runner. Threads talk through queues. No locks in `core`.
- Rule 5: the engine thread never blocks on telemetry, logging, metrics or the network.
- Decisions 0008 and 0009 fix what the loop drives: `Runner.add / execute / release` with the lifecycle and failure invariants in `runner/base.py`, and one `StopCheck` per request that turns sampled tokens into `Chunk(text, finish_reason)`.
- C2's events and the trace checker (`telemetry/check.py`) already define the life cycle a trace must show: arrived, admitted, a prefill step, first token, finished or aborted, with `completion_tokens` equal to the tokens sampled.
- The API thread (M1-T4) is asyncio. A prefill blocks for far longer than an event loop may stall, so the model cannot run on it (E5).

**Measured fork.** Does `MLXRunner` work when `load_mlx` runs on one thread and `execute` on another? MLX 0.32.4 documents `mx.new_stream` streams as usable only on their creating thread (`mlx/core/__init__.pyi`), and mlx-lm's server loads inside its generation thread (`mlx_lm/server.py`, `_generate`). A throwaway run on the author's M4 (not committed) generated 32 greedy tokens three ways: loaded and run on the main thread, loaded on the main thread and run on a worker, and loaded and run on a worker. All three gave the same tokens. So the runner can be loaded at startup, before the engine thread starts, as decision 0008's usage shows.

## Decision

**Usage.** The server (M1-T4) calls it like this, from its event loop:

```python
runner, tokenizer = load_mlx(DEFAULT_MODEL)  # startup, before the engine thread exists
engine = EngineClient.start(runner, tokenizer.detokenizer, EngineConfig(), emit=on_event)

try:
    stream = engine.submit(
        prompt_ids,
        SamplingParams(temperature=0.7, top_p=1.0, top_k=0, seed=None),
        max_tokens=256,
        stop=("\n\nUser:",),
        trace=headers.get("traceparent"),
    )
except Unavailable:  # full or stopped: 503 with Retry-After
    ...
except ValueError:  # the request cannot run on this model: 400
    ...
async with stream:  # leaving early (disconnect, error) sends an abort ticket
    async for out in stream:
        match out:
            case Delta(text):  # zero or more
                ...
            case Finished(text, reason, prompt_tokens, completion_tokens):  # exactly one of
                ...
            case Failed(reason):  # these two ends it
                ...

engine.health()  # Health(alive, in_flight, waiting, running); never waits on the engine
await engine.close()  # at shutdown
```

`ullm sim` uses the same client over `FakeRunner`, with client coroutines instead of HTTP:

```python
engine = EngineClient.start(
    FakeRunner(SIM_INFO, SIM_LATENCY),
    FakeDetokenizer,
    EngineConfig(source="sim"),
    emit=events.append,
)
await asyncio.gather(*(client(engine, rng) for _ in range(args.clients)))
await engine.close()  # the engine thread has exited: events is complete
report = check_trace(frames(events))  # sim exits non-zero on any problem
```

**Modules.** Four, all in `ullm.core` and MLX-free.

`core/request.py`: one request's state, engine thread only.

```python
class RequestState(Enum):
    WAITING, PREFILLING, RUNNING, PREEMPTED, FINISHED, ABORTED

TRANSITIONS: Final[Mapping[RequestState, frozenset[RequestState]]]  # E4, the one list
# WAITING → PREFILLING | ABORTED; PREFILLING → RUNNING | ABORTED;
# RUNNING → FINISHED | PREEMPTED | ABORTED; PREEMPTED → WAITING | ABORTED

AbortReason = Literal["client_disconnect", "shutdown", "error"]  # the C2 req.aborted reasons
Emit = Callable[[Event], None]  # appends; never blocks (rule 5)

class Request:
    def __init__(self, ticket: Submit, seq: SeqId, stop: StopCheck, emit: Emit, t: float): ...
        # emits req.arrived
    rid, seq, prompt, sampling, trace, tenant      # read-only
    state: RequestState                            # read-only
    prefilled: int      # prompt tokens in the runner's KV: PREFILLING's progress (E13)
    def admit(self, t: float) -> None              # WAITING → PREFILLING, emits req.admitted
    def append(self, token: int, t: float) -> Chunk
        # feeds the StopCheck; the first token moves PREFILLING → RUNNING (req.first_token);
        # a finishing chunk moves RUNNING → FINISHED (req.finished, with usage and timings)
    def abort(self, reason: AbortReason, t: float) -> None  # any live state → ABORTED
    def prefill_plan(self) -> PrefillPlan          # the prompt from `prefilled`, sample=True
    def decode_input(self) -> int                  # the last sampled token
```

Every state change goes through one private `_move(to)`, which raises `IllegalTransition(ValueError)` for a pair not in `TRANSITIONS` and changes nothing. Each public method emits its own event, so a transition cannot happen without one. `PREEMPTED → WAITING` has no C2 event; nothing reaches `PREEMPTED` until E11 (M3). `completion_tokens` counts every sampled token, the end token included, as `check.py` requires.

`core/scheduler.py`: which requests run next. Pure decisions; it never calls the runner.

```python
@dataclass(frozen=True, slots=True)
class Scheduled:
    plan: StepPlan
    rows: tuple[Request, ...]      # one per sampled row, in plan order
    admitted: Request | None       # admitted this step: the engine calls runner.add for it

class Scheduler:
    def __init__(self, max_num_seqs: int): ...
    def add(self, req: Request) -> None         # back of the waiting queue
    def remove(self, req: Request) -> None      # an ended request, waiting or running
    def schedule(self, t: float) -> Scheduled | None   # None: nothing waits or runs
    waiting: int; running: int                  # for step events and Health
```

`schedule` follows E6: admit the oldest waiting request while fewer than `max_num_seqs` are admitted (it calls `req.admit(t)`); if it admitted one, plan that request's prefill (DEC-7: one prefill per step); otherwise plan one decode over every running request. E6's step 1 (remove finished) happens at the end of the step that finished them, so a reply's last chunk is not delayed a step; the effect on the next plan is the same. `EngineConfig` refuses `max_num_seqs != 1` with a `ValueError` naming M2-T2, as decision 0008 did for batched decode: the body is E6's general rule, but only 1 is tested and `MLXRunner` refuses a batch until M2-T1.

`core/engine.py`: the engine thread's loop, and the types that cross the bridge.

```python
@dataclass(frozen=True, slots=True)
class Submit:  # API → engine. Plain data, checked when built, like the plans
    rid: str
    prompt: tuple[int, ...]
    sampling: SamplingParams
    max_tokens: int
    stop: tuple[str, ...]
    trace: str | None = None  # W3C traceparent, carried across the thread boundary (C4)
    tenant: str | None = None  # DEC-9 hook; set from M8


@dataclass(frozen=True, slots=True)
class Abort:
    rid: str  # client disconnect


@dataclass(frozen=True, slots=True)
class Shutdown:
    pass


Ticket = Submit | Abort | Shutdown


@dataclass(frozen=True, slots=True)
class Delta:
    text: str  # never empty


@dataclass(frozen=True, slots=True)
class Finished:
    text: str
    reason: FinishReason
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True, slots=True)
class Failed:
    reason: AbortReason


Output = Delta | Finished | Failed


@dataclass(frozen=True, slots=True)
class Load:
    waiting: int
    running: int


Deliver = Callable[[Sequence[tuple[str, Output]], Load], None]  # once per loop turn; never blocks


@dataclass(frozen=True, slots=True)
class EngineConfig:
    max_num_seqs: int = 1
    max_in_flight: int = 64  # submitted and not yet ended; beyond it, Unavailable
    replica: str = "r1"
    source: Literal["engine", "sim"] = "engine"


class Engine:
    def __init__(
        self,
        runner: Runner,
        new_detokenizer: Callable[[], Detokenizer],
        config: EngineConfig,
        inbound: SimpleQueue[Ticket],
        deliver: Deliver,
        emit: Emit,
        clock: Callable[[], float] = time.monotonic,
    ): ...
    def run(self) -> None:
        """The engine thread's body. Returns after a Shutdown ticket."""
```

One turn of `run`, as in roadmap 5.3:

1. Take tickets: block on `inbound.get()` only when nothing waits or runs, then drain with `get_nowait` until empty. `Submit` creates a `Request` and adds it to the scheduler. `Abort` of a live request releases it from the runner, aborts it and delivers `Failed("client_disconnect")`; an `Abort` for an unknown rid (already ended) is ignored. `Shutdown` aborts every live request with `Failed("shutdown")`, delivers, and returns.
2. `schedule`; `runner.add` for the admitted request; `runner.execute(plan)`.
3. If `execute` raises, the runner has failed those sequences (decision 0008, invariant 4): release them, abort them with `"error"`, log once, and carry on with the others.
4. Otherwise emit one `step` event (`ms` is the runner's `total_ms`, so `ullm sim` shows the latency model), then `append` each row's token. A finished request is released from the runner and removed from the scheduler in the same turn.
5. `deliver` everything this turn produced, with the current `Load`, in one call.

Any other exception is a bug in the core: `run` emits `req.aborted("error")` for every live request, so the trace still ends cleanly, and re-raises. The engine thread then dies.

`core/client.py`: the API thread's side.

```python
class Unavailable(Exception):
    why: Literal["full", "stopped"]


@dataclass(frozen=True, slots=True)
class Health:
    alive: bool
    in_flight: int
    waiting: int
    running: int


class EngineClient:
    """Every method runs on the event loop's thread; `start` binds the client to that loop."""

    @classmethod
    def start(
        cls,
        runner: Runner,
        new_detokenizer: Callable[[], Detokenizer],
        config: EngineConfig,
        emit: Emit,
    ) -> "EngineClient": ...
    def submit(
        self,
        prompt: tuple[int, ...],
        sampling: SamplingParams,
        *,
        max_tokens: int,
        stop: tuple[str, ...] = (),
        trace: str | None = None,
    ) -> "RequestStream": ...
    def health(self) -> Health: ...
    async def close(self) -> None: ...


class RequestStream(AsyncIterator[Output]):
    rid: str

    async def __aenter__(self) -> "RequestStream": ...
    async def __aexit__(self, *exc: object) -> None: ...  # sends Abort unless already ended
```

- `start` reads `runner.info()` once, before the engine thread starts, and keeps it; afterwards only the engine thread calls the runner. It starts the engine on a daemon thread whose wrapper, when `run` ends for any reason, posts `_engine_exited(error)` to the loop with `call_soon_threadsafe`.
- `submit` checks what only the model knows, raising `ValueError`: a non-empty prompt, token IDs in `[0, vocab_size)`, and `len(prompt) + max_tokens <= max_model_len`, so a sequence cannot outgrow the context mid-reply. `SamplingParams`, `Submit` and `StopCheck` check the rest. It raises `Unavailable("full")` when `max_in_flight` requests are open and `Unavailable("stopped")` once the engine has exited or `close` has begun. It names the request `f"{replica}-{n}"` from its own counter, so ids cannot collide, then puts a `Submit` on the inbound `SimpleQueue`.
- The client keeps one `asyncio.Queue` per open request, keyed by rid (E5's "one queue per request"). The engine's `deliver` is `loop.call_soon_threadsafe(client._on_outputs, batch, load)`: one wakeup per engine turn, not one per token. `_on_outputs` runs on the loop, routes each output to its queue, drops outputs for a rid that is no longer open, closes a request on its terminal output, and stores the `Load`.
- `_engine_exited` runs on the loop too. It marks the client stopped and fails every open stream with `Failed("error")`, or `"shutdown"` after `close`. Because `submit` and `_engine_exited` both run on the loop thread, a request is either refused or failed; none can wait forever.
- `health` reads only loop-thread state: the thread's `is_alive()`, the number of open requests, and the last `Load`. It never waits on the engine, so it stays fast during a long step.
- `close` refuses new submits, puts `Shutdown`, and awaits the exit callback.

**Invariant across the bridge:** every accepted `submit` yields exactly one terminal output, `Finished` or `Failed`, and nothing after it. The engine delivers one for every ticket it ends, and the client's exit callback ends the rest.

**`ullm sim`** (`src/ullm/cli.py`, `[project.scripts] ullm = "ullm.cli:main"`, and `make sim` calls it): runs the real engine on `FakeRunner` with N client coroutines in one process. Each client draws a prompt length, `max_tokens` and, for a share of them, an early disconnect from a seeded `random.Random`. It prints a summary, groups the collected events into frames about 15 times a second, runs `check_trace` and exits non-zero on any problem. `--trace PATH` also writes the frames with `telemetry/trace.write`. `hello.source` is `"sim"`. `FakeDetokenizer` (in `runner/fake_runner.py`) spells token `i` as `" t{i}"`. HTTP serving on `FakeRunner` comes with the server, not here.

**Tests** (written first, in `/build`):

- `tests/unit/test_request.py`: every pair `(state, to)` outside `TRANSITIONS` raises and changes nothing; each legal path emits the expected events; timings from an injected clock.
- `tests/unit/test_scheduler.py`: FCFS; a prefill before any decode; `remove` from waiting and running; `None` when empty.
- `tests/unit/test_engine.py`: `Engine.run` on the test thread, with tickets queued in advance and list adapters: aborts of waiting and running requests, a runner failure (`FakeRunner` with a raising `sleep`) fails only its rows, exactly one terminal output per `Submit`, `runner.add` count equals `release` count, and the events pass `check_trace`.
- `tests/integration/test_bridge.py` (asyncio, `FakeRunner`): E5's proof without HTTP. 50 concurrent clients all finish; each stream's text equals its own sequence's tokens; every abort releases its sequence; `health()` returns quickly during a 500 ms modelled prefill; an engine that dies fails every open stream and reports `alive=False`; `Unavailable("full")` at `max_in_flight`.
- An `mlx`-marked test drives the golden prompts through `EngineClient` and `MLXRunner` and compares the sampled token IDs with `reference()`. This is rule 1 through the loop, not only the runner.
- `ullm sim` in a short configuration exits 0.

## Alternatives considered

- **E5 as written: a bounded inbound `queue.Queue`, with each ticket carrying its own output queue and loop.** The bound does not bound what it is meant to: the engine drains the inbound queue every turn into the scheduler's waiting list, which then grows without limit, so a 503 fires only when many requests arrive within one step. An `Abort` can hit a full queue, so the API must drop it or block the loop. And no lock-free way exists to fail requests that were put on the queue just as the engine died. Bounding open requests on the API side bounds waiting, running and in transit together, needs no shared counter, and turns the death race into two callbacks on one thread.
- **The engine as a coroutine on the event loop, running only `execute` in a single-thread executor.** One thread fewer, but the scheduler and the requests would live on the API thread, against rule 2's letter. The scheduler's work would then compete with HTTP handling, and the core would depend on asyncio.
- **Tickets that carry a stateful `StopCheck`, built and checked on the API thread.** It validates `max_tokens` and `stop` for free, but tickets stop being plain data that tests and telemetry can record. `Submit.__post_init__` checks the same fields, as the plans do.
- **Requests that return their events instead of emitting them.** The caller could drop one, and the trace checker would only find out in a test. Emitting inside each transition makes "every transition emits" structural.
- **One `call_soon_threadsafe` per token.** Simplest to write, but it wakes the event loop once per token per request. One call per engine turn wakes it once per step.

## Consequences

- **Roadmap edits in the same pull request** (3.6). E5's "Inbound queue: thread-safe and bounded. A full queue makes the API reply 503" becomes: the inbound queue is unbounded, and the client bounds open requests at `max_in_flight`, beyond which the API replies 503. E6's prefill token budget has no effect while each step prefills one whole prompt (DEC-7) and chunked prefill (E13) does not exist. The scheduler leaves it out, and `hello.limits.max_prefill_tokens` reports `max_model_len`, the most one step can prefill.
- E4 fields with no reader yet are left out: the block table (E8, M3) and the cache salt (E12, M3). The tenant and the trace context are kept, because DEC-9 asks for them on every ticket.
- Tickets wait in the inbound queue for at most one step, so a new request's queue time includes the step in progress, which may be a long prefill.
- `ullm sim` traces come from a real clock and are not reproducible; the committed fixtures stay `telemetry/synthetic.py`'s. A sim run is checked, not compared.
- Logging happens only on failure paths. Python's logging handlers are synchronous, so a failure step pays for one log write. Rule 5's budget allows that on a path that is already exceptional; O2 replaces it with counters.
- M1-T4 maps `ValueError` from `submit` and from `SamplingParams` to 400, and `Unavailable` to 503 with `Retry-After`. That covers decision 0009's hand-off, a temperature such as `1e-40`.
- M2-T2 lifts the `max_num_seqs` limit; the loop and the bridge do not change. M3 adds `PREEMPTED` paths and the block table to `Request`, and a block-manager check to admission.
