# uLLM: the plan

The plain-language plan you follow from an empty repo to a finished project. The full specification that agents build from is [`AGENT_ROADMAP.md`](AGENT_ROADMAP.md).

---

## 1. The goal

**Build a proof-of-work project that lets you confidently demonstrate the skills employers are asking for, with evidence anyone can check.**

uLLM is not a product and is not competing with oMLX, vllm-mlx or anything else. Other engines are yardsticks we measure against. Three things make it work as proof:

- **Technically sound.** The engine's output matches mlx-lm token for token, and that is tested on every change.
- **Verifiable.** Every claim you make points to something a stranger can check: a test, a committed benchmark run, a live demo link.
- **Strong.** Real depth in the hard part (the engine internals), and enough breadth to cover what inference, applied-AI and backend roles screen for.

There are no deadlines. A milestone is finished when its gate passes.

---

## 2. What we are building

A small, readable LLM inference engine for Apple Silicon, written from scratch on MLX, with a live terminal that shows every scheduling decision as it happens.

| Part | What it does | Built with |
|---|---|---|
| **Engine** | Serves one model to many users at once: continuous batching, block-based KV cache, prefix caching, preemption, OpenAI-compatible streaming API, tool calling | Python, MLX |
| **Router and gateway** | Fronts 2–3 engine copies. Sends each request to the copy that already has its prompt cached. Health checks, retries, circuit breaker, API keys, rate limits, usage metering | Go, Redis, Postgres |
| **Terminal** | Dense real-time UI: every request, every engine step, every memory block | TypeScript, React, canvas |
| **Proof** | Equivalence tests against mlx-lm, property tests, quality evals that gate CI, reproducible benchmarks, a claims ledger | pytest, hypothesis, GitHub Actions |

```
 curl / OpenAI SDK / coding agent                 Terminal (browser)
            │  HTTP + SSE                               ▲  live telemetry
            ▼                                           │
   Go router and gateway ───────────────────────────────┘
            │
    ┌───────┼───────┐
    ▼       ▼       ▼
  Engine  Engine  Engine      each one:  API → queue → [ scheduler → KV blocks → model (MLX) → sampler ] → stream
            │
   Prometheus · Grafana · traces
```

The terminal is the face of the project. It is the dense, live UI you wanted to build, and it is the part most people will actually open:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ uLLM · r1   TOK/S 412   TTFT p50 180ms  p99 950ms   TPOT 14ms   BATCH 7/16   KV 61%   ● LIVE     │
├───────────┬─────────────────────┬────────────────────────────────────────────┬───────────────────┤
│ REPLICAS  │ REQUESTS            │ STEP CHART   one candle = 1 s of steps     │ LOAD LAUNCHER     │
│ r1 ● 412  │ id state   tok ttft │   ┃     ╻                                  │ workload  A B C D │
│ r2 ● 388  │ 42 RUN      87  180 │   ┃ ╻   ┃ ╻   ╻    a long wick = a stall   │ concurrency ━━●━━ │
│ r3 ○ down │ 43 RUN      12  950 │   ╹ ┃ ╻ ╹ ┃ ╻ ┃                            │ rate · max_tokens │
│           │ 44 PREEMPT  40  190 │  ▂▅▇▅▂▇▅▃▂▅▇▅▂▃    bars = tokens per second│ [   FIRE LOAD   ] │
│ RUNS      │ 45 WAIT      –    – │                                            │                   │
│ A chat    │ 46 WAIT      –    – │ BATCH STRIP  rows = requests, cols = steps │ PLAYGROUND        │
│ B prefix  │ …                   │  A  d d d d d d . . . .   d = decode token │ > your prompt     │
│ C mixed   │                     │  B  d d d d d d . d d d   P = prefill      │ tokens stream     │
│ D agent   │                     │  D  . . . . . . P d d d                    │ here…             │
├───────────┴─────────────────────┴───────────────────────┬────────────────────┴───────────────────┤
│ Running │ Waiting │ Finished │ Preempted │ Router       │ KV BLOCK GRID   1 square = 1 block     │
│ id  state  prompt  gen  ttft   tpot  blocks  hit        │ S S S S S S S S A A B K C C A . K B    │
│ 42  RUN    3,050    87  180ms  14ms  191     98%        │ . . C B A K K . C . B . . K K A C .    │
│ 43  RUN      210    12  950ms  15ms   14      0%        │ S shared  K cached  A-C owned  . free  │
└─────────────────────────────────────────────────────────┴────────────────────────────────────────┘
```

It has two modes. *Live* connects to your running engine. *Replay* plays back a recorded run, and is what gets hosted on a public link, so anyone can open the terminal with nothing to install.

---

## 3. What it proves

This table is the reason each piece exists.

| Skill employers ask for | Where it shows up | How anyone can check it |
|---|---|---|
| LLM inference internals: KV cache, batching, prefix caching, preemption | The engine | Output-equivalence tests against mlx-lm, property tests |
| Performance work with numbers you can explain | Benchmarks against real engines | Committed raw results; every gap has a stated cause |
| Concurrency, API design, streaming | Engine thread, queues, OpenAI-compatible API | The official OpenAI SDK passes against it in CI |
| Evals and quality gates; quantization trade-offs | Eval suite | Datasets in the repo; a quality drop fails CI |
| Go, load balancing, failure handling | Router | A test that kills a replica mid-run |
| Agents: tool calling, structured output, MCP | Agent-facing API | Tool-call eval; an MCP server a coding agent can use |
| Observability | Prometheus, Grafana, OpenTelemetry | Dashboards; one trace across Go and Python |
| Real-time frontend | Terminal | The hosted replay |
| Rate limiting, metering, SQL, Redis, idempotency | Gateway | Tests for 429s and a ledger that never double-counts |
| Security | Cache timing leak, found and fixed | An attack script with before and after results |
| Docker, Kubernetes, autoscaling | Deployment blueprint | One command brings up the cluster |
| Disciplined use of AI coding tools | How the repo was built | Pull-request history, decision records, your catch log |

---

## 4. The build order

Five stages. Each ends with something you can run, show and claim. Do them in order.

| Stage | What gets built | What you can show | What you can claim |
|---|---|---|---|
| **1. An engine you can watch** (M0–M2) | Generation loop, OpenAI-compatible API, continuous batching, live terminal, metrics, first benchmarks, hosted replay | The terminal live while 16 clients hit the engine. A benchmark table. A public demo link | A from-scratch inference engine with continuous batching, verified against mlx-lm, with a real-time terminal |
| **2. Memory and quality** (M3–M4) | Block-based KV cache, prefix caching, preemption, the cache timing leak and its fix, quantization evals gating CI | Shared-prefix blocks lighting up in the KV grid. The quantization trade-off table | Prefix caching with a measured TTFT gain; property-tested memory manager; evals in CI; a side channel found and closed |
| **3. Scale-out, agents, v1.0** (M5–M7) | Go router with prefix-aware routing, tracing, tool calling, structured output, MCP server, agent-trace workload, final benchmarks, write-up | Kill a replica live and nothing is lost. An agent session replayed. The published write-up | A prefix-aware Go router with measured cache-hit gains; tool calling and MCP; a full write-up |
| **4. Platform** (M8–M9) | API keys, rate limits, usage ledger, fair-share scheduling, Docker images, Kubernetes blueprint with autoscaling | Rate limits firing per key. A cluster scaling on queue depth | Multi-tenant gateway with Redis and Postgres; Kubernetes deployment |
| **5. Depth tracks** (optional) | Pick by the roles you target: chunked prefill, speculative decoding, shared-prefix attention, a RAG ops agent | One focused before-and-after result each | One extra claim each |

**You can start applying when Stage 1 passes its gate.** Everything after it makes the project stronger while you are already in the market.

Which depth tracks to pick:

| If you are targeting | Pick |
|---|---|
| Inference and AI-infra roles | Chunked prefill, then speculative decoding |
| Applied-AI roles | The ops agent (LangGraph, RAG, evals, prompt-injection tests) |
| Backend and platform roles | Stage 4 already covers these; no extra track needed |

---

## 5. How to build it with Claude Code

Follow the same loop every time. The discipline is what makes the result trustworthy.

**For each task**

1. Open a fresh session and say: *"Read docs/AGENT_ROADMAP.md and do task M1-T3."* One task per session.
2. If the task is marked `[design]` or `[spike]`, the agent proposes a design first. Read it, push back, approve. Then it builds.
3. The agent writes the tests that pin down correctness before the code.
4. Run a code review on the result (`/code-review`) before merging anything in the engine.
5. Merge only when CI is green. No exceptions for small changes.
6. Read the teach-back notes the agent leaves: what was built, why, and three questions an interviewer could ask.

**For each milestone**

1. Run its gate (`/gate M2`). Every item must pass with evidence.
2. Write that milestone's section of `docs/DESIGN.md` yourself.
3. Take the quiz (`/quiz M2`) without looking at the code. Re-read whatever you miss.
4. Add any new claim to the claims ledger with its evidence.

**Rules that keep it sound**

- **Two lanes at most.** The engine is the main lane and gets your attention. A second lane (terminal, then evals, then router) runs in a separate git worktree against the simulated backend. A third lane would produce code faster than you can understand it.
- **No claim without evidence.** Every number comes from a committed benchmark run. Nothing is estimated.
- **Slower is fine, unexplained is not.** uLLM will trail production engines in places. Report the gap and its cause.
- **Nothing off-roadmap without a record.** If you or an agent want to add or change something, write down why and update the agent roadmap in the same change.

---

## 6. Your part

The agents write most of the code. These are what turn it into a project you can defend for an hour.

- **Read every engine-core diff**: scheduler, block manager, engine loop, KV code. Skim the plumbing.
- **Write `docs/DESIGN.md` in your own words.** Agents can check it for accuracy. They do not write it.
- **Take the quiz at every gate.**
- **Keep the catch log** (`docs/AI_LOG.md`): each time you catch an agent mistake, two lines on what it got wrong and how you noticed. Interviewers ask for exactly this story.
- **Run the long benchmarks**: plugged in, Low Power Mode off, other apps closed.
- **Keep interview prep going alongside**: DSA and system design, outside project time.

---

## 7. Resume lines

Templates. Fill the brackets only from the claims ledger.

**After Stage 1**

- Built uLLM, a from-scratch LLM inference engine for Apple Silicon (Python, MLX) with a continuous-batching scheduler and an OpenAI-compatible streaming API; [N]× throughput at [8] concurrent requests over sequential serving, with output verified token-for-token against mlx-lm in CI.
- Built a real-time observability terminal (TypeScript, React, canvas) streaming per-step scheduler telemetry, with [X]% measured engine overhead and a hosted replay of recorded runs.

**After Stage 2**

- Implemented a block-based KV cache with ref-counted prefix sharing, LRU eviction and preemption; cut time-to-first-token by [X]% on shared-prefix workloads; invariants verified with property-based tests.
- Built quantization quality evals (perplexity, exact match, tool-call validity across bf16, 8-bit, 4-bit) that gate CI.
- Demonstrated a cross-tenant prompt-cache timing side channel and closed it with per-tenant cache salting.

**After Stage 3**

- Wrote a prefix-aware request router in Go (health checks, retries, circuit breaker) across [3] replicas; [X]% prefix-cache hit rate against [Y]% for round-robin.
- Added tool calling, JSON-schema structured output and an MCP server; replayed recorded coding-agent sessions as a benchmark.

**After Stage 4**

- Built a multi-tenant gateway: API keys, Redis token-bucket rate limits, and an idempotent Postgres usage ledger that never double-counts retries; deployed to Kubernetes with autoscaling on queue depth.

---

## 8. Start here

M0 (foundations) is built, merged and protected: the four CI jobs are required on `main`. Before M1:

1. Read the M0 decision records in `docs/decisions/`, especially 0006: greedy output can differ between Macs at near-tied tokens, so equivalence compares against mlx-lm on the same machine.
2. Read panels 0, 1, 5 and 6 of the architecture walkthrough. That is the material for milestone M1.
3. Open a session and say: *"/task M1-T1"*. It is a design task, so the agent shows you the design and waits for your approval.
4. Keep the catch log (`docs/AI_LOG.md`): add an entry each time you catch an agent mistake.
