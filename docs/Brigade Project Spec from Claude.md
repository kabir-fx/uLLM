# LLM Inference Engine for Apple Silicon: Build Reference

## 1. Overview

A small, readable inference engine that serves an LLM to many users at once on an M-series Mac. It sits behind a Go router that spreads requests across several copies of the engine.

**It implements:**
- continuous batching
- a paged KV cache
- prefix caching
- preemption
- pluggable scheduling
- an OpenAI-compatible streaming API with tool calling

**It comes with:**
- standard metrics (Prometheus, Grafana, OpenTelemetry)
- quality evals that gate CI
- a live web terminal
- reproducible benchmarks against production engines

**Why:**
- **The gap:** no readable inference engine runs on Macs. nano-vLLM needs NVIDIA GPUs, and the Mac engines are built for speed, not readability.
- **What it covers for hiring (2026 job postings):**
  - **Inference roles:** KV cache, batching, routing, measured performance.
  - **Applied AI roles:** tool calling, evals, agent workloads.
  - **SWE roles:** concurrency, APIs, Go, observability, CI.

**Out of scope:** multi-GPU, CUDA, training or fine-tuning, cloud deployment, beating production engines on raw speed.

---

## 2. Core concepts

| Concept | Problem | Implementation |
|---|---|---|
| Prefill vs decode | Prefill (reading the prompt) is limited by compute. Decode (one token per step) is limited by memory bandwidth, so one request at a time underuses the GPU | Scheduler treats the two phases separately |
| Continuous batching | A static batch waits for its slowest request | Each step, finished requests leave and waiting ones join |
| Paged KV cache | Reserving max-length memory per request wastes RAM | Fixed blocks (16–32 tokens). Each request has a block table |
| Prefix caching | A shared system prompt is recomputed for every request | Hash each full block (chained with the parent block's hash). Shared blocks get ref counts. Unreferenced blocks are evicted LRU |
| Preemption | KV memory runs out mid-generation | Pause the newest request, free its blocks, recompute it later |
| Scheduling | Order of service | FCFS now. Per-key fair share later (§7) |
| Prefix-aware routing | Replicas that don't know about each other's caches recompute the same prefixes | Router sends a request to the replica that already holds its prefix, otherwise to the least-loaded one |

---

## 3. Architecture

```
Clients / coding agent (OpenCode)
        │ OpenAI-compatible HTTP + SSE
        ▼
 Go router ── prefix-aware routing, least-load fallback, health checks,
        │     timeouts, retries, circuit breaker, /metrics
        ▼
 Engine replicas ×2–3 (Python, run natively on the Mac)
   API server (FastAPI) → request queue → Engine loop (dedicated thread):
     Scheduler → Block manager → Model runner (MLX) → Sampler → Output stream
   /metrics (Prometheus) · OpenTelemetry spans · telemetry WebSocket
        │
        ▼
 Prometheus + Grafana (docker compose)      Live terminal (React)
```

**Design decisions:**
- **Reuse mlx-lm models unchanged.** Implement `PagedKVCache` with mlx-lm's cache interface (`update_and_fetch(keys, values)`).
- **Attention:** gather a request's blocks into a contiguous tensor, then call `mx.fast.scaled_dot_product_attention`.
- **Variable-length batches:** left-padding plus attention masks. Read mlx-lm's batch generation code first.
- **MLX is lazy:** call `mx.eval` at each step end, and only time code after it.
- **Threading:** asyncio for the API, one engine thread, thread-safe queues between them. Only the engine thread calls MLX.
- **Docker on macOS can't use the Metal GPU.** Engines run natively, and only Prometheus and Grafana run in docker compose.
- **Router benchmarks:** the replicas share one GPU, so report cache hit rate and TTFT, not throughput.
- **Tool calling:**
  - Accept `tools` and `tool_choice`.
  - Parse Qwen3's tool-call output into OpenAI `tool_calls`.
  - Return `usage` token counts.
- **Design notes:** every design decision goes in `docs/DESIGN.md`, written by you.

---

## 4. Tools

| Area | Choice |
|---|---|
| Hardware | MacBook Air M4. 16 GB RAM is enough for 2–3 replicas of a 0.6B model, or one 1.7B model |
| Engine | Python 3.12, `uv`, `mlx`, `mlx-lm` |
| Models | `mlx-community/Qwen3-0.6B` (dev, router tests), `Qwen3-1.7B` (benchmarks). bf16, 8-bit and 4-bit variants for evals |
| API | FastAPI, uvicorn, pydantic, SSE |
| Router | Go (`net/http`, `httputil.ReverseProxy`) |
| Observability | `prometheus_client`, OpenTelemetry SDK, Prometheus, Grafana, docker compose |
| Live terminal | Vite, React, TypeScript, `<canvas>` (request feed, batch view, KV block grid) |
| Tests | pytest, hypothesis |
| CI | GitHub Actions on a macOS arm64 runner (MLX on CPU) |
| Load generator | asyncio + httpx |
| Agent harness | OpenCode (or any coding agent that accepts OpenAI-compatible endpoints) |
| Baselines | `mlx_lm.server`, Ollama, vllm-metal |
| System monitoring | `powermetrics` (GPU frequency, thermal throttling) |

---

## 5. Testing and evals

- **Output equivalence:** with greedy decoding, output matches `mlx_lm.generate` token for token, for one request and for many concurrent requests.
- **Property tests:**
  - No block is allocated twice.
  - Ref counts never go negative.
  - Freed memory can always be reused.
- **Quantization evals:** compare bf16, 8-bit and 4-bit on:
  - perplexity over a fixed text
  - exact match on a 100–200 item task set
  - speed and memory use
- **Tool-call eval:** % of tool calls that are valid and parse correctly.
- **CI gate** (runs on every push):
  - lint, type check, unit and property tests
  - a short greedy-equivalence test on Qwen3-0.6B
  - a small eval subset with a minimum score threshold

---

## 6. Benchmarking

**Workloads:**

| Workload | Shape | Tests |
|---|---|---|
| A: Chat | Prompts 100–1k tokens, outputs 50–300 | Throughput, batching |
| B: Shared prefix | 4–8 concurrent requests sharing a 2–4k-token system prompt | Prefix caching, routing |
| C: Mixed | One 8k-token prompt among short chats | Latency fairness |
| D: Agent traces | Recorded multi-turn traces from OpenCode on a small repo task, replayed | Per-turn prefix hit rate, TTFT |

**Load:** fixed concurrency (1, 2, 4, 8, 16) and Poisson arrivals at fixed rates.

**Metrics:**
- Output tokens/sec.
- TTFT and TPOT, p50/p99.
- Max concurrent sequences.
- KV utilization and memory waste %.
- Prefix hit rate.
- Router vs round-robin: hit rate and TTFT.

**Baselines:** engine v0 (sequential), mlx-lm server, Ollama, vllm-metal. Use the same model and quantization for all.

**Run conditions:**
- Plugged in, Low Power Mode off, other apps closed.
- 2-minute cooldown between runs.
- Report the median of 3 runs.
- Log GPU frequency with `powermetrics`.
- Run long benchmark matrices unattended.

---

## 7. Plan

| Stage | Build | Done when |
|---|---|---|
| 1 | Model loading, generation loop with contiguous KV, OpenAI-compatible streaming API with `usage`, load generator, CI | Output matches mlx-lm. v0 and mlx-lm baselines recorded. CI green |
| 2 | Continuous batching, FCFS scheduler, telemetry bus, live terminal v1 (request feed and batch view), Prometheus `/metrics` and Grafana, README | **Resume-ready:** public repo, demo GIF, benchmark table (v0 vs mlx-lm vs batching at concurrency 1–16) |
| 3 | Paged KV cache, block manager, prefix caching, preemption, KV block grid, Go router skeleton (proxy, round-robin, health checks, timeouts, retries) | Prefix-cache gain on workload B. Memory waste measured. Router serving 2–3 replicas |
| 4 | Prefix-aware routing vs round-robin, quantization evals with CI gate, tool calling and workload D, full benchmarks, write-up | Final results in README. Write-up published |

**If scope has to shrink:**
- **Batching is taking long:** ship the resume-ready stage anyway and move metrics to stage 3.
- **Paged KV is taking long:** build the whole router in stage 4 and move tool calling to the later list.
- **Never cut the quantization evals.**

**Dropped:** oMLX baseline, 4B-model runs, shortest-first policy, latency charts in the React terminal, custom Metal kernel.

**Later, in this order:**
1. API keys, per-key rate limits (Redis token bucket, HTTP 429), and a usage ledger in Postgres that never double-counts retries, plus fair-share scheduling per key
2. An MCP server exposing engine metrics and controls
3. JSON-schema structured output
4. Goodput at latency targets
5. Chunked prefill
6. Kubernetes blueprint (`kind`, Helm, autoscaling)
7. Speculative decoding

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| Variable-length batching in MLX is complex | Padding and masks first. Study mlx-lm's batch code |
| Gather-based paged attention is slow | Measure and report it honestly |
| Thermal throttling (the Air has no fan) | Follow the run conditions in §6 |
| RAM limits | Use 0.6B for multi-replica tests. Close other apps |
| Scope creep | Follow the plan and the order of the later list |
| Not understanding generated code | Write `docs/DESIGN.md` yourself before calling any module done |

---

## 9. References

- vLLM / PagedAttention: https://arxiv.org/abs/2309.06180
- Orca (continuous batching): https://www.usenix.org/conference/osdi22/presentation/yu
- Sarathi (chunked prefill): https://arxiv.org/abs/2308.16369
- nano-vLLM: https://github.com/GeeeekExplorer/nano-vllm
- MLX: https://github.com/ml-explore/mlx
- mlx-lm: https://github.com/ml-explore/mlx-lm
- vllm-metal: https://vllm.ai/blog/2026-09-22-vllm-metal-v0-28-0
- Skills research behind these additions: `reports/AI and SWE job skills 2026.md`
