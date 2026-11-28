# uLLM: Agent Master Roadmap

The single source of truth for building uLLM. Every piece that has to be built is specified in this file. Read section 0, then the catalog entries your task names, before doing anything.

**Contents:** [0 Current state](#0-current-state) · [1 Goal](#1-goal) · [2 Skills and evidence](#2-skills-and-evidence) · [3 Rules](#3-rules) · [4 Verified facts](#4-verified-facts) · [5 System design](#5-system-design) · [6 Build catalog](#6-build-catalog) · [7 Milestones](#7-milestones) · [8 Depth tracks](#8-depth-tracks) · [9 Risks and fallbacks](#9-risks-and-fallbacks)

---

## 0. Current state

**Update this block at the end of every session.** It is the memory between sessions.

| | |
|---|---|
| Phase | M3 in progress: M3-T3 shipped as #36; M3-T2 as #35; M3-T1 as #34. M2 built: `/gate M2` passed every item but M2-T13 `[user]` (DESIGN.md, the quiz), which the author takes later (decision 0031); `v0.1` waits for them. M2-T12 shipped as #32; M2-T11 as #31; M2-T6 as #30, M2-T10 as #29, M2-T9 as #28, M2-T8 as #26, M2-T5 as #27, M2-T4 as #25, M2-T7 as #24, M2-T3 as #23, M2-T2 as #22, M2-T1 as #21. M1 done (gate G1 passed). M0 done |
| Remote | `https://github.com/kabir-fx/uLLM.git`, public. `main` is protected: the four CI jobs are required and the branch must be up to date; no force-push or deletion. Task pull requests stack and merge bottom-up with merge commits (decision 0001) |
| Next task | M3-T4, admission by free blocks, memory metrics and the KV usage panel (E6, E8, O3). Open for the author: M2-T13 (DESIGN.md for M1 and M2, the quiz, resume lines), then the `v0.1` tag. Batched replies are held to rule 1 at the engine (decision 0021); the server equivalence test still sends one request at a time. `check_trace` checks the row cap but not the slot budget (decision 0020); the scheduler property test checks it at the runner |
| Lane A (engine) | M3-T3 shipped as #36 (decision 0034): C3 v2, plans carry each sequence's block table and `release(seq, freed)` names the blocks that became Free; `RunnerInfo.block_size` (16); the `Scheduler` owns a `BlockManager` of `ceil(max_kv_slots / block_size) + max_num_seqs` blocks, so slot admission never runs it out until M3-T4 admits by blocks. `ullm.runner.kv.store`: `KVStore`, `BlockKV` (layout B, the default) and `ContiguousKV` (E7, the reference); `MLXRunner(model, info, kv="blocks" | "contiguous")`. A prefill is cut into blocks and its cache dropped; a joining row is gathered from its blocks, and its partial block leaves the store; a decode copies each block it fills. `make equivalence` runs both layouts at batch 1, 2, 4, 8 and requires equal tokens (claims row 42). `tests/property/test_kv_store.py` drives both stores against one reference; `tests/plans.py` builds plans with tables for runner tests. M3-T2 shipped as #35 (decision 0033): the KV layout is E9's B, E7's contiguous working tensor plus a block store of record. A block is copied out with `mx.contiguous` when it fills, and a joining row is gathered from its blocks. `scripts/kv_layouts.py` measured it on the real model (`bench/results/m3/kv-layout/`, each sample in a fresh process): at 8 rows and 1,024 tokens, B is 0.91 of E7, A (gather every step) 3.56, A2 (one pool, `mx.take`) 3.35. At that size B holds 1.78 times E7's KV, which M3-T4 must budget. C's split attention is slower than the fused kernel with MLX's own operations. One process kept its memory footprint's peak, and the run was killed: a candidate report to MLX. Claims rows 40 and 41. M3-T1 shipped as #34 (decision 0032): `ullm.core.block_manager.BlockManager(num_blocks, block_size=16)` owns each `SeqId`'s block table; Free, In use and Cached blocks; sharing through `allocate(..., shared=...)`, only blocks with a key (full by construction); keys are opaque until E10; eviction is implicit, least recently cached first, a freed table tail first. Nothing calls it yet. Sizing from a memory budget and E8's metrics are M3-T4's (the author's call). `tests/property/test_block_manager.py` is a Hypothesis state machine that models the Cached order. M2-T12 shipped as #32 (decision 0030): README v1 with the demo GIF (`make demo-gif`, Playwright video of the hosted replay, ffmpeg), the hosted replay link, a Mermaid architecture diagram, the Sustained c1 to c8 table that `bench.readme` (`make readme`, last step of `make bench-sweep`) writes between markers from the committed VALID sessions, refusing mixed setups; `tests/integration/test_readme_tables.py` in ci-core fails when it is stale; `make verify` runs equivalence at 128 tokens, the property tests, `ullm sim` and `make bench-smoke` (2 min 49 s on the M4); claims rows 34 to 39. M2-T11 shipped as #31 (decision 0029): `make grafana` runs Prometheus 3.7.3 and Grafana 12.3.0, pinned by digest, on 127.0.0.1; Prometheus scrapes `host.docker.internal:8000` every 5 s, a name `ullm serve` accepts on `GET /metrics` only (`_LocalHosts`); Grafana is anonymous Viewer with login off and a foreign `Host` redirected; one dashboard on metrics that exist now, KV usage moved to M3-T4, breaker state and router load to M5-T1. M2-T6 shipped as #30 (decision 0028): `make bench-sweep` runs workload A at c1, c2, c4, c8 and c16 against batched `mlx_lm.server` without cache, into `bench/results/m2/sweep/`; `bench.steps` splits uLLM's recordings, `bench.analysis --sweep` prints `comparison.md`, `scripts.step_costs` times decode steps and repacks offline. At Sustained, uLLM's TTFT is 0.841 to 0.424 of the baseline's from c1 to c8 and output tok/s 1.006 to 1.120 (claims 30 to 32); TPOT 1.044 at c4 from one-prompt-per-step prefill (row 31). M1's 1.167 at c1 does not recur: 1.014 now (row 32). c16 DISCARDED twice on swap: the baseline holds 11.48 GiB (row 33). At 16 rows a repack costs about 1.2 decode steps, for M3. M2-T5 (decision 0024): `GET /metrics` in the Prometheus text format (`prometheus-client` 0.26.0). The `Hub` feeds each drain to `Metrics` on the event loop, so the engine thread never touches a metric. A scrape flushes the hub first, so it counts every reply a client has read. Gauges come from `EngineClient.health()`. `--telemetry off` gives 404. Token counters count work done. After a bench run, the counters equal `raw.jsonl` (claims row 26). M2-T4 (decision 0023): the engine thread emits into a bounded `Bus`; a `Hub` on the event loop fans it out about 15 times a second to bounded `Subscription`s, one per `/telemetry/stream` tab or recorder; `ullm serve --telemetry on|off --record PATH`; Ctrl+C cuts open replies after 5 s. Overhead at c8, Sustained: output tok/s 0.993 [0.981, 1.017], within the twin's noise (claims row 25). M2-T3 (decision 0021): engine equivalence at `max_num_seqs` 1, 2, 4, 8; batch 1 exact, above 1 every token in mlx-lm's top 2 and within 0.25 logprob on uLLM's own prefix; `tests/property/test_scheduler.py` drives `Engine` on `FakeRunner` with Hypothesis. M2-T2 (decision 0020): `Scheduler` admits the waiting head only while rows ≤ `max_num_seqs` and rows × the longest reservation (`prompt + max_tokens`) ≤ `max_kv_slots` (40,960, never below `max_model_len`); FCFS, no skipping. `ullm serve --max-num-seqs --max-kv-slots`. M1 bench groups pin `--max-num-seqs 1`. M2-T1 (decision 0019): `BatchKV`, batched decode in `MLXRunner` |
| Lane B (contract-isolated) | M2-T10 shipped as #29 (decision 0027): Playwright 1.65.1; `make check-ts` runs a smoke test of the hosted replay and keyboard tests in headless Chromium on 127.0.0.1:4183; `make terminal-perf` checks the T7 budgets locally (60 fps on a synthetic `stress` trace with 2,304-block snapshots and 200 finished rows; flat live-mode heap from 20 to 40 repeats; hosted replay ready ≤ 1 s) and writes `bench/results/m2/terminal-perf/` (claims rows 28 and 29); Space and the arrow keys drive the replay; the requests tabs follow the WAI-ARIA tabs pattern; `:focus-visible` ring. Lane B has no open M2 task; M4 is next for Lane B. M2-T9 shipped as #28 (decision 0026): replay mode has play, pause, speed, a scrubber and a run picker (`terminal/src/runs.ts`); the header shows REPLAY and each real run's machine and commit from its session's `env.json`; `make terminal-hosted` builds a replay-only site; `.github/workflows/pages.yml` publishes it only on `workflow_dispatch`, after the author enables Pages. Next for Lane B: M2-T10. M2-T8 shipped as #26 (decision 0025): `bench.control` on 127.0.0.1:8090 runs one load at a time (closed loop, or `open_loop` with Poisson arrivals from seed 0) against a loopback target fixed at startup; the launcher and playground panels show only in live mode. M2-T7 shipped as #24 (decision 0022): step chart (candles per second, processed and sampled tok/s bars), batch strip (lanes), requests table (four tabs, windowed rows), all own Canvas 2D or DOM, no new dependency. M1-T8 shipped as #19 (decision 0017) |
| Blocked on the author | M1-T7 leftovers: a yes or no on deleting the five old, untracked run directories in `bench/results/m1/`, close pipeline gap 1 (`enforce_admins`); three optional M1-T1 follow-ups to accept or decline: equivalence at 128 tokens in CI (about 17 s more, and `test_goldens.py` shares the setting), a test that `release` frees KV memory, and `load_mlx` resolving the pinned snapshot locally (needs `huggingface_hub` in engine code) |
| Open pipeline gaps | 1. `enforce_admins` is off on `main`, so an admin can merge past a red required check. Fix, on the host: `gh api -X POST repos/kabir-fx/uLLM/branches/main/protection/enforce_admins`. 2. CI does not run the suppression ratchet in `QUALITY.md`; the bar-guard job that would show new suppressions was declined. 3. The pre-push hook runs only in a clone that ran `git config core.hooksPath scripts/git-hooks` |
| Last session | `/ship M3-T3` as #36: the `/build` tick commit had cut this file to section 0; the next commit restored sections 1 to 9 from `main`. `/build M3-T3`: design approved with its three recommended answers (the engine builds its manager now, a required `freed`, AC power). Proof VERIFIED at `e72a2a2` by the verifier; its claim 3 as first worded (8 concurrent replies equal each reply alone) failed at base `e41a11f` too, from decision 0021's batch near-ties, and head's replies equal base's. Review NOT READY at `e72a2a2` on five Required findings, all fixed and shown: a joining row's stale partial block stayed in the store, two comments and decision 0034 contradicted the code, and 0034 quoted figures from an uncommitted script; plus two Optional ones taken (a failed-prefill test, an unknown layout raises). `make ci` passed at `5c4fd61`. `make bench-smoke` varied more with run order and heat than with the layout; the pull request carries the four pairs. Not taken, for the author: a `--max-kv-slots` of 10^9 now builds a 2.9 GB pool at startup (operator-only; M3-T4 sizes the pool from memory). From M3-T2: the `kv_layouts` report marks the tree dirty when only untracked files or its own outputs differ. From the gate A session: claims row 21 says "(row 29)" where it means row 32; the GIF's header shows the row 25 recording's "fab197c dirty"; `bench.control`'s default port 8090 can clash with another local service. Earlier open items: `summary.md` prints GiB under the label GB; `bench.session` exits 1 with a traceback on an invalid group while its docstring says 2, and also exits 1 on a crash, which the Makefile's `-` on the c16 line hides; `--sweep` gives duplicate column headers for two sessions at one concurrency; in `docs/decisions/README.md` the 0027 row sits below the template. Lane B items still open from M2-T10's session: a bad `?run=` or `?speed=` leaves an error page with no picker; `ullm serve` sends `commit` and `machine` null in `hello` (Lane A); `ullm serve --fake` says `source: engine`; `TERMINAL_ORIGINS` omits `vite preview`'s :4173; a trace whose frame `t` goes backwards never ends; control accepts lax types; stopped runs report 0 totals; `[::1]` not a loopback target; `run-ullm` skill says the terminal has nothing to run; `docs/decisions/README.md` lacks rows for 0011, 0016, 0017 and 0023; the replay panel's selects overflow at 1600 px; `bench.conditions` records but does not enforce B5 for `terminal-perf`; the stress trace averages 14 frames/s, not 15 |

---

## 1. Goal

uLLM (called "Brigade" in older documents) is a **proof-of-work project**. It exists so its author can confidently demonstrate the engineering skills that are in demand for inference, applied-AI and backend roles, with evidence anyone can check.

**What gets built:** a small, readable LLM inference engine for Apple Silicon, written from scratch on MLX, behind a Go router and gateway, with a dense real-time terminal that shows every scheduling decision as it happens, and a test, eval and benchmark suite that proves it works.

**It is not a product.** It does not compete with mlx-lm, vllm-mlx, vllm-metal, oMLX or anything else. Other engines are used as baselines to calibrate against. Being slower than a production engine is fine when the gap is measured and explained.

Three properties decide every trade-off, in this order:

1. **Technically sound.** Correct first: greedy output matches mlx-lm token for token, and invariants hold under property tests.
2. **Verifiable.** Every claim maps to an artifact a stranger can check: a test in CI, a committed benchmark run, a hosted replay, a script. A feature is not done until its evidence exists.
3. **Strong.** Real depth where it counts (the engine internals), and enough breadth to show the skills in section 2.

What follows from that for agents:

- Prefer depth with proof over more features. Do not add a feature this file does not list without recording why (3.6).
- Prefer designs the author can explain in an interview over clever ones.
- **No deadlines.** A milestone is done when its gate passes. Do not write deadlines, target dates, schedules or time estimates into any document. Measured figures, such as how long a check or a benchmark takes, are fine.
- Never write "first", "only" or similar novelty claims anywhere.

---

## 2. Skills and evidence

The reason each piece exists. "Where" refers to catalog IDs in section 6.

| Skill in demand | Where in uLLM | Evidence a reviewer can check | Milestone |
|---|---|---|---|
| LLM inference internals: prefill and decode, KV cache, continuous batching, prefix caching, preemption | E1–E12 | Equivalence suite against mlx-lm, property tests, benchmark tables | M1–M3 |
| Performance work with numbers: TTFT, TPOT, p50/p99, throughput, goodput | B1–B6 | Committed raw results, generated tables, a stated reason for every gap | M1, M2, M7 |
| Concurrency: threads, asyncio, queues, backpressure, cancellation | E5, A1 | 50-client integration test, `/health` stays fast during a long step | M1 |
| API design: streaming, OpenAI semantics, exact token accounting, error handling | A1 | The official OpenAI SDK passes against it in CI | M1 |
| Testing discipline and CI/CD | Q1–Q4, F2 | CI status on every pull request | all |
| Real-time frontend: TypeScript, React, canvas | T1–T8 | The hosted replay, frame-rate tests | M2–M5 |
| Observability: Prometheus, Grafana, OpenTelemetry | O1–O4 | Dashboard, one trace across Go and Python, counters matching the load generator | M2, M5 |
| Evals and quality gates; quantization with quality measurement | Q5–Q7 | Checked-in datasets, the trade-off table, an eval threshold that fails CI | M4 |
| Go, load balancing, failure handling, distributed-systems basics | R1–R8 | Chaos test (kill a replica mid-run), routing benchmark | M5 |
| Agent workloads: tool calling, structured output, MCP | A2–A4, B2 (workload D) | Tool-call eval, SDK round trip, MCP server used from a coding agent, agent-trace replay | M6 |
| Technical writing | P3, `DESIGN.md` | The published write-up | M7 |
| Multi-tenant platform: auth, rate limiting, metering, SQL, Redis, idempotency | R9–R11, E16 | Tests for 429 behaviour and for a ledger that never double-counts under retries | M8 |
| Security: tenant isolation, side channels | E12 | An attack script with before and after results | M3 |
| Containers, Kubernetes, autoscaling | DEP1–DEP2 | One command brings the cluster up on kind; recorded autoscaling run | M9 |
| RAG, vector databases, agent frameworks, guardrails | G1 | Eval set results, prompt-injection test results | Depth track |
| Disciplined use of AI coding tools | The process in section 3 | Pull-request history, decision records, the author's catch log | all |
| Open-source contribution | P5 | A merged upstream pull request | any |

---

## 3. Rules

### 3.1 Precedence

This file says what to build. The other files in `docs/` are background and are not needed to know what to build:

- `Project Architecture Walkthrough.md` is the text version of the author's hand-drawn architecture diagrams (panels S0–S13). It explains *how each mechanism works* with worked examples. Read the relevant panel when you want the picture. Do not read the `.excalidraw` file; the `.md` has the same content.
- `Brigade Project Spec from Claude.md` is the original scope.
- `private/AI SWE Job Skills 2026.md` is the hiring research behind section 2. `docs/private/` is gitignored: it exists only on the author's machine and is never committed.

Where any of them differs from this file, this file wins. Known differences are listed in 5.4.

### 3.2 Engineering rules

Non-negotiable. If a task seems to require breaking one, stop and raise it.

1. **Correctness anchor.** With greedy decoding, uLLM's token IDs equal mlx-lm's reference for: one request; many concurrent requests; batch sizes 1, 2, 4, 8; prefix cache on and off; block-based and contiguous KV; a forced preemption. A divergence fails the test unless the logged top-2 logit margin at the first differing position is under the documented near-tie threshold (decision 0012: none for one machine at batch size 1). Above batch size 1, every token must instead be in mlx-lm's top 2 and within 0.25 logprob of its best, scored by mlx-lm on uLLM's own prefix (decision 0021). Never loosen the test to make it pass.
2. **One owner.** Only the engine thread touches MLX, the scheduler and KV state. No locks in `core`. Threads communicate through queues only.
3. **`ullm.core` never imports `mlx`.** A test enforces it.
4. **MLX is lazy.** Call `mx.eval` at the end of each step. Never record a timing before it.
5. **Never block the engine thread** on telemetry, metrics, logging or the network.
6. **No request ID as a Prometheus label.** Labels are low-cardinality: replica, kind, reason.
7. **Fair benchmarks.** Same model, quantization, chat template, thinking setting, prompts and `max_tokens` for every target. Count real output tokens from `usage`. Follow B5.
8. **No invented numbers.** Every figure in the README, write-up or resume text comes from a committed result. Unmeasured means a placeholder.
9. **Tests ship with behaviour**, in the same pull request.
10. **Stay readable.** The engine is meant to be read. Prefer a clear 30 lines to a clever 10. Keep `core` plus `runner` near 2,500 lines excluding tests and report the real count in the README. Comments explain why, not what.
11. **Build the machinery, do not import it.** Allowed from mlx-lm: `load`, the model modules, `models.base` helpers, the tokenizer wrapper and chat template, detokenizer utilities. Not allowed in engine code: `generate_step`, `stream_generate`, `batch_generate`, `BatchGenerator`, `BatchKVCache`, `LRUPromptCache`. Reading them is encouraged; they are the baseline.
12. **Verify before designing against a library.** Read the installed source in `.venv` for the pinned version. Do not rely on memory of MLX or mlx-lm APIs.
13. **Pin dependencies.** A new dependency needs a one-line justification in the pull request.
14. **Local by default.** Servers bind to 127.0.0.1. Control endpoints are local-only. No secrets in the repo.
15. **Ask first** before anything outward-facing or hard to undo: pushing, publishing the hosted demo, deleting files, force operations.

### 3.3 Definition of done, for every task

- The catalog entry's proof items are met and demonstrated by running them.
- `make check` passes locally and CI is green.
- Engine changes: the equivalence suite passes.
- Performance-relevant changes: before and after `make bench-smoke` numbers are in the pull request.
- Design choices are recorded as a short decision record in `docs/decisions/NNNN-title.md`.
- Section 0 and the task checkbox in section 7 are updated.
- If the task makes a new claim possible, `docs/CLAIMS.md` gains a row (P2).
- The pull request records VERIFIED proof and READY review verdicts with the commit they cover.
- The author gets teach-back notes (3.4).

### 3.4 A session

Tasks run through keel's skills (decision 0007). The author starts `/build` and `/ship` by typing them; an agent cannot. One task per session; stop at the task boundary.

1. Read section 0, the task in section 7, the catalog entries it names, and any contract (6.2) it touches. If section 0 says another task is in flight, or the task's prerequisites are not merged, stop and say so.
2. `[design]` tasks: `/architect`, then the author's approval before any production code. `[spike]` tasks: the measurement plan, then the author's approval before any production code.
3. `/build` the task, in small conventional commits on a branch named like `m2/t3-scheduler`. Write the anchoring tests first where an anchor exists (equivalence, invariants, contract tests). Engine changes: `make equivalence`. Performance-relevant changes: `make bench-smoke` on the base before the first commit, and again after. Outside `/build`, an agent writes no production code for a task; it asks the author to type `/build <ID>`.
4. Once the verdicts are in, on the same branch: add the decision records, update section 0, and tick the task. The tick is the done marker; the roadmap has no `Status:` lines.
5. `/ship` it, in the same chat as `/build`; in a new chat it runs proof and review again. One pull request per task, its body recording the verdicts, merged only when CI is green. No history rewrite after the verdicts. On a stacked branch (decision 0001), `prove`, `vet` and `/ship` take the previous task's branch as the base, not `main`.
6. Finish with **teach-back notes** for the author: up to seven bullets on what was built and why it is shaped that way, three questions an interviewer could ask about it, and anything you were unsure of. Do not write `docs/DESIGN.md`; that file is the author's own work. Fact-check it when asked.

### 3.5 Lanes

- **Lane A, engine (critical path):** M1 → M2 engine tasks → M3 → engine parts of M6 and M8.
- **Lane B, contract-isolated:** terminal work, then M4 (evals), then M5 (router), in its own git worktree, developed against fixture telemetry and `ullm sim`.
- Lane B may not change a contract (6.2) without a pull request that Lane A has seen.
- No third lane. The limit is how much the author can review and understand, not agent throughput.

### 3.6 Changing this roadmap

This file is a plan, not scripture. Change it when that serves the goal in section 1: say so before acting, write a decision record, and update this file in the same pull request. Never deviate silently. Never weaken a rule in 3.2 or a gate.

### 3.7 Standing tasks

- **Upstream candidates.** A bug, doc gap or benchmark issue found in MLX, mlx-lm or a baseline goes into `docs/upstream-candidates.md` (P5).
- **Catch log.** When the author catches an agent mistake, help record it accurately in `docs/AI_LOG.md`. Do not soften it.

---

## 4. Verified facts

Checked on the author's machine against the versions listed. Re-verify after any version change.

**Machine:** MacBook Air M4, 10 cores, 16 GB, macOS 27.2. Installed: Python 3.12.8 and 3.13.7, uv 0.12, Go 1.25, Node 24, pnpm 10, bun 1.2, Docker 28, git 2.50, `gh` 2.104, Homebrew. Not installed: Ollama.

**Libraries:** `mlx==0.32.4`, `mlx-lm==0.32.2` on Python 3.12.

**GitHub's hosted `macos-26` runner** (decision 0005, from the CI probe): 3 virtual M1 cores, 7.5 GB, and a paravirtualized Metal GPU with about 5 GB working set. MLX runs on that GPU by default. A bf16 matmul gave the same result there as on the author's M4 GPU; whole-model greedy agreement is tested by M0-T7. CI timings are meaningless and no benchmark runs in CI.

**Model:** Qwen3-0.6B has 28 layers, 8 KV heads, head dim 128, vocabulary 151,936. In bf16 its KV costs 2 × 28 × 8 × 128 × 2 bytes ≈ 112 KB per token, so a 16-token block is about 1.8 MB and a 4 GB pool is about 2,300 blocks or 36,000 tokens. Weights are about 1.2 GB in bf16. Real memory budget for weights plus KV on this machine: about 10–11 GB.

**Models and goldens** (decision 0006): every model is pinned to a Hugging Face revision in `ullm.models`. For Qwen3-0.6B the only EOS id is 151645, `enable_thinking: false` renders an empty `<think></think>` block, and the tokenizer has 151,669 entries. Golden greedy output has exact bf16 ties (margin 0) at 6 of 495 positions. It reproduces exactly on the author's M4, but diverges at near-ties on the CI runner's GPU. Equivalence therefore compares uLLM against mlx-lm run on the same machine.

**The KV cache contract in mlx-lm 0.32.2** (from `mlx_lm/models/qwen3.py`, `base.py`, `cache.py`):

- The model is called as `model(inputs, cache=[one object per layer])`.
- Per layer, attention does `rope(q, offset=cache.offset)`, `rope(k, offset=cache.offset)`, then `keys, values = cache.update_and_fetch(keys, values)`, then `mx.fast.scaled_dot_product_attention(q, keys, values, scale, mask)`.
- Incoming `keys` and `values` are `[B, n_kv_heads, L, head_dim]`. The return value is the full K and V to attend over.
- `cache.offset` is an int, or a per-row `mx.array` for batches.
- The mask is built once from the first layer's cache: `cache[0].make_mask(N, return_array=..., window_size=...)`. `mlx_lm.models.base.create_causal_mask(N, offset, window_size, right_padding, left_padding)` supports left-padded batches.
- So a custom cache needs `offset`, `update_and_fetch` and `make_mask`. That is the plug point.

**mlx-lm already batches.** `mlx_lm.server` runs a generation thread over a `BatchGenerator` that adds and removes requests between steps. Flags: `--decode-concurrency`, `--prompt-concurrency`, `--prefill-step-size`, `--prompt-cache-size`, `--prompt-cache-bytes`. It also ships `mlx_lm.perplexity`, `mlx_lm.evaluate`, `mlx_lm.benchmark` and tool parsers (`tool_parsers/json_tools.py` handles `<tool_call>` JSON).

**Other engines, for calibration** (from their own documentation, not benchmarked here): [vllm-mlx](https://vllm-mlx.is-a.dev/guides/continuous-batching/) has continuous batching and a paged cache with ref-counted prefix sharing. [vllm-metal](https://docs.vllm.ai/projects/vllm-metal/en/latest/) runs the vLLM scheduler on MLX with an experimental paged-attention path that uses a custom Metal kernel, because MLX's attention has no variable-length interface. [oMLX](https://github.com/ml-explore/mlx/discussions/3203) builds on vllm-mlx.

**KV layout micro-spike.** Standalone MLX operations, each followed by `mx.eval`, bf16, Qwen3-0.6B shapes. Indicative only; not a model benchmark.

| Operation | Measured |
|---|---|
| Write 8 token slots into one per-layer pool array (slice update and index scatter behave the same) | 18.8 MB pool: 1.0 ms · 75 MB pool (≈2,300 blocks): 2.4 ms · 301 MB pool: 6.5 ms |
| Gather blocks into `[B, H, T, D]` for one K of one layer | B=8 × 512 tokens: 0.6 ms · B=8 × 2,048: 1.6 ms · B=16 × 2,048: 2.9 ms |
| One array per block holding all layers, one concatenate per request | 512 tokens: 1.6 ms · 2,048 tokens: 6.3 ms |

Implications: a write costs in proportion to the **pool** size, so one big mutable pool tensor would cost about 2.4 ms × 56 arrays ≈ 130 ms per decode step. Any layout that re-materialises all KV every step pays for the total context every step: about 35 ms at B=8 × 512 tokens and 50–90 ms at B=8 × 2,048, where a whole decode step should take roughly 15 ms. This is why E9 is decided by measurement.

---

## 5. System design

### 5.1 Architecture

```
 Clients: curl · OpenAI SDK · coding agent · MCP client          Terminal (browser)
        │  OpenAI-compatible HTTP + SSE                                ▲ telemetry over SSE
        ▼                                                              │
 Go router and gateway :8000 ──────────────────────────────────────────┘
   prefix-aware routing · least-load fallback · health checks · retries · circuit breaker
   API keys · rate limits (Redis) · usage ledger (Postgres) · telemetry fan-in · /metrics
        │
        ▼
 Engine replicas :8001–8003 (Python processes, native on the Mac)
   API thread (FastAPI, asyncio) ──inbound queue──▶ ENGINE THREAD ──per-request output queues──▶ API thread
                                                     scheduler → block manager → runner (MLX or fake) → sampler
   /metrics · OpenTelemetry spans · /telemetry/stream
        │
        ▼
 Prometheus · Grafana · Jaeger (docker compose)
```

### 5.2 Repo layout

```
uLLM/
├── src/ullm/
│   ├── core/          MLX-free: request.py, scheduler.py, policy.py, block_manager.py, engine.py
│   ├── runner/        base.py (protocol), mlx_runner.py, fake_runner.py, sampler.py, kv/
│   ├── server/        FastAPI app, OpenAI schemas, SSE, tool-call parser, /health, /metrics
│   ├── telemetry/     events.py (schema), bus.py, recorder.py, prometheus.py
│   ├── mcp/           MCP server
│   └── cli.py         ullm serve | sim | ...
├── tests/             unit/ property/ equivalence/ integration/
├── bench/             workloads/ loadgen analysis results/
├── evals/             datasets/ harness results/
├── router/            Go module: cmd/ullm-router, internal/{proxy,health,breaker,routing,telemetry,auth,ratelimit,ledger}
├── terminal/          Vite + React + TypeScript
├── deploy/            docker-compose.yml, dashboards/, helm/
├── scripts/           maintenance scripts outside the package: model fetch, contract export, goldens, git-hooks/
├── docs/              ROADMAP.md, AGENT_ROADMAP.md, DESIGN.md, CLAIMS.md, AI_LOG.md, decisions/
├── .claude/commands/  gate, quiz
├── .github/workflows/
├── Makefile · CLAUDE.md · QUALITY.md · README.md · LICENSE
```

### 5.3 Request path

```
client → router → API thread → inbound queue → ENGINE THREAD → output queue → API thread → SSE → client

ENGINE THREAD, one step, forever:
  1. drain the inbound queue (new tickets, abort tickets); never wait
  2. nothing to do → sleep until a ticket arrives (no busy spin)
  3. scheduler: remove finished · make room for running · admit waiting (FCFS) · choose prefill or decode
  4. runner: one forward pass, sample, mx.eval
  5. push text chunks to output queues; append telemetry events
```

### 5.4 Design decisions

Each of these differs from, or adds to, the older documents.

| ID | Decision | Reason |
|---|---|---|
| DEC-1 | The terminal is a first-class surface with its own charts, not a side view next to Grafana | It is the author's motivation and the most visible proof |
| DEC-2 | Telemetry is a versioned, typed, recordable event stream. The terminal reads live SSE or a recorded file through one interface. A static build with recorded runs is hosted publicly | A demo anyone can open, deterministic UI tests, and the UI can be built before the engine |
| DEC-3 | The engine core is MLX-free and talks to a `Runner` protocol. `FakeRunner` is deterministic with a simulated latency model. `ullm sim` runs the real engine on it | Core logic is tested fast on any machine. The terminal, router and Kubernetes blueprint all run against it |
| DEC-4 | No novelty claims. Other engines are baselines | Section 1 |
| DEC-5 | The scheduler and KV management are written here, not imported (rule 11) | They are the proof of work |
| DEC-6 | The physical KV layout is chosen by a spike on the real model. The logical block manager is the same under every candidate | Section 4 measurements |
| DEC-7 | Prefill one request per step (`[1, L]`, no padding). Decode is batched (`[B, 1]`). A sampled prefill runs inside the runner as `[1, L-1]` then `[1, 1]`, to match mlx-lm (decision 0008) | Simplest design that keeps equivalence easy. Revisit only with data |
| DEC-8 | Headline performance claims are against uLLM's own sequential mode. The gap to mlx-lm's batching is reported and explained | mlx-lm's server already batches |
| DEC-9 | Cheap hooks from the start: a `PREFILLING` state with progress (for chunked prefill), a salt in the prefix hash chain (tenant isolation), a policy interface (fair share), an optional tenant on every request, trace context on every ticket, a versioned telemetry schema | Minutes now, a refactor later |
| DEC-10 | Telemetry uses SSE. Controls use ordinary POST requests | One-way data, plain HTTP, passes through the proxy |
| DEC-11 | Once the router exists, the terminal talks only to the router, which merges the replicas' telemetry | One origin for the browser, and a real Go concurrency piece |
| DEC-12 | Workload D is record-and-replay: capture real agent sessions through the router, replay them verbatim | A very small model cannot drive a coding agent well enough to generate realistic traces |
| DEC-13 | The router fingerprints message text, not token blocks | The router has no tokenizer. Same chained-hash idea, coarser grain |

### 5.5 Ports

Engine alone: 8000. With the router: router 8000, replicas 8001–8003. Terminal dev server 5173, Prometheus 9090, Grafana 3000, Jaeger 16686, Redis 6379, Postgres 5432.

---

## 6. Build catalog

Every piece of the system. Each entry says what it does and what proves it. Milestone tasks in section 7 refer to these IDs.

### 6.1 Foundations

**F1 Repo and tooling.** Monorepo per 5.2. Python 3.12 managed by uv, `mlx` and `mlx-lm` pinned to section 4. ruff, a type checker (strict on `core` first), pytest, hypothesis. Go module for the router. pnpm for the terminal. `Makefile` targets: `check` (lint, types, unit and property tests), `test`, `equivalence`, `bench-smoke`, `sim`, `up` (router plus replicas), `verify` (P2).

**F2 CI.** GitHub Actions. Jobs: `core` (Linux: lint, types, tests with `FakeRunner`, no MLX), `mlx` (macOS arm64: equivalence subset on Qwen3-0.6B, about 32 tokens per prompt), `router` (Go vet, test, chaos test against `ullm sim`), `terminal` (typecheck, lint, test, build), later `evals` (Q7). All required on `main`. First task: find out whether MLX runs on the hosted macOS runner, on GPU or CPU, and record the answer.

**F3 Agent workflow.** `CLAUDE.md` under 60 lines: what the project is, a pointer to this file, the commands, the rules in brief. Project commands in `.claude/commands/`: `gate` (run a milestone's gate checklist and report each item as passed or failed with evidence), `quiz` (ten interview questions on a milestone, then grade the author's answers against the code). Tasks run through keel's `/architect`, `/build` and `/ship` (3.4, decision 0007).

**F4 Models and goldens.** Confirm the exact mlx-community repo ids for Qwen3-0.6B in bf16, 8-bit and 4-bit, and Qwen3-1.7B. A fetch script. Golden fixtures: token IDs from mlx-lm greedy generation over a fixed prompt set, stored with the library versions that produced them and regenerated on a version change. Thinking mode is off for all goldens, evals and benchmarks.

**F5 Docs skeleton.** `README.md` (work in progress until M2), `docs/DESIGN.md` (headings only; the author writes it), `docs/CLAIMS.md`, `docs/AI_LOG.md`, `docs/upstream-candidates.md`, `docs/decisions/`.

### 6.2 Contracts

Freeze these early. Lanes depend on them and nothing else.

**C1 OpenAI-compatible API.**

- `POST /v1/chat/completions`. Fields: `model`, `messages`, `stream`, `stream_options.include_usage`, `max_tokens` and `max_completion_tokens`, `temperature`, `top_p`, `stop`, `seed`. Extensions: `top_k`, `chat_template_kwargs` (for `enable_thinking`). Added later: `tools`, `tool_choice` (A2), `response_format` (A3), `cache_salt` (E12).
- Non-streaming reply: one JSON with `choices[0].message`, `finish_reason` and `usage`.
- Streaming reply: SSE events in this order: a delta with the role, one content delta per decoded chunk, a final delta with `finish_reason`, a usage chunk with empty `choices` when `include_usage` is set, then `data: [DONE]`.
- `usage` is exact, counted by the engine. `prompt_tokens_details.cached_tokens` reports prefix-cache hits from M3.
- Errors use OpenAI's error format: 400 (missing messages, bad values, prompt longer than the context, unsupported options such as `n > 1`), 401 (bad API key, from M8), 404 (unknown model), 429 with `Retry-After` (rate limit, from M8), 503 with `Retry-After` (queue full). A mid-stream failure sends an error event and closes.
- Also `GET /v1/models`, `GET /health`, `GET /metrics`, `GET /telemetry/stream`.
- The only change a client needs is `base_url`.

**C2 Telemetry protocol v1.**

- Frame: `{"v": 1, "replica": "r1", "seq": n, "t": <monotonic ms>, "events": [...]}`, flushed about 15 times a second.
- Events: `hello` (model, block size, block count, limits), `req.arrived`, `req.admitted`, `req.first_token`, `req.finished` (reason, usage, timings), `req.preempted`, `req.aborted`, `step` (kind prefill or decode, row request ids, tokens, ms, queue depth), `kv.snapshot` (one byte per block, compactly encoded), `router.decision` (from M5).
- Defined once as pydantic models in `ullm/telemetry/events.py`. JSON Schema is exported and TypeScript types are generated from it. CI fails when the generated types are stale.
- Recorded as compressed JSONL. The terminal's replay mode and its tests read the same format.

**C3 Runner protocol.** The scheduler produces a step plan: either prefill given token ranges for one sequence, or decode one token for a set of sequences. The runner executes exactly that, samples, calls `mx.eval`, and returns one token per decoding row plus timings taken after the eval. The runner owns every tensor. The core owns every decision. Exact Python signatures are settled in M1-T1 and recorded.

**C4 Engine and router.** `/health` returns 200 only while the engine thread is alive, with a small JSON load summary (in flight, waiting, free blocks). 503 with `Retry-After` on a full queue. `traceparent` is propagated and carried on the ticket across the thread boundary. `X-Request-Id` is echoed. From M8 the router passes a tenant id header.

**C5 Benchmark results.** One directory per run under `bench/results/`: `config.json`, `env.json` (commit, machine, library versions, power state), `raw.jsonl` (one line per request with client-side timings and `usage`), a `/metrics` snapshot, the `powermetrics` log, a generated `summary.md`. README tables are generated from these by script and never typed by hand.

### 6.3 Engine

**E1 Runner.**

- `MLXRunner`: loads once at startup with `mlx_lm.load` (config, safetensors weights, tokenizer and chat template). mlx-lm's model modules are used unchanged, with our own per-layer cache objects on the plug point in section 4. Prefill input is `[1, L]`, decode input is `[B, 1]`; a sampled prefill runs as `[1, L-1]` then `[1, 1]` (decision 0008). Logits `[B, L, vocab]` are reduced to the last position `[B, vocab]` for the sampler.
- `FakeRunner`: deterministic tokens from a seed and the request id. Sleeps by a latency model (prefill = base + per token, decode = base + per row). Can inject failures for router tests.
- Chat template: messages become one string in the model's trained format, then token IDs. A wrong template is a silent quality loss, so the rendered template has a golden test.
- *Proof:* Q3.

**E2 Sampler.** Per row: greedy at temperature 0, otherwise temperature, top-k and top-p. A seeded random state per request, so a given seed reproduces. Runs on `[B, vocab]` logits. *Proof:* unit tests for greedy equals argmax, top-k and top-p masks, seed determinism.

**E3 Stop check and streaming detokenizer.** Stop on the model's end tokens (`finish_reason: "stop"`), on `max_tokens` (`"length"`), and on stop strings: text that could be the start of a stop string is held back, and the stop string itself is never emitted. One streaming detokenizer per request, which holds back incomplete characters. *Proof:* unit tests including multi-byte characters and stop strings split across tokens.

**E4 Request and state machine.** A request holds: id, prompt token IDs, sampling settings, `max_tokens`, stop strings, its output queue, optional tenant and cache salt, trace context, timestamps (arrival, admitted, first token, finished), the tokens generated so far, and its block table, which the block manager holds for it (decision 0032). States: `WAITING → PREFILLING → RUNNING → FINISHED`; `RUNNING → PREEMPTED → WAITING` (at the front); any state `→ ABORTED`. Illegal transitions raise. Each transition emits a telemetry event. `PREFILLING` carries progress so a prompt can be prefilled across several steps later (E13).

**E5 Engine loop and thread bridge.**

- Two threads: the asyncio API thread and one engine thread. A decode step blocks for milliseconds and a prefill for much longer, so the model cannot run on the asyncio thread.
- Inbound queue: thread-safe and unbounded, because the engine drains it into the waiting queue every step (decision 0010). Tickets are a new request (id, prompt IDs, settings, trace context), an abort or a shutdown. The API side bounds open requests (submitted and not yet ended) at `max_in_flight`; beyond it the API replies 503.
- Waiting queue: the scheduler's own list, touched only by the engine thread. Two queues exist so no scheduler state is shared between threads.
- Output: one queue per request. The engine pushes chunks to the API thread's event loop thread-safely; the handler sleeps on its queue in between.
- Loop: as in 5.3. Idle means blocking on the inbound queue, never spinning.
- A client disconnect makes the API send an abort ticket; the engine frees that request's resources at the next step.
- If the engine thread dies, `/health` fails, the error is logged and in-flight requests receive an error event. Shutdown stops admitting, then finishes or aborts what is running.
- *Metrics:* queue depth, time in queue, idle share, step time.
- *Proof (on `FakeRunner`):* 50 concurrent clients all finish, no stream receives another's tokens, every abort releases its resources, `/health` stays fast during a long step.

**E6 Scheduler.** One step does, in order:

1. Remove finished requests: release their blocks, send the final event.
2. Make room for running requests: each will write one token, so its last block needs a free slot. From M3: allocate a block when the last is full; if the pool is empty, reuse a Cached block; if none, preempt (E11).
3. Admit waiting requests oldest first while they fit. Stop at the first that does not fit; nobody behind it may skip ahead.
4. Choose the step type: if a request was admitted, this is a prefill step for it and running requests wait one step; otherwise it is a decode step for every running request.
5. Hand the plan to the runner (C3).

- Admission limits, checked per waiting request: `max_num_seqs` (start at 16), a prefill token budget per step (start at 2,048; it takes effect with chunked prefill, E13: while a step prefills one whole prompt, the budget is `max_model_len`, decision 0010), and from M3 enough free blocks for the prompt after prefix hits plus a small safety watermark. These are starting values to tune from the concurrency sweep.
- *Until preemption (M3-T6), decision 0035:* each admitted request reserves the blocks for `prompt + max_tokens`, beside decision 0020's padded slot bound, because a decode that finds the pool empty has no legal move without E11. M3-T6 replaces the reservation with the prompt-plus-watermark rule above.
- The ordering policy is an interface: FCFS now, fair share later (E16).
- Known trade-off: a long prefill stalls every decoder for that step (visible on workload C). E13 fixes it.
- *Metrics:* waiting queue length, batch size per step, step time by kind, tokens per step.
- *Proof:* every request finishes, FCFS order holds, nothing leaks after a run, equivalence at batch 1, 2, 4, 8 (Q3).

**E7 Contiguous batch KV.** Our own per-layer cache for the running batch: K and V as `[B, H, T, D]`, left-padded, with per-row offsets for RoPE and a mask built with `create_causal_mask(..., left_padding=...)`. A request joins at its first decode (its KV is copied in) and leaves when it is released (the batch is repacked without its row, so no left padding is common to every row). Membership changes copy; steps write in place (decision 0019). This is the KV for M2 and the reference that E9 is measured against.

**E8 Block manager** (logical, MLX-free).

- A fixed pool of blocks sized at startup from a memory budget. Block size 16 tokens, configurable.
- A free list, one block table per request, and a ref count per block.
- Block states: **Free**; **In use** (ref count ≥ 1, never evicted); **Cached** (ref count 0, contents and hash kept, reusable on a later prefix hit, evicted least-recently-used when the pool runs short).
- Operations: can-fit check, allocate N, append one slot, share an existing block, free a request's blocks, evict.
- Life cycle: prefill takes ceil(len / 16) blocks; decode fills the last block and takes one more when it is full; finishing returns all of them.
- Waste is at most 15 slots per request (its last block only).
- *Metrics:* KV utilisation, waste %, free blocks, max concurrent sequences.
- *Proof:* Q2 invariants.

**E9 KV store** (physical layout) `[spike]`. Candidates:

- **A. Gather every step.** Fixed blocks, block tables, K and V gathered into a padded batch tensor each step (per-block arrays, never one mutable pool). The vLLM-faithful design.
- **B. Gather on membership change.** The running batch keeps a contiguous working tensor as in E7. Blocks are the store of record for sharing, caching and accounting. A request's KV is gathered once when it joins or resumes and written through as blocks fill.
- **C. Segmented with decomposed attention.** Immutable shared-prefix segments plus private tails; attention is computed per segment and merged, so a shared prefix is read once for the whole batch and never copied (the idea in the Hydragen paper, Juravsky et al.). Needs our own attention call for Qwen3.

The spike measures, on the real model, decode step time against batch size and context length for each candidate and for E7, plus the memory each really uses. *Decision rule:* in order of simplicity (A, then B), take the first whose decode step time at B=8 with 1k-token contexts is within about 20% of E7. C is prototyped only far enough to estimate its gain. If neither A nor B qualifies, stop and discuss with the author. Whatever is chosen, report A's measured cost as the "gather tax". The result is implemented behind a `KVStore` interface. *Proof:* equivalence against E7.

*Decided (decision 0033, M3-T2):* B. At B=8 and 1k tokens, B is 0.91 of E7 and A is 3.56 (the gather tax). At B=8 and 1k tokens, B holds 1.78 times E7's KV memory, because the working tensor and the blocks each hold a copy. C's split attention is slower than the fused kernel with MLX's own operations (`bench/results/m3/kv-layout/`).

**E10 Prefix caching.**

- At admission, split the prompt's token IDs into 16-token blocks and chain-hash each **full** block: `h_i = hash(h_{i-1}, tokens_i)`, with the chain root seeded by the cache salt. Chaining is required because the same tokens after a different history have different K and V.
- Look each hash up in the table (hash → block). A hit reuses the block: ref count +1, added to the request's block table. The first miss stops matching. Only the remaining tokens are prefilled, into new blocks.
- Register the hashes of newly filled blocks so later requests can hit them. A partial last block is never hashed or shared. Shared blocks are full and therefore read-only; new tokens go into the request's own last block.
- On finish the ref count drops; at 0 the block becomes Cached.
- If every block hits, still recompute the last prompt token to get next-token logits.
- Report hits as `usage.prompt_tokens_details.cached_tokens`.
- Document the gotcha: one changed early token, such as a timestamp, gives zero hits.
- *Metrics:* prefix hit rate (hit tokens / prompt tokens), evictions.
- *Proof:* equivalence with the cache on and off; TTFT with the cache on and off on workloads B and D.

**E11 Preemption.** Admission only checks that the prompt fits, so memory can run out mid-generation. When a running request needs a block, the pool is empty and nothing is Cached: pause the **newest** running request (least work to redo), release its blocks (its full blocks stay Cached, so its re-prefill usually hits the prefix cache), and put it at the **front** of the waiting queue. When re-admitted it re-prefills its prompt plus the tokens it already sent, then continues with its sampler state intact. The client sees a pause and no repeated text. There is no swap to CPU; unified memory is the same RAM. *Metric:* preemptions. Many preemptions mean the pool is too small or the batch cap too high. *Proof:* a forced preemption still matches the golden output.

**E12 Cache isolation and the timing side channel.** A `cache_salt` request field seeds the hash chain, so requests with different salts never share blocks. From M8 the salt is derived from the API key. The demonstration: a script that, without salting, tells whether another tenant has sent a given prefix by measuring TTFT (report how reliably), and that falls to chance with salting. Document the cost: no sharing across tenants. *Proof:* the script's before and after results, committed.

**E16 Fair-share scheduling.** A policy implementation for the E6 interface that orders admission by how much service each tenant has received, so one heavy tenant cannot starve the rest. *Proof:* a mixed-tenant workload showing bounded wait for the light tenant, and a no-starvation property test. (Built in M8. E13–E15 are in section 8.)

### 6.4 API

**A1 API server.** FastAPI, uvicorn, pydantic. Implements C1. Maps fields: `messages` → chat template, `max_tokens` and `stop` → stop check, `temperature`, `top_p`, `top_k`, `seed` → sampler, `stream` → reply mode. The model name must be the loaded one. Detects client disconnects and sends abort tickets. CORS only for the terminal's origin. *Proof:* the OpenAI Python SDK passes against it, streaming and not; SSE framing unit tests; error-format tests.

**A2 Tool calling** `[design]`.

- Accept `tools` and `tool_choice`. `auto` (default) lets the model decide, `none` ignores tools. `required` or a named tool returns 400 until A3 exists, then is enforced through it.
- The chat template writes the tool list into the prompt. The model emits `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`. The engine never runs tools; the client does and sends the result back as a `tool` message.
- Parser: text → OpenAI `tool_calls` (`id`, `type: "function"`, `function.name`, `function.arguments` as a JSON **string**). Several blocks give several calls. `finish_reason: "tool_calls"`.
- Streaming: normal text flows as usual. From `<tool_call>` on, text is held back; at `</tool_call>` one `tool_calls` event is sent. Raw tags never reach the client.
- Malformed output (broken JSON, unknown tool name, missing required argument, tag never closed): never crash. Return the text as ordinary content and count the failure.
- Agents resend the whole history each round, so round N shares its prefix with round N−1 and E10 keeps TTFT low.
- *Proof:* parser unit tests on good and broken samples, an SDK round trip with a fake tool, Q6.

**A3 Structured output** `[design]`. `response_format` with `json_object` or a JSON schema, enforced by an existing constrained-decoding library used as a logits processor in the sampler path, with per-row constraint state in a batch. Check candidate libraries against the pinned MLX before choosing and record the choice. Unlocks `tool_choice: "required"` and named tools. *Proof:* valid-against-schema rate on an eval set, and the measured throughput cost.

**A4 MCP server.** A small Model Context Protocol server over the engine and router. Read tools: current metrics, active and recent requests, KV statistics, configuration, latest benchmark summary. Control tools (local only): switch scheduling policy, change admission limits. Usable from a coding agent such as Claude Code. *Proof:* an MCP client test in CI that lists tools and calls one; a recorded session of an agent using it.

### 6.5 Telemetry and observability

**O1 Telemetry bus, stream and recorder.** Implements C2. The engine thread appends small event objects to a bounded buffer that drops the oldest on overflow and counts drops. The API thread serialises and flushes frames about 15 times a second on `/telemetry/stream`. A KV snapshot is one byte per block and is throttled. A closed tab costs nothing. The recorder writes any run to a trace file. *Proof:* throughput with telemetry on and off, reported as overhead; a schema round-trip test; recordings pass the trace checker `ullm.telemetry.check` (decision 0004).

**O2 Prometheus metrics.** `/metrics` on every engine and on the router.

- Engine histograms: `ttft_seconds`, `tpot_seconds`, `e2e_seconds`, `step_seconds{kind}`, `queue_wait_seconds`.
- Engine gauges: `queue_depth`, `batch_size`, `kv_blocks{state}`, `kv_waste_ratio`.
- Engine counters: `prompt_tokens_total`, `generation_tokens_total`, `prefix_cache_hit_tokens_total`, `requests_total{finish_reason}`, `preemptions_total`, `aborts_total`, `telemetry_dropped_total`.
- Router: `requests_total{replica}`, `routing_decisions_total{kind}` (prefix, least-load, round-robin), `retries_total`, `upstream_errors_total`, `replica_healthy`, `breaker_state`, `rate_limited_total`.
- Updating a metric is a counter bump. Percentiles come from histograms, never averages.
- *Proof:* counters equal the load generator's totals after a run.

**O3 Grafana.** `deploy/docker-compose.yml` with Prometheus scraping the host every 5 seconds and Grafana with one provisioned dashboard: throughput, TTFT and TPOT p50/p99, batch size, queue depth, KV usage, prefix hit rate, preemptions, load per replica, breaker state. Engines never run in Docker on the Mac: the Linux VM cannot reach the Metal GPU.

**O4 Tracing.** OpenTelemetry in the router and the engine, Jaeger in compose. One trace per request with a span per phase: `router.request`, `router.forward`, `api.request`, `queue.wait`, `engine.prefill`, `engine.decode` (token count as an attribute). Never a span per token. `traceparent` carries the trace from Go into Python and the ticket carries it across threads. Cross-check server TTFT against client TTFT; a large gap is a bug or a queue. *Proof:* a test asserting one connected trace with the expected spans.

### 6.6 Terminal

Stack: Vite, React, TypeScript (strict), pnpm, Vitest, Playwright. Canvas 2D for the step chart, the batch strip and the KV grid, with no chart library (decision 0022). A virtualised table for requests. Look: dark, dense, tabular numerals, a monospace face for figures. It uses the layout grammar of trading terminals with its own identity; never copy a product's branding, colours or assets.

**T1 Shell, store and sources.** The dense grid layout and design tokens. An external store with ring buffers; telemetry never passes through React state per event. Canvases draw on `requestAnimationFrame`; React renders layout and controls. A `TelemetrySource` interface with two implementations: live SSE and replay from a trace file.

**T2 Core panels.**

- Stats ticker: tokens per second, TTFT p50/p99, TPOT, batch size against its cap, queue depth, KV used.
- Request feed: id, state pill (waiting, prefilling, running, preempted, finished), tokens, TTFT.
- Step chart: one candle per second of engine step latency (open, high, low, close; a long wick is a stall), with two bar graphs underneath, processed and sampled tokens per second, prefill and decode coloured differently.
- Batch strip: rows are requests, columns are engine steps, each cell a prefill or a decode token. Shows requests joining and leaving, and the prefill stall.
- Requests table with tabs (Running, Waiting, Finished, Preempted): prompt tokens, generated tokens, TTFT, TPOT, blocks, cache-hit %, age.

**T3 Load launcher and playground.** The launcher picks a workload, concurrency or arrival rate, and `max_tokens`, then calls a local-only control endpoint that starts the Python load generator (a browser cannot hold 16 streams to one origin). The playground sends one prompt and streams the reply.

**T4 Replay and hosted build.** Replay mode with play, pause, speed and a scrubber, and a run picker in place of the launcher. A static build bundled with recorded runs, published at a public URL (DEP3) and always labelled as a replay, with the machine and commit of the recording.

**T5 KV block grid.** One square per block from `kv.snapshot`: owned (coloured by request), shared prefix (ref count above 1), Cached, free. Hover shows the owner. Selecting a request highlights its blocks. Adds the preempted state and a cache-hit column to T2.

**T6 Replicas and routing.** A replica list with health and load, and a feed of routing decisions (prefix hit, least-load, retry, breaker).

**T7 Quality.** Budgets: 60 fps while receiving 15 frames a second with about 2,300 blocks and 200 table rows; flat memory over a long session; the hosted replay becomes interactive quickly. *Proof:* a frame-rate test, a Playwright smoke test driven by a replay trace, keyboard basics.

**T8 Tenants view.** Per-key usage, rate-limit hits and share of service (with M8).

### 6.7 Router and gateway

Go, standard library first (`net/http`, `httputil.ReverseProxy` or a small streaming proxy). A separate process, so it stays up when an engine crashes. The replicas share one GPU on this machine, so the router demonstrates routing and failover, not extra throughput; say so wherever results are shown.

**R1 Proxy.** Forwards OpenAI requests and streams SSE back, flushing every chunk. Timeouts (starting values): connect about 1 s, first byte about 30 s, gap between chunks about 30 s, no total timeout because a long answer is not a failure. **Retry rule:** retry on another replica only before the first byte has reached the client (connect failure or 503). After that a failure becomes an error event. When the client leaves, cancel the upstream request so the engine aborts.

**R2 Health checks.** `GET /health` on each replica about every 2 s. Three misses mark it down and stop routing to it. A pass brings it back.

**R3 Circuit breaker.** Per replica. Closed → open after 5 consecutive failures → half-open after a cool-down of about 10 s, letting one trial through → closed on success, open again on failure. Health checks catch a dead process; the breaker catches a replica that passes `/health` but fails real requests.

**R4 Load-aware selection.** Least-loaded healthy replica by the router's own in-flight count.

**R5 Prefix-aware routing.** Fingerprint each request with a chained hash per message over its text (and the tenant, from M8). Keep an LRU-capped table from hash to replica. Route to the replica owning the longest matching prefix if it is healthy and not overloaded; otherwise fall back to R4. Record the request's hashes against the replica chosen. Turn N of a conversation contains turn N−1, so it lands on the replica that already holds its cache. The overload check stops one popular system prompt piling onto one replica.

**R6 Telemetry fan-in.** Subscribe to every replica's telemetry stream and merge into one `/telemetry/stream`, tagging frames with the replica and dropping the oldest under backpressure.

**R7 Record mode.** An opt-in flag that writes request bodies and arrival offsets to a trace file for workload D. Local only; the file contains prompt text.

**R8 Router metrics.** As listed in O2.

**R9 API keys.** `Authorization: Bearer <key>`. Keys are stored hashed in Postgres with a tenant and a tier. A CLI creates and revokes keys. A bad key gets 401. The tenant id is passed to the engine, where it seeds the cache salt (E12) and feeds fair share (E16).

**R10 Rate limiting.** Per-key token buckets in Redis, updated atomically, for requests and for tokens. Over the limit returns 429 with `Retry-After`. Decide and document what happens when Redis is unreachable.

**R11 Usage ledger.** A Postgres table keyed by request ID with a uniqueness constraint, written from the engine's exact `usage` when a stream completes. Writes are idempotent, so a retry or a replayed write never double-counts. Schema migrations are versioned. A per-key usage endpoint reads it. *Proof:* tests that inject retries and a crash between response and write, then check the totals.

*Proof for the router as a whole:* unit tests for breaker transitions and routing decisions; a chaos test that kills a replica mid-run and checks that requests not yet streaming succeed elsewhere and mid-stream requests end with a clean error event.

### 6.8 Tests and evals

Tests ask whether the code is correct: pass or fail, the same every run. Evals ask whether the output is good enough: a score against a threshold.

**Q1 Unit tests.** Sampler, stop check, detokenizer, chat template, SSE framing, tool-call parser, hashing, breaker, routing.

**Q2 Property tests.** Block manager, as a stateful machine over random sequences of allocate, share, append, preempt, free and evict, checking after every operation: no block is owned twice; ref counts never go negative; free + used + cached equals the total; a block with ref count ≥ 1 is never free or evicted; freed blocks are reusable. Scheduler on `FakeRunner`: every request finishes; FCFS order holds; nothing leaks.

**Q3 Output equivalence.** The correctness anchor in rule 1. The same prompts go through mlx-lm and through uLLM with greedy decoding, on the same machine in the same session; token IDs must match. Committed goldens are not a strict cross-machine reference (decision 0006). Variants: one request, many at once, batch sizes, prefix cache on and off, block-based against contiguous, a forced preemption. On a mismatch, log the top-2 margin; a near-tie under the threshold of decision 0012 is recorded, anything else fails. Batch sizes above 1 score every token on uLLM's own prefix instead (decision 0021).

**Q4 Integration.** The OpenAI SDK contract test, 50 concurrent clients, the router chaos test, an MCP client test.

**Q5 Quantization evals.** bf16, 8-bit and 4-bit (pre-quantized weights, no training) compared on: perplexity over a fixed text, exact match on a 100–200 item task set with one right answer each, tool-call validity (Q6), tokens per second, and memory. Datasets are checked into the repo under licenses that allow it, with attribution. Greedy decoding, one template and thinking setting. Never tune prompts on the eval set.

**Q6 Output-format evals.** Tool calls: the share that parse and match the declared schema. Structured output: the share valid against the schema.

**Q7 CI eval gate.** A subset of about 20 items runs in CI against a threshold of baseline minus a small noise margin, so a real quality drop turns CI red. Full evals run on the author's machine.

### 6.9 Benchmarks

**B1 Harness.** Seeded workload files → a load generator (asyncio and httpx) → any target that speaks the OpenAI API → results per C5 → an analysis script that writes tables and charts. The client times every token.

**B2 Workloads.**

| | Shape | Tests |
|---|---|---|
| A, chat | Prompts of 100–1k tokens, outputs of 50–300 | Throughput, batching |
| B, shared prefix | 4–8 concurrent requests sharing a 2–4k-token system prompt | Prefix caching, routing |
| C, mixed | One 8k-token prompt among short chats | Latency fairness, the prefill stall |
| D, agent traces | Recorded multi-turn agent sessions, replayed verbatim | Per-turn prefix hit rate and TTFT |

**B3 Load modes and metrics.** Closed loop at concurrency 1, 2, 4, 8, 16. Open loop with Poisson arrivals at fixed rates, to find the knee where TTFT takes off. Per request: TTFT, TPOT = (last − first) / (tokens − 1), end-to-end latency. Per run: output tokens per second, p50 and p99, goodput (the share of requests meeting stated TTFT and TPOT targets), and from the server snapshot KV utilisation, waste %, prefix hit rate and max concurrent sequences. Optional: energy per thousand tokens from the `powermetrics` log.

**B4 Baselines.** uLLM in sequential mode; `mlx_lm.server` configured sequential and in its default batched mode (work out and document the flags); vllm-metal (attempt it, and if setup fails document what was tried); vllm-mlx; Ollama (it cannot load MLX weights, so use the closest quantization and say so). Same model and quantization everywhere else.

**B5 Run conditions.** Plugged in, Low Power Mode off, other apps closed, the compose stack stopped, GPU frequency and power logged with `powermetrics`. Engines take turns on the same prompts in one session, with a twin of the baseline as the noise floor and a discarded warm-up before every turn; blocks are sorted into Peak and Sustained from the clock log afterwards, and each figure states its regime (decision 0016, which replaced cooldowns and the median of three on the fanless Mac). Long sessions run unattended.

**B6 Reporting.** uLLM will likely be slower than mlx-lm and vllm-metal in places. Report it, and explain each gap with our own measurements. Every table states its conditions.

### 6.10 Deployment

**DEP1 Container images.** The router, the engine in `sim` mode, and the static terminal. (The MLX engine stays native.)

**DEP2 Kubernetes blueprint.** A kind cluster on the Mac. A Helm chart for the router and `sim` replicas, Redis and Postgres. Autoscaling of replicas on queue depth from Prometheus. Always described as a deployment blueprint with a simulated model backend, not GPU serving. *Proof:* one command brings it up; a recorded load run shows replicas scaling.

**DEP3 Hosted replay.** The static terminal build with recorded runs on a public URL. Publishing needs the author's approval each time.

### 6.11 Proof and release

**P1 README.** Demo GIF and replay link at the top, what it is in two sentences, the architecture diagram, generated benchmark tables with their conditions, a "verify it yourself" section, quickstart, a guided tour of the code, honest limitations.

**P2 Claims ledger.** `docs/CLAIMS.md`: one row per claim the project makes, with the evidence (test name, results directory, script, link) and how to reproduce it. `make verify` runs the equivalence subset, the property tests and a benchmark smoke run on any Apple Silicon Mac. No claim appears in the README or on a resume without a row here.

**P3 Write-up.** One article: what was built, the numbers, the reason behind every gap, what the KV spike showed, what the author would do differently.

**P4 Releases.** A tag and short changelog at the end of each stage (section 7).

**P5 Upstream contribution.** One small merged pull request to MLX, mlx-lm or a baseline, drawn from `docs/upstream-candidates.md`.

**P6 Author's artifacts.** `docs/DESIGN.md` in the author's words, one section per milestone. `docs/AI_LOG.md`. The gate quiz. Agents support these and do not write them.

### 6.12 Example agent (depth track)

**G1 Ops agent.** A small agent in its own directory, outside the engine, that answers questions about a running uLLM: tools through the MCP server (A4), retrieval over the project's docs (pgvector), built with LangGraph, served by uLLM itself or any OpenAI-compatible endpoint. A 30–50 question eval set with pass criteria. A prompt-injection test: tool results containing injected instructions, the share the agent obeys, a mitigation, and the share afterwards.

---

## 7. Milestones

A milestone is done when its gate passes. Tags: `[design]` needs approval before code, `[spike]` is a bounded measurement, `[user]` needs the author, `[B]` can run in Lane B.

### Stage 1: an engine you can watch

#### M0 Foundations

- [x] **M0-T1** Repo bootstrap (F1). `git init` here, add the remote, bring in its initial commit, lay out 5.2, `Makefile`, `.gitignore`, README stub. Leave the existing files in `docs/` in place.
- [x] **M0-T2** `CLAUDE.md` and the workflow commands (F3).
- [x] **M0-T3** Python tooling, pinned (F1).
- [x] **M0-T4** Contracts v0 `[design]` (C1, C2, C3): pydantic models, schema export, TypeScript generation with a staleness check.
- [x] **M0-T5** Telemetry fixtures (O1): a seeded generator and two committed sample traces, so Lane B can start with no engine.
- [x] **M0-T6** CI (F2).
- [x] **M0-T7** Models and goldens (F4).
- [x] **M0-T8** Docs skeleton (F5).
- [x] **M0-T9** `[user]` Install and log in to `gh`.

**Gate G0:** `make check` and all CI jobs green (or the `mlx` job's limitation documented with its fallback); contracts merged; fixtures available.

#### M1 One request, end to end

- [x] **M1-T1** `[design]` Runner protocol, `MLXRunner` with our own generation loop and a contiguous cache, `FakeRunner` (E1, C3).
- [x] **M1-T2** Sampler, stop check, detokenizer (E2, E3).
- [x] **M1-T3** `[design]` Request state machine, engine loop, thread bridge, scheduler interface at `max_num_seqs = 1`, `ullm sim` (E4, E5, E6).
- [x] **M1-T4** API server (A1).
- [x] **M1-T5** Tests: unit (Q1), single-request equivalence (Q3), SDK contract and 50 clients (Q4).
- [x] **M1-T6** Benchmark harness, workload A, run conditions (B1, B2, B5).
- [x] **M1-T7** First baselines (B4): uLLM sequential, `mlx_lm.server` sequential and batched.
- [x] **M1-T8** `[B]` Terminal shell, ticker and feed on fixture telemetry (T1, part of T2).

**Gate G1:** equivalence and SDK contract tests green in CI; baseline results committed; `curl` streams from the real model.

#### M2 Continuous batching and terminal v1

- [x] **M2-T1** `[design]` Contiguous batch KV (E7).
- [x] **M2-T2** Scheduler: continuous batching, admission limits, FCFS (E6).
- [x] **M2-T3** Batching tests: equivalence at batch 1, 2, 4, 8; scheduler properties (Q2, Q3).
- [x] **M2-T4** Telemetry bus, stream, recorder, measured overhead (O1).
- [x] **M2-T5** Prometheus metrics (O2).
- [x] **M2-T6** Benchmark sweep on workload A at concurrency 1–16 against the baselines, with an explanation of every gap (B3, B6).
- [x] **M2-T7** `[B]` Step chart, batch strip, requests table (T2).
- [x] **M2-T8** `[B]` Load launcher and playground (T3).
- [x] **M2-T9** `[B]` Replay mode and the hosted build (T4, DEP3).
- [x] **M2-T10** `[B]` Terminal quality gates (T7).
- [x] **M2-T11** Grafana dashboard (O3).
- [x] **M2-T12** README v1, claims ledger, `make verify` (P1, P2).
- [ ] **M2-T13** `[user]` DESIGN.md sections for M1 and M2, the quiz, resume lines filled from results (P6).

**Gate A (ready to apply; tag `v0.1`):** public repo with green CI; batched greedy output matches mlx-lm; the benchmark table is generated from committed runs; the terminal works live and the hosted replay opens in a clean browser; every README claim has a ledger row; the author passes the quiz.

### Stage 2: memory and quality

#### M3 Blocks, prefix caching, preemption

- [x] **M3-T1** Block manager with stateful property tests (E8, Q2).
- [x] **M3-T2** `[spike]` KV layout on the real model, decision recorded (E9).
- [x] **M3-T3** KV store behind its interface; equivalence against contiguous (E9, Q3).
- [ ] **M3-T4** Admission by free blocks; memory metrics, and their KV usage panel in the Grafana dashboard (E6, E8, O3).
- [ ] **M3-T5** Prefix caching; equivalence on and off; `cached_tokens` (E10).
- [ ] **M3-T6** Preemption; forced-preemption equivalence (E11).
- [ ] **M3-T7** Cache salt and the timing side-channel demonstration (E12).
- [ ] **M3-T8** `[B]` KV block grid; new recorded runs for workloads B and C (T5).
- [ ] **M3-T9** Benchmarks: workload B with the cache on and off, workload C's stall, memory waste, the gather-tax report (B2, B3).

**Gate G3:** property tests and every equivalence variant green; prefix-cache effect on workload B committed; the layout decision recorded and explainable by the author.

#### M4 Evals `[B]`

- [ ] **M4-T1** Eval harness and datasets (Q5).
- [ ] **M4-T2** Full quantization run: the trade-off table for bf16, 8-bit, 4-bit.
- [ ] **M4-T3** CI eval gate (Q7, F2).

**Gate G4 (tag `v0.2` with G3):** the table is generated from committed results; a deliberately degraded configuration turns the CI gate red.

### Stage 3: scale-out, agents, and v1.0

#### M5 Go router `[B]`

- [ ] **M5-T1** Proxy, health checks, router metrics and their breaker-state and load-per-replica panels in the Grafana dashboard, chaos test against `ullm sim` (R1, R2, R8, O3).
- [ ] **M5-T2** Circuit breaker and load-aware selection (R3, R4).
- [ ] **M5-T3** Prefix-aware routing (R5). Needs E10.
- [ ] **M5-T4** Telemetry fan-in; replicas panel and decision feed; `make up` (R6, T6).
- [ ] **M5-T5** Tracing across Go and Python (O4).
- [ ] **M5-T6** Routing benchmark on workload B with the real engine: prefix-aware against round-robin, hit rate and TTFT. No throughput claim.

**Gate G5:** chaos test green in CI; routing benchmark committed; one connected trace per request.

#### M6 Agent-facing API

- [ ] **M6-T1** `[design]` Tool calling (A2).
- [ ] **M6-T2** Tool-call eval across the three quantizations (Q6).
- [ ] **M6-T3** `[design]` Structured output and its eval (A3, Q6).
- [ ] **M6-T4** Router record mode; capture and replay workload D (R7, B2).
- [ ] **M6-T5** MCP server (A4).

**Gate G6:** SDK tool round trip green; evals committed; workload D results committed; the MCP server used from a coding agent.

#### M7 v1.0

- [ ] **M7-T1** Final benchmark matrix: workloads A–D, both load modes, all baselines, goodput, Qwen3-1.7B for headline tables.
- [ ] **M7-T2** Write-up (P3).
- [ ] **M7-T3** README final, claims ledger complete, hosted replay updated with every panel (P1, P2, T4).
- [ ] **M7-T4** Upstream contribution (P5).
- [ ] **M7-T5** `[user]` DESIGN.md complete, quiz, resume updated (P6).

**Gate B (tag `v1.0`):** every claim in the README and write-up has a ledger row that reproduces; all gates G0–G6 still pass.

### Stage 4: platform

#### M8 Multi-tenant gateway

- [ ] **M8-T1** API keys and tenant propagation (R9, C4).
- [ ] **M8-T2** Rate limiting in Redis (R10).
- [ ] **M8-T3** Usage ledger in Postgres with idempotency tests (R11).
- [ ] **M8-T4** Fair-share scheduling (E16).
- [ ] **M8-T5** Tenants view; Redis and Postgres in compose (T8).
- [ ] **M8-T6** Security review of the gateway.

**Gate G8 (tag `v1.1`):** 401, 429 and ledger tests green; fairness shown on a mixed-tenant workload; salting tied to keys.

#### M9 Deployment blueprint

- [ ] **M9-T1** Container images (DEP1).
- [ ] **M9-T2** Helm chart on kind (DEP2).
- [ ] **M9-T3** Autoscaling on queue depth, with a recorded run.

**Gate G9 (tag `v1.2`):** one command brings the cluster up from a clean machine state; the scaling run is committed.

---

## 8. Depth tracks

Optional, after v1.0, in any order the author chooses by target role. Each is self-contained and adds its own claim.

- **E13 Chunked prefill.** Split a long prompt into chunks and alternate them with decode steps, using the `PREFILLING` progress state. *Proof:* TPOT p99 and short-request TTFT on workload C before and after; equivalence preserved.
- **E14 Speculative decoding.** Prompt-lookup drafting first, then a small draft model for a larger target. *Proof:* acceptance rate, end-to-end speedup, identical output under greedy decoding.
- **E15 Shared-prefix attention.** Layout C from E9. *Proof:* decode throughput on workload B before and after; equivalence preserved.
- **G1 Ops agent.** See 6.12.
- **A5 Anthropic-compatible endpoint**, so Claude Code can be a client.
- **DEP4 Live simulated demo.** The router and `sim` replicas on a free hosting tier, feeding the hosted terminal live, clearly labelled as a simulated model backend.

Not doing: a custom Metal kernel, a CUDA port, multi-GPU, fine-tuning, paid cloud GPUs, LangChain inside the engine.

---

## 9. Risks and fallbacks

| # | Risk | Response |
|---|---|---|
| R-1 | Block-based KV with a per-step gather makes decode several times slower (section 4) | E9 spike before building; three candidates; decision rule; the logical layer is unaffected |
| R-2 | Batched bf16 arithmetic flips near-tied greedy picks | Measured: exact ties at 6 of 495 golden positions, and divergence at near-ties across machines (decision 0006). Compare against mlx-lm on the same machine; log the top-2 margin against a documented threshold. Measured for batching (decision 0021): every flip at margin 0.25 or less, uLLM's token always mlx-lm's second. Never loosen the test |
| R-3 | The mlx-lm baseline already batches, so there is no easy win | DEC-8. The value is a measured, explained gap |
| R-4 | MLX may not run usefully on hosted macOS CI runners | Measured (decision 0005): MLX runs on the runner's paravirtualized GPU. Core tests still run on Linux with `FakeRunner`. If golden output diverges on the runner, CI compares against mlx-lm run on the same runner. Fallback: a local pre-push gate, then a self-hosted runner |
| R-5 | A fanless laptop throttles and distorts results | B5 run conditions |
| R-6 | 16 GB shared by replicas, the KV pool and Docker | The 0.6B model for multi-replica runs; compose stopped during benchmarks; pool sized from a memory budget |
| R-7 | The terminal grows without limit | Panels are fixed per milestone (6.6); budgets in T7; Lane B only |
| R-8 | The author cannot defend agent-written code | Teach-back notes, author-written DESIGN.md, the quiz at every gate, the catch log |
| R-9 | A small model produces poor tool calls and agent traces | Tool calling is reported as an eval result, not promised as a capability. DEC-12 for traces |
| R-10 | A baseline cannot be installed | Document what was tried in the results and move on |
| R-11 | MLX or mlx-lm changes underneath | Pinned versions; goldens record their versions; rule 12 |
| R-12 | Scope keeps growing because there is no deadline | Stage order is fixed; depth tracks only after v1.0; rule 3.6 for anything new |

If a piece cannot be made to work, do not fake it and do not silently drop it: record what was tried, what was measured, and the decision, then update this file. Never dropped under any circumstances: the equivalence suite, the property tests, the quantization evals, the benchmark method, the claims ledger, the author's own artifacts.
