# M2-T6: The concurrency sweep

**State:** in build. You will run the five sessions on your idle Mac when the agent asks.
**Decision record:** [0026](../decisions/0026-benchmark-sweep.md) (the full design, for agents) · **Spec:** roadmap M2-T6 (B3, B6)

## The goal

This slice measures uLLM with batching against batched `mlx_lm.server` at concurrency 1, 2, 4, 8 and 16. It also explains each gap with our own measurements.

## The problem

- The M1 tables pin uLLM to one sequence. No committed result shows uLLM with batching against a baseline.
- At c1, uLLM needs 1.167 times the time per token of sequential `mlx_lm.server`. Nobody explained this gap (B6).
- Workload A has 256 requests. At c16, 32 prompts a chunk leave no warm-up requests.
- The Mac has 16 GB. Batched `mlx_lm.server` peaked at 9.68 GB at c8. Nobody measured c16.
- A server that starts for each chunk overwrites its recording. The committed M2-T4 recording holds the last chunk only.
- Rule 7: every figure comes from a committed result. Rule 10: no `generate_step` in engine code.

## How it works

```
make bench-sweep
   ├── bench.session  m2-c1 … m2-c16 ──▶ bench/results/m2/sweep/c<N>/
   │      one server per chunk, taking turns:
   │      ullm (records ullm-<block>.jsonl.gz) · baseline · twin · mlx-lm-batched
   ├── bench.steps    on each session  ──▶ c<N>/steps.md
   └── bench.analysis --sweep          ──▶ comparison.md

scripts.step_costs (offline, one process) ──▶ step-costs.json, step-costs.md
gap report, by hand, every figure linked ──▶ README.md
```

1. **Group files**, `bench/groups/m2-c1.json` to `m2-c16.json`. Each names the same engines. The baseline is batched `mlx_lm.server` without its prompt cache. uLLM runs with `--max-num-seqs 16` everywhere.
2. **`{block}` placeholder** in `bench/session.py`. The session puts the block number into the launch when the server starts. Then each chunk of uLLM writes its own recording.
3. **`bench/steps.py`** reads the recordings of one session. It gives decode step time by batch size, and prefill time per 1,000 tokens. It gives the share of time in prefill, in decode, and between steps. It also gives the stall that the prefill of a new request puts on the requests that decode.
4. **`scripts/step_costs.py`** loads the model once, with no server. It times one decode step of uLLM at 1 to 16 rows. It also times the repack of the batch KV. At one row, it compares the decode loop of mlx-lm with the same loop that waits on each step.
5. **`bench/analysis.py --sweep`** prints one table per regime. Each row is an engine, and each column is a concurrency.
6. **The gap report**, `bench/results/m2/sweep/README.md`, names each gap outside the twin noise, and its cause. A gap without a measured cause says so.

The agent read these facts in the code:

- `generate_step` in mlx-lm starts the next step before it waits on the current step. uLLM waits on each step before it starts the next. The agent thinks that this causes the c1 gap, but did not measure it yet.
- uLLM copies the whole batch KV when a request joins or leaves the batch.
- The `step` event time includes that copy, the forward pass and sampling.

## What you get at the end

- `make bench-sweep` runs five sessions and writes `bench/results/m2/sweep/`.
- `comparison.md` holds the output rate, TTFT and TPOT ratios at each concurrency, Peak first.
- Each session has `summary.md`, `steps.md` and its raw data.
- `step-costs.md` holds the offline step times.
- `README.md` explains every gap, and `docs/CLAIMS.md` gets a row for each headline.
- New tests prove these behaviours:
  - A server that starts for each chunk gets the block number in its launch.
  - A resident group with `{block}` is refused.
  - `bench.steps` gives the hand-computed figures on a small recording.
  - The sweep table gives one cell per engine and concurrency, with the twin noise.

What you do not get yet: open loop with Poisson arrivals (M7-T1), and other baselines of B4 (M7-T1).

## Where this puts the project

```
M0  █████████
M1  ████████
M2  █████▓██░░░░░
```

This slice opens M2-T12, the README with the benchmark table.

## Choices the agent made

You can still change these before `/ship`.

- **One server for each chunk, at every concurrency.** M1 kept the c1 servers loaded. Then the c1 row has a different lifecycle from the other rows.
- **Prompts per chunk:** 12 at c1, c2 and c4, 24 at c8 and 30 at c16. Each chunk has as many warm-up requests as the concurrency, and 2 at c1.
- **Memory limits come from a smoke session** of c1 and c16, one block each. Each limit is the peak, rounded up, plus 1 GB.
- **The step time split comes from an offline script,** not from a new field in the `step` event. A new field changes contract C2 and the terminal types.

## What can go wrong

- c16 has 8 blocks only. If few blocks are Peak, the Peak row has no interval.
- Batched `mlx_lm.server` with its cache can push the Mac into swap at c16. Then the session is DISCARDED. The smoke session shows the peak first.
- The offline script times parts of a step in a quiet process. It explains a gap, but it is not the benchmark result.

## Words

| Word | Meaning |
|---|---|
| sweep | The five sessions, one for each concurrency |
| step | One engine pass: a prefill of one prompt, or a decode of every running row |
| repack | The copy of the batch KV when a request joins or leaves the batch |
| stall | Prefill time of another request while a request waits for its next token |
| pipelining | Starting the next step on the GPU before the host reads the current result |

## Learn from this slice

A benchmark result says how much slower an engine is. It does not say why. The step log and a small offline timing turn a ratio into a cause.

Continuous batching helps the output rate, but each new request stops the batch for its whole prefill. At high concurrency, requests join often, so this stall grows.

Test yourself: why does TPOT at c16 grow for uLLM even when the decode step itself is fast?

<details><summary>Answer</summary>

uLLM prefills each new prompt alone, in one step. Every running request waits for that step. At c16, a new request joins about each time a request ends, so the running requests wait often.

</details>

## Result
