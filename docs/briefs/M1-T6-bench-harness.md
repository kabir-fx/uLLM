# M1-T6: The benchmark harness

**State:** built. You can change the choices below before `/ship`.
**Decision record:** [0013](../decisions/0013-bench-harness.md) (the full design, for agents) · **Spec:** roadmap M1-T6 (B1, B2, B5)

## The goal

This slice adds one harness that measures any OpenAI-compatible server in the same way. It writes each run to a directory that a script turns into tables and a chart.

## The problem

- No tool measures a served request. `make bench-smoke` exits 2.
- M1-T7 compares uLLM with `mlx_lm.server`. Rule 7 says that both targets get the same prompts, chat template, thinking setting and `max_tokens`.
- Rule 8 says that every number comes from a committed result. C5 sets the files of that result.
- B5 sets the run conditions: AC power, Low Power Mode off, and a `powermetrics` log. It also asks for a warm-up, cooldowns, and the median of three runs.
- Neither target has `/metrics` yet. `powermetrics` must run as root.

## How it works

```
bench/make_workload_a.py ──▶ bench/workloads/a.json
                                      │
bench/run.py ── conditions.py (B5 check, powermetrics)
     │
     ├── launch the target, wait for /health
     ├── loadgen.py ── N clients ── SSE ──▶ target
     └── bench/results/<run>/ ──▶ analysis.py ──▶ summary.md, latency.svg
```

1. **`bench/make_workload_a.py`.** It reads UltraChat dialogues at a pinned revision. It keeps 256 prompts of 100 to 1,000 tokens. It counts the tokens with the uLLM tokenizer and gives each request a `max_tokens` from 50 to 300.
2. **`bench/schema.py`.** It defines the workload file and each result file.
3. **`bench/loadgen.py`.** N clients send requests from one queue. Each client reads the raw SSE stream and records the time of each text chunk.
4. **`bench/conditions.py`.** It reads the power source and Low Power Mode with `pmset`. It refuses a full run that breaks B5. It starts and stops `powermetrics`.
5. **`bench/run.py`.** It starts the target, sends a warm-up, and then sends three measured repeats with a cooldown between them. It writes the C5 files.
6. **`bench/analysis.py`.** It calculates TTFT, TPOT, latency and tokens per second. It writes `summary.md` and `latency.svg`, and a table that compares runs.

## What you get at the end

- `make bench-smoke` measures the real model on 16 requests. It took 81 s on the M4.
- `uv run python -m bench.run --target ullm --launch "uv run ullm serve"` makes a full B5 run.
- `uv run python -m bench.analysis <run>... --compare` writes each summary and prints a table that compares the runs.
- Tests prove the SSE timing, the metric formulas, the B5 refusal, and a full run against `ullm serve --fake`.

What you do not get yet: the baseline runs (M1-T7), Poisson arrivals (M2-T6), and KV data in summaries (M2-T5).

## Where this puts the project

```
M0  █████████
M1  █████▓░░
```

The next slice is M1-T7, the first baselines.

## Choices the agent made

You can change these before `/ship`.

- The prompts come from UltraChat, not ShareGPT. ShareGPT holds the text of real users, and this repo is public.
- One run directory holds the warm-up and all three repeats. C5 says one directory per run.
- The client reads raw SSE with `httpx2`, not the OpenAI SDK. The SDK hides the time when each chunk arrives.
- TPOT uses the token count from `usage`. A chunk can hold more than one token.
- matplotlib is a new dev dependency, for the chart.

## What can go wrong

- A full run stops before it starts if you are on battery or did not run `sudo -v`. The run prints each failed condition.
- The workload script needs the network and the model. The committed `a.json` does not.
- A target that does not send `usage` gives no TPOT or throughput. The summary shows those requests as failed.

## Words

| Word | Meaning |
|---|---|
| run | One target at one concurrency, measured under B5, in one directory |
| repeat | One measured pass over the requests of a run |
| warm-up | Requests that a run sends first and does not count |
| TTFT | The time from send to the first text chunk |
| TPOT | The mean time per output token after the first token |

## Learn from this slice

A closed loop keeps N requests open at all times. A new request starts only when one ends. So the server sets the pace, and the result shows its throughput at that concurrency.

TPOT divides by the token count from `usage`, not by the chunk count. A server can hold back an incomplete character and send two tokens in one chunk. A chunk count would then show a false TPOT.

**Test yourself:** why does the harness count `prompt_tokens` mismatches?

<details><summary>Answer</summary>A mismatch shows that the target rendered a different prompt. Then the comparison breaks rule 7, and the numbers are not fair.</details>

## Result

**DONE.** The proof is VERIFIED and the review is READY at `2ccec75`. The fix commits up to `ad3478e` are covered. Three agent runs did the work: one verifier, one reviewer, and one scoped re-check. `make ci` passed at `ad3478e`: 374 core tests, 8 equivalence tests and 86 `mlx` tests.

Changes against the plan:

- The B5 check runs `powermetrics` once with `sudo -n`. The plan used `sudo -n -l`, but any unrelated NOPASSWD rule lets that pass.
- A full run whose `powermetrics` wrote no log exits 1. The comparison table marks it "not B5".
- `--compare` is a flag. It prints the table to the terminal.
- `bench.analysis` refuses a run when the workload file changed after the run.
- You chose to run `mlx_lm.server` with its prompt cache on and off in M1-T7. Decision 0013 records the plan.

Proof and review found these bugs, and the agent fixed each one:

- One unreadable SSE payload stopped the whole run. The run then lost all of its repeats. Now that request fails, and the run continues.
- A full run with a dead `powermetrics` exited 0 as a B5 run.
- `--compare <run> <run>` stopped with `IsADirectoryError`.
- `--concurrency 0` sent no requests and exited 0.
- No test checked the request body, the median over repeats, or the exit code after a failed request. Now a mutant of each makes a test fail.
- Claims 17 and 18 and decision 0013 did not match the code in four places.

Proven on the real machine: you ran a full B5 run on AC power. It wrote 136 `powermetrics` samples, and the last sample is complete. In your terminal, `can_sudo()` returned `True` after `sudo -v`.

Not proven: a run that has a `/metrics` file. Neither target serves `/metrics` before M2-T5.

For later slices:

- In your B5 run, repeats 2 and 3 were 3.5 times slower than repeat 1 for the same request. Thermal pressure stayed "Nominal". M1-T7 must find the cause before it reports a baseline.
- In a harness run, `mlx_lm.server` gave fewer tokens on 3 of 8 requests than it gave for each request alone. Its prompt cache is a possible cause. M1-T7 must check this.
- A stream that hangs stops the run, because requests have no timeout.

Read these yourself:

- `bench/conditions.py:54`: does the probe test the same command that `PowerMetrics` starts?
- `bench/loadgen.py:27`: does each bad payload become a failed request, and not a crash?
- `docs/decisions/0013-bench-harness.md:87`: do you agree with the plan for prompt caches in M1-T7?
