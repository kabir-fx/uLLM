# M1-T7: The first baselines

**State:** in build. You give the green light when the Mac is idle, plugged in, and the sudoers rule is in place.
**Decision record:** [0014](../decisions/0014-m1-baselines.md) (the full design, for agents) · **Spec:** roadmap M1-T7 (B4)

## The goal

This slice measures uLLM against `mlx_lm.server` on workload A under B5. The results go into `bench/results/m1/`, with one table that compares the runs.

## The problem

- No committed result compares uLLM with a baseline. Gate G1 needs one.
- B4 asks for `mlx_lm.server` in a sequential mode and in its default batched mode. It has no flag for a sequential mode.
- Rule 7 asks for the same request for every target. Rule 8 says that every number comes from a committed run.
- In M1-T6, repeats 2 and 3 of a probe ran 3.5 times slower than repeat 1. The cause was not known.
- In M1-T6, `mlx_lm.server` gave other token counts than uLLM for the same prompts.

## How it works

```
make bench-baselines ── bench/baselines.sh
     │
     ├── 6 × bench.run (B5 check, powermetrics, 4 warm-up, 3 repeats)
     │        ├── ullm serve
     │        └── scripts/mlx_lm_server.py --model <pinned snapshot>
     └── bench.analysis --compare ──▶ bench/results/m1/comparison.md
```

1. **`bench/baselines.sh`.** It runs six runs, one after the other, with 120 s of idle between them.
2. **`bench.run --model`.** It sends the pinned snapshot path to `mlx_lm.server`. Without it, the server can load another model from the cache.
3. **`bench.run --extra-body`.** It adds `"seed": 0` to the requests of the sequential `mlx_lm.server` runs. A seeded request takes the single-request path of the server. At temperature 0 the seed changes no token.
4. **`scripts/mlx_lm_server.py`.** It starts `mlx_lm.server` with the listen queue `ullm serve` has. With its own queue of 5, macOS refused connections opened while the server was busy.
5. **`bench.analysis --compare`.** It writes the table across the six runs.

The agent measured these facts on 8 requests of workload A:

- With the seed and the prompt cache off, `mlx_lm.server` gave the same text as uLLM for 8 of 8 requests.
- Without the seed, `mlx_lm.server` uses its batch path for each request. Then 6 of 8 replies were different.
- The prompt cache of `mlx_lm.server` changed 4 of 8 replies, also on the single-request path.
- The slowdown in M1-T6 came from another process. A `node` process used 99% of one CPU core. The GPU clock fell from 1,465 MHz to 450 to 900 MHz. On a quiet Mac, the three repeats gave the same TPOT, 16.1 ms.

## The six runs

| Run | Concurrency | What it shows |
|---|---|---|
| `ullm` | 1 | uLLM, one request at a time |
| `mlx-lm-sequential-no-cache` | 1 | The same tokens as uLLM, so the speed compares directly |
| `mlx-lm-sequential` | 1 | The effect of the prompt cache |
| `ullm` | 8 | uLLM with a queue, because M1 has no batching |
| `mlx-lm-batched` | 8 | What batching gives, with the default prompt cache |
| `mlx-lm-batched-no-cache` | 8 | What batching gives without the prompt cache |

## What you get at the end

- `make bench-baselines` runs the six runs. It takes about 3 h on an idle Mac.
- `bench/results/m1/comparison.md` holds the table. Each run directory holds its `summary.md` and its raw data.
- New tests prove that `--extra-body` cannot replace a common field, and that `config.json` records it. They also prove that a run refuses a model that the target does not serve.

What you do not get yet: vllm-metal, vllm-mlx and Ollama (rest of B4), and the concurrency sweep (M2-T6).

## Where this puts the project

```
M0  █████████
M1  ██████▓░
```

## Choices the agent made

You can change these before `/ship`.

- The run directories have fixed names, such as `bench/results/m1/ullm-c1`. A second run cannot write over a committed run.
- The script stops at the first run that fails. You then read why before the next run.
- After the runs, the agent will compare the token counts of `ullm` and `mlx-lm-sequential-no-cache` for each request. They must agree.

## What can go wrong

- Another busy process slows the GPU. B5 cannot detect it. The agent will read the `powermetrics` log of each run, and will run again any run with a clock drop.
- The batched replies of `mlx_lm.server` are not the same text as uLLM's. Their lengths are different, so their token totals are different too.
- A full run stops before it starts on battery, or when `powermetrics` needs a password.

## To resume

The state for a new session, if the agent loses this chat:

1. Branch `m1/t7-baselines`. The harness, the script, decision 0014, claim 18 and the upstream candidates are committed.
2. Wait for the author to say that the Mac is idle, on AC power, and has the sudoers rule.
3. Run `make bench-baselines` in the background. Then do the result checks in decision 0014.
4. Commit the results and add the claim rows for the numbers. Fill in **Result** and update section 0. Then run `prove` and `vet`.

## Words

| Word | Meaning |
|---|---|
| single-request path | The code in `mlx_lm.server` that serves one request without `BatchGenerator` |
| prompt cache | The KV state that `mlx_lm.server` keeps from earlier prompts and uses again |
| contention | Another process that uses the CPU or GPU during a run |
