# M1-T7: The first baselines

**State:** the code is built and reviewed. The sessions wait for your green light on an idle Mac.
**Decision record:** [0016](../decisions/0016-paired-sessions.md) (the full design, for agents), with [0014](../decisions/0014-m1-baselines.md) and [0015](../decisions/0015-mlx-cache-limit.md) · **Spec:** roadmap M1-T7 (B4), and your `Fanless Mac Benchmarking - Final Approach.md`

## The goal

This slice measures uLLM against `mlx_lm.server` on workload A, on a fanless Mac. Each result is a ratio between engines that ran in turns. Then the heat of the Mac does not decide the result.

## The problem

- No committed result compares uLLM with a baseline. Gate G1 needs one.
- The Mac has no fan. The GPU holds 1470 MHz for some minutes, then the OS caps it.
- The plan of decision 0014 ran each engine alone, with repeats and rests. Every run failed its clock check.
- Your Final Approach replaces that plan. This slice must build it exactly, and name each place where it cannot.
- Rule 7 asks that every number comes from a committed result. The figures of the two spikes are not committed yet.

## What changed in this second design pass

The agent checked the first version of decision 0016 against your approach and against the spike data. It found three faults:

1. **The concurrency 8 group cannot fit in memory.** The agent loaded the five c8 servers together. Their footprints came to 29.3 GB, and swap rose to 18.8 GB. Your approach says: load one server at a time when the servers do not fit. The first version did not do that.
2. **The twin gate would discard spike 1.** The first version checked the twin on every metric. In spike 1, the twin's output tokens per second in Peak was 0.992, outside 0.5%. Your floors are TPOT floors, so the gate now checks TPOT.
3. **Some evidence came from the thermal label.** The first version took some figures from the thermal-label segments that it rejects. The agent calculated every figure again with the capped rule.

The capped rule itself is correct. It gives your counts for spike 1 exactly: 3 Peak, 5 Transition, 4 Sustained.

## How it works

```
make bench-baselines
   ├── bench.session  m1-c1.json ──▶ bench/results/m1/c1/
   │      settle 240 s, then all servers loaded, taking turns:
   │      ullm · baseline · baseline twin · mlx-lm-sequential
   ├── bench.session  m1-c8.json ──▶ bench/results/m1/c8/
   │      settle 600 s, then one server per chunk, taking turns:
   │      ullm · baseline · baseline twin · mlx-lm-batched · (sequential)
   └── bench.analysis --compare  ──▶ bench/results/m1/comparison.md
```

1. **Group file**, `bench/groups/m1-c1.json` and `m1-c8.json`. It names the engines, the baseline, the concurrency and the gate limits. It also says if the servers stay loaded or start for each chunk.
2. **`bench/plan.py`.** It cuts workload A into blocks. Each block has one prompt from each prompt-length stratum. The order of engines rotates through a Williams design.
3. **`bench/session.py`.** It starts the servers and runs the blocks. Each chunk starts with warm-up requests that are not counted. After each chunk it records memory, swap and power.
4. **`bench/regimes.py`.** It reads the `powermetrics` log. A sample is capped when the GPU got a lower state than it asked for. Each block becomes Peak, Sustained or Transition.
5. **`bench/paired.py`.** It divides one engine's figure by the baseline's, prompt by prompt, in the same block. The interval comes from a bootstrap over whole blocks.
6. **`bench/analysis.py`.** It writes `summary.md`, which starts with VALID or DISCARDED and names each gate that failed.

The agent measured these facts with scripts in `bench/.scratch/thermal-spike/design-checks/`:

- With a margin of 0.3, the capped rule gives 3, 5 and 4 blocks for spike 1. A margin of 0.5 gives the same counts.
- The baseline and its twin must each be capped for half their busy time. If the two are pooled, the counts become 3, 4 and 5.
- The twin's TPOT ratio was 0.998 to 1.000 in every regime of both spikes.
- After decision 0015, uLLM had no gap over 100 ms. Its TPOT ratio to mlx-lm was 1.156 to 1.170 in Peak, and 1.168 in Sustained.
- The four c1 servers together used 7.2 GB. One batched `mlx_lm.server` alone used 7.0 GB without its prompt cache and 11.3 GB with it.

## What you get at the end

- `make bench-baselines` runs two sessions, one at concurrency 1 and one at concurrency 8.
- `bench/results/m1/comparison.md` holds the ratios to the baseline, Peak first. Each session has its `summary.md` and its raw data.
- New tests prove these behaviours:
  - The blocks hold one prompt from each stratum, and the warm-up prompts never get measured.
  - The engine orders are balanced.
  - A block sorts to Peak, Sustained or Transition by the capped rule, with each decider checked alone.
  - The paired ratio and its interval match values that the agent calculated by hand.
  - Each gate discards a session that breaks it.
  - A session with fake servers writes every file and stops every server, in both server modes.
  - The sorter gives your block counts on both committed spikes.

What you do not get yet: the concurrency sweep and concurrency 16 (M2-T6). You do not get a Low Power Mode table, or the other baselines of B4.

## Where this puts the project

```
M0  █████████
M1  ██████▓░
```

This slice opens M1-T8 and gate G1. M2-T6 and M3-T9 use the session, the regimes and the paired ratios again.

## What you approve

These choices beat other options:

- **Paired sessions replace each engine alone.** The other option kept `bench.run` per engine and added an orchestrator. Two modules then owned the servers.
- **The capped rule sorts the blocks.** The thermal label and the clock mix failed in both spikes.
- **A Williams design gives the engine orders.** Your approach says "every ordering". With 4 engines that is 24 orders, but c1 has only 21 blocks. A Williams design keeps the balance with 4 orders. For 3 engines it is every ordering.
- **The session records memory after each chunk, not after each block.** Only the engine that just ran can change its memory.
- **Every probe checks AC power and Low Power Mode.** Then B5 holds for the whole session, not only at its start.
- **`bench.run` stays for `make bench-smoke` only.** Claims come from sessions only.

Your answers to the open questions:

1. **Concurrency 8 servers.** You chose one server for each chunk, as your approach says.
2. **Twin gate.** You chose the TPOT median within 0.5% for Peak and 3% for Sustained. The gate does not test the interval.
3. **Memory limits.** You chose a peak footprint limit for each engine and zero swap growth. The limits are 3 GB for uLLM and the sequential baseline, and 4 GB for sequential with cache. They are 11 GB for batched without cache and 14 GB for batched with cache. The agent measured swap with your apps open, but the sessions run on an idle Mac.
4. **Sequential row at concurrency 8.** The agent adds `mlx-lm-sequential-no-cache` to c8, and you did not change it. It shows how much batching gives, apart from the engine.
5. **Chunk at concurrency 8.** The agent uses 24 prompts and 8 warm-up requests, and you did not change it. Each c8 server starts cold, so its warm-up runs one full batch. Nobody measured this.
6. **Settle time.** The agent uses 240 s before c1, as the spikes did, and 600 s before c8. You did not change it. The `powermetrics` log records both waits.
7. **The evidence.** You chose to commit both spikes, with the clock logs gzipped, under `bench/results/m1-spikes/`. The design-check scripts go with them. A test then checks the sorter against your counts.
8. **The old runs.** The agent will ask you before it deletes the five directories in `bench/results/m1/`. Git does not track them.

## What can go wrong

- The Peak sample is small: only the first minutes of each session. More Peak needs more sessions.
- Sustained numbers do not repeat between sessions. A table never shows them without their regime and clock mix.
- The batched server with cache uses 12.3 GB at its peak. With your apps open, it pushed 94 to 111 MB into swap per chunk. On an idle Mac, nobody measured this. Any swap growth makes the c8 session DISCARDED. So the agent first runs one c8 block as a smoke session on the idle Mac.
- Nobody tested a server that starts for each chunk. The twin of c8 shows if it works. If the twin disagrees, the session is DISCARDED.
- The capped rule is proven on two spikes at concurrency 1 only.

## Words

| Word | Meaning |
|---|---|
| block | The same prompts, run by each engine of the group, one chunk each |
| chunk | One engine's turn in a block: its warm-up requests, then the block's prompts |
| twin | A second copy of the baseline server, which gives the noise floor |
| capped | A `powermetrics` sample where the applied GPU state is lower than the requested state |
| Peak | A block with no capped sample |
| Sustained | A block where the baseline and its twin are each capped for at least half their busy time |
| Transition | A block that is neither Peak nor Sustained. It is in no table |
| paired ratio | One engine's figure divided by the baseline's, for the same prompt in the same block |
| floor | The largest distance from 1.00 of the twin's interval. A smaller engine gap is not claimed |
| resident | All servers of a group stay loaded for the whole session |
| per chunk | A server starts before its chunk and stops after it |

## Learn from this slice

A benchmark on a hot machine compares the heat as much as the engines. When engines take turns on the same prompts, the heat is nearly the same for each pair. The ratio then measures the engines.

A twin shows the noise of the method. If two identical servers disagree by 1%, a gap of 1% between engines tells you nothing.

A test on many metrics fails more often than a test on one metric. Each metric gives the noise one more chance to cross the limit. That is why the twin gate uses TPOT only.

Test yourself: why does the baseline, and not uLLM, decide when a block is Sustained?

<details><summary>Answer</summary>

`mlx_lm.server` asks for the highest GPU state whenever it is busy, so a cap always shows on it. uLLM asks for less, so the cap shows on it only some of the time.

</details>

## Result

**NOT DONE.** The code is built, and the proof and the review pass. The real sessions have not run.

- The verifier ran a real c1 smoke session. All 64 requests passed, and no server stayed up.
- The capped rule gives your counts on both spikes: 3, 5, 4 and 7, 2, 7.
- The review found 4 Required problems. The agent fixed each one with a test that failed first.
- A block that the clock log does not cover is now Transition, not Peak.
- The memory figures in 0016 have no committed output. The c1 and c8 sessions will give them.
- To finish, you give an idle Mac on AC power. Then the agent runs one c8 block as a smoke session, and after that `make bench-baselines`.

