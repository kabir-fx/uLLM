# M2-T10: Terminal quality gates

**State:** in build. You do not need to do anything now.
**Decision record:** [0027](../decisions/0027-terminal-quality-gates.md) (the full design, for agents) · **Spec:** roadmap M2-T10 (T7)

## The goal

A real browser proves that the terminal stays at 60 fps under load, keeps its memory flat, and opens fast. You can also drive the replay and the requests tabs from the keyboard.

## The problem

- No test opens the terminal in a browser. The canvases, the animation loop and the hosted build have no test.
- T7 sets three budgets, but nothing measures them.
- The KV block grid comes in M3, so the frame-rate test can cover only the panels that exist now.
- CI timings mean nothing, so no budget can run in CI (roadmap 4).
- The keyboard can do almost nothing in the terminal now.

## How it works

```
stress.jsonl.gz ─▶ vite preview (dist/)        ─▶ perf/*.spec.ts ─▶ bench/results/m2/terminal-perf/
bundled runs   ─▶ vite preview (dist-hosted/) ─▶ e2e/*.spec.ts  ─▶ make check-ts, CI
```

1. `src/ullm/telemetry/synthetic.py`: a new `stress` scenario. It keeps 64 requests in flight on a 2,304-block pool, at about 15 frames a second.
2. `terminal/playwright.config.ts`: two projects, `smoke` and `perf`, each on its own `vite preview` server.
3. `terminal/e2e/`: the smoke test and the keyboard tests, on the hosted build.
4. `terminal/perf/`: the three budgets. `make terminal-perf` runs them and writes the result and the machine.
5. `terminal/src/keys.ts`: the replay keys. Space plays or pauses. The arrow keys seek 5 s.
6. `terminal/src/ui/RequestsTable.tsx`: the tabs follow the WAI-ARIA tabs pattern.
7. `terminal/src/ui/app.css`: a visible focus ring on every control.

The agent measured the frame rate first, with a throwaway script. Chromium held 60 fps on the stress trace, headless and headed.

## What you get at the end

Run `make check` for the smoke test and the keyboard tests. Run `make terminal-perf` on the M4 for the budgets. The result goes to `bench/results/m2/terminal-perf/`.

New tests:

- The hosted replay opens in a clean browser, plays, draws on each canvas, and logs no error.
- Space plays and pauses. The arrow keys move the position by 5 s.
- The arrow keys, Home and End move between the requests tabs.
- Tab reaches every control, and each control shows a focus ring.
- The frame rate stays at 60 fps over the full stress trace.
- In live mode, the heap after 40 repeats of the stress trace equals the heap after 20.
- The hosted replay is ready in 1 s or less.

What you do not get yet:

- The KV block grid in the frame-rate test: the M3 task that builds T5.
- A budget for live mode on its own. Replay and live mode feed the same store and panels.

## Where this puts the project

```
M2  T1 █  T2 █  T3 █  T4 █  T5 █  T6 ░  T7 █  T8 █  T9 █  T10 ▓  T11 ░  T12 ░  T13 ░
```

After this slice, Lane B has no open task in M2. M4 is next for Lane B.

## Choices the agent made

You can still change these before `/ship`.

- The frame-rate budget needs a mean of 58 fps or more. The p99 frame interval must be 25 ms or less.
- The memory budget compares 40 repeats with 20 repeats. The rings are full only after 19 repeats.
- The memory test uses live mode. A replay keeps a copy of the whole trace, and that copy hides a small leak.
- The keys do nothing when the focus is on a form control, a button or a tab. Those keep their own behaviour.

## What can go wrong

- The CI terminal job downloads Chromium on each run. This adds time to the job.
- `make check` needs Chromium. The first run downloads about 100 MB.
- A fast Mac can hide a slow panel. The budgets are for the M4 only.

## Words

| Word | Meaning |
|---|---|
| smoke test | A short browser test that shows the hosted replay opens and works. |
| time to interactive | The time from the start of navigation until the replay panel shows a duration above zero. |
| stress trace | The `stress` fixture, made by the synthetic generator for the budgets. |

## Learn from this slice

Frame rate in a browser is the gap between two `requestAnimationFrame` callbacks. Any long task on the main thread makes a gap longer than 16.7 ms. So the test measures all the work, not only the drawing.

A leak test must let the store fill first. A store with ring buffers grows until each ring is full, and then it stays flat.

**Test yourself:** why does the memory test use live mode and not a replay?

<details>
<summary>Answer</summary>

A replay keeps a copy of the whole trace in memory. That copy is about 18 MB for ten repeats. A small leak hides in it, and a leak that keeps frames costs a replay nothing. Live mode keeps no frames, so its heap shows each leak.

</details>

## Result
