# M2-T7: Step chart, batch strip and requests table

**State:** DONE at `2babb83`. Type `/ship` to open the pull request.
**Decision record:** [0021](../decisions/0021-core-panels.md) (the full design, for agents) · **Spec:** roadmap M2-T7 (T2)

## The goal

You see each engine step in the terminal. A step chart shows step latency and token rates. A batch strip shows the requests in each step. A requests table lists every request by state.

## The problem

- The terminal shows only the ticker and the feed. You cannot see batches form, requests join and leave, or a prefill stall.
- Telemetry must not pass through React state for each event (T1). The canvases must draw on `requestAnimationFrame`.
- The T7 budget names 200 table rows and 15 frames a second. The table must render only the rows that you can see.
- Lane B may not change a contract (roadmap 3.5). The panels read C2 as it is.
- Fixture data must show the SIMULATED label (decision 0003). The new panels sit below the header that shows it.

## How it works

```
 frames ─▶ TelemetryStore ──┬─▶ seconds (one bucket per second) ─▶ drawStepChart ─▶ step chart canvas
                            ├─▶ steps (the newest steps) ─────────▶ stripModel ─▶ drawStrip ─▶ batch strip canvas
                            └─▶ request rows ─▶ tableRows(tab) ─▶ visibleRange ─▶ requests table
```

1. **`foldStep`** (`src/timeline.ts`). It adds each step to a bucket for its second of trace time. A bucket holds the open, high, low and close of the step latency. It also holds the tokens that prefill and decode steps processed and sampled.
2. **`TelemetryStore`** (`src/store.ts`). It keeps the buckets and the newest steps in ring buffers. Each request row also gets its cached tokens, its token times, its end time and its blocks.
3. **`stripModel`** (`src/strip.ts`). It puts the newest steps into columns. Each request gets a lane, and a lane that is free again goes to a later request.
4. **`tableRows`** (`src/table.ts`). It gives the rows for one tab, with TTFT, TPOT, blocks, cache-hit % and age.
5. **Panels** (`src/ui/`). `StoreCanvas` draws a canvas in each animation frame when the store changed. `RequestsTable` renders only the rows in view.

## What you get at the end

- `pnpm -C terminal dev` shows the three new panels on the `steady-chat` fixture at `http://127.0.0.1:5173`.
- `?trace=telemetry/prefix-pressure.jsonl.gz` also shows preempted requests, cache hits and blocks.
- New tests prove the buckets, the lanes, the tabs, the row window and the drawn shapes.

What you do not get yet:

- Play, pause and a scrubber. They arrive in M2-T9.
- A frame-rate test in a real browser. It arrives in M2-T10.
- Blocks for a live engine. The engine sends `kv.snapshot` from M3.

## Where this puts the project

```
M0  █████████
M1  ████████
M2  ██░░░░▓░░░░░░
```

This slice opens M2-T9 and M2-T10 for Lane B.

## Choices the agent made

You can still change these before `/ship`. Decision 0021 gives the reasons.

- The agent draws the step chart with Canvas 2D and adds no chart library. You chose this from the agent's recommendation.
- Under the candles, two bar graphs show tokens per second. One counts processed tokens and one counts sampled tokens. You asked for both.
- The batch strip gives each request a lane, not a fixed row. The strip then needs about `max_num_seqs` rows.
- The blocks column counts the blocks that a request owns alone, from the newest `kv.snapshot`. Shared blocks have no single owner.
- A running request shows a TPOT from its step times. A finished request shows the TPOT of the engine.

## What can go wrong

- The newest bucket is not complete. Its candle and bars grow until the second ends.
- A column in the strip is one step, not one unit of time. A long prefill step is as wide as a short decode step. The candle wick shows the time.
- The table has a fixed row height. A row with a longer text gets cut.

## Words

| Word | Meaning |
|---|---|
| bucket | The steps that ended in one second of trace time, folded into one candle and two bars. |
| candle | Open, high, low and close of the step latency in one bucket. |
| lane | One row of the batch strip. It holds one request at a time. |

## Learn from this slice

**Fold, then draw.** The store folds each step into a bucket when the frame arrives. The canvas only reads the buckets. A draw then costs the same for a long run as for a short run.

**Draw outside React.** React renders each canvas element one time. A loop on `requestAnimationFrame` draws the canvas when the store version changes.

Test yourself: why does the strip use lanes and not one row for each request?

<details><summary>Answer</summary>
The strip shows hundreds of steps, and many requests start and end in that time. One row each would need many rows. Not more than `max_num_seqs` requests are in one step, so lanes need about that many rows.
</details>

## Result

**State:** DONE. The review was NOT READY at `d8b4d99`, and the fix commits up to `1d6b91f` close every Required finding. The proof was NOT VERIFIED at `d8b4d99` for claim 3 as the agent wrote it. You accepted the corrected claim, and `2babb83` tests it. `make ci` passes at `2babb83`. This change used 2 agent runs.

**Against the plan:** the build followed the plan. You chose two bar graphs, so the chart has three panes.

**What proof and review found:**

- The agent told the verifier that the tab counts always equal the rows in the feed. That is false past 200 requests. The feed shows the newest 200 arrivals, and the table shows every request that the store keeps. A new test now covers more than 200 requests.
- A comment said that React renders the canvas one time. React renders the panel again on each update, but it keeps the same canvas element. The comment and decision 0021 now say this.
- Three tests did not fail when the reviewer changed the code. Now each of the three changes makes a test fail.
- Decision 0021 gave the prefill median as 523. The correct median is 510.5.

**What the verifier saw:** all 46 candles of `steady-chat` match the trace. When the verifier deleted one line in `timeline.ts`, 14 candles did not match. The strip has 400 cells of 400. The canvases change while the replay runs and stop when it ends.

**What is not proven:**

- A window resize, and screens with a `devicePixelRatio` other than 1.
- The live stream. M2-T4 brings `/telemetry/stream`.
- The frame rate. M2-T10 brings Playwright and the budget.

**Read these yourself:**

- `terminal/src/strip.ts:21-44`: the reuse of lanes and the empty column between two requests.
- `terminal/src/store.ts:191-210`: the decoding of `kv.snapshot`, and blocks set to null at the end.
- `terminal/src/ui/StoreCanvas.tsx:13-43`: the loop on `requestAnimationFrame`, the resize, and the stable `draw`.
