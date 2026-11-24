# 0027: Terminal quality gates

**Status:** accepted · **Task:** M2-T10 (T7)

## Context

T7 sets three budgets: 60 fps while receiving 15 frames a second with about 2,300 blocks and 200 table rows; flat memory over a long session; the hosted replay becomes interactive quickly. Its proof is a frame-rate test, a Playwright smoke test driven by a replay trace, and keyboard basics. Decisions 0017 and 0022 deferred every browser test to this task.

Facts this design rests on:

- No browser test exists. UI tests render with `react-dom/server` under Node (decision 0017).
- The KV block grid is T5, which lands in M3. The store already folds each `kv.snapshot` into the requests table's `blocks` column (`TelemetryStore.applySnapshot`, `terminal/src/store.ts`).
- The synthetic generator flushes a frame each 1/15 s (`FRAME_US`, `src/ullm/telemetry/synthetic.py`).
- `@playwright/test` 1.65.1 is the newest release (`pnpm view @playwright/test version`). Its Chromium is Chrome Headless Shell 159.0.8154.6.
- A spike on the author's M4 replayed a 2,304-block trace in Chromium for 10 s at about 25 s into the trace: 60.0 fps headless and 60.0 fps headed, frame interval p99 16.8 ms and 17.7 ms.
- Roadmap 4 says CI timings mean nothing and no benchmark runs in CI.
- A replay holds its whole trace in memory (`replaySource`'s `frames`), so a leak that keeps frames costs a replay nothing, and a 10× replay stream holds about 18 MB that hides a small leak. A first design sampled the heap over a 10× replay stream; with a store that kept every ended request, it still gave 0.99× (review of M2-T10).
- Live mode (`liveSource`) keeps no frames. Fed the stress trace through a routed SSE body, its heap after a forced GC is 3.6 MB after 2 repeats, 4.13 MB after 20 and 4.13 MB after 40. The TTFT and TPOT rings hold 4,096 samples and the trace adds 218 a repeat, so the rings are full after 19 repeats. Between 20 and 40 repeats, a store that keeps every ended request grew the heap 1.16×, and a page that keeps every frame 1.85×.

## Decision

**The author's calls.** The frame-rate test covers the panels that exist; the task that builds the T5 grid adds it to the same test. Smoke and keyboard tests run in `make check-ts`, and so in CI. The budgets run locally with `make terminal-perf`. Time to interactive is at most 1 s on a local preview of the hosted build, no throttling, median of 5 cold loads. Keyboard basics are replay shortcuts, WAI-ARIA tabs and a visible focus ring.

**Stress trace.** A third synthetic scenario, `stress` (seed 3): open-loop arrivals at 4 a second, `max_num_seqs` 64, a 2,304-block pool. It gives 984 frames over about 70 s, 218 requests, at least 200 finished. `tests/unit/test_synthetic_traces.py` holds that shape. It is a fixture, not a bundled run, so the run picker does not show it.

**Playwright.** `terminal/playwright.config.ts` has two projects. `smoke` runs `terminal/e2e/*.spec.ts` against `vite preview` of `dist-hosted/` on 127.0.0.1:4183: the hosted replay opens in a clean browser, plays, draws, and works from the keyboard. `perf` runs `terminal/perf/*.spec.ts`; it also uses `vite preview` of `dist/` on 127.0.0.1:4184, because the local build accepts `?trace=`. `make check-ts` installs Chromium (a no-op when present) and runs `smoke`. `make ci-terminal` first installs Chromium with its system libraries, which the Linux runner needs.

**Budgets, as `make terminal-perf` checks them.** Each is a pass or fail, and the measured values go to `bench/results/m2/terminal-perf/result.json`, with `env.json` from `bench.conditions`.

| Budget | Measured as | Pass |
|---|---|---|
| 60 fps | `requestAnimationFrame` timestamps over the whole `stress` trace at speed 1, with the finished tab open | mean ≥ 58 fps and p99 frame interval ≤ 25 ms |
| Flat memory | live mode, fed the `stress` trace repeated as one stream (no `hello` between repeats; `seq`, `t` and request ids shifted) as a routed SSE body; JS heap and DOM nodes after a forced GC once the stream ends; 3 cold loads at 20 repeats and 3 at 40 | median at 40 ≤ 1.05 × median at 20, for the heap and for DOM nodes |
| Interactive | from navigation start to the replay panel showing a duration above zero, on the hosted build, 5 cold browser contexts | median ≤ 1,000 ms |

The base is 20 repeats, not 1, because the rings are full only after 19. The ports 4183 and 4184 keep the tests apart from a `vite preview` on 4173.

**Keyboard.** In replay mode, a `keydown` listener on `document` maps Space to play or pause, and ← and → to a seek of 5 s of trace time. It ignores a key with a modifier, and a key aimed at a form control, a button or a tab, which keeps their own behaviour. The mapping is a pure function in `terminal/src/keys.ts`. The requests tabs follow the WAI-ARIA tabs pattern: only the selected tab is in the Tab order, and ←, →, Home and End move the selection and focus. `:focus-visible` draws a 2 px `--accent` outline on every control.

## Alternatives

- **The frame-rate test waits for T5.** Rejected by the author: the panels that exist need a budget now.
- **Budgets in CI.** Rejected: shared runners give noisy timings, and loose thresholds would weaken the bar.
- **No browser in CI.** Rejected: a broken hosted build would pass CI.
- **Memory over replay loops that reset the store.** Rejected: a `hello` clears the store, so its rings never fill and a leak in them stays hidden.
- **Memory over one long replay stream.** Rejected after review: the replay's copy of the trace hides a leak (Context).
- **Throttled time to interactive (4× CPU, Fast 4G).** Rejected by the author in favour of the local measure.

## Consequences

- The CI terminal job downloads Chromium on each run.
- `make check` needs Chromium; the first run downloads about 100 MB.
- The KV grid's task adds the grid to the frame-rate test and reruns `make terminal-perf`.
- Live mode is measured through replay: both feed the same store and panels.
