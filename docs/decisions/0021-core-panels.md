# 0021: The step chart, batch strip and requests table

**Status:** accepted · **Task:** M2-T7 (T2)

## Context

T2 adds three panels to the shell from decision 0017: a step chart (one candle per second of step latency, with token bars under it), a batch strip (rows are requests, columns are steps) and a requests table with Running, Waiting, Finished and Preempted tabs. T1 forbids telemetry in React state per event and asks for canvases on `requestAnimationFrame`. T7 budgets 15 frames a second and 200 table rows. Section 6.6 suggested a canvas chart library, lightweight-charts or uPlot.

Facts this design rests on:

- lightweight-charts 5.3.0 (Apache-2.0) requires the attribution notice and a link to tradingview.com on the page that shows the chart (its `README.md`, line 128, on unpkg). uPlot 1.6.32 has no candlestick series.
- In `steady-chat`, a prefill step processes 510.5 tokens (median of 74) and samples 1; a decode step processes and samples one per row, up to 8. On one y-axis for processed tokens, the decode bars are a thin base under the prefill bars (seen in a screenshot of the build).
- `step.t` is the step's end, after `mx.eval` (C2). The engine's TPOT is `(t_last − t_first) / (n − 1)` (`ullm/core/request.py:157`).
- `kv.snapshot.states` is base64, one byte per block; byte `3 + i` is a block owned by `owners[i]` alone, and shared blocks (byte 2) name no owner (`ullm/telemetry/events.py`, `BLOCK_OWNED_BASE`).

## Decision

**Layout.**

| File | Owns |
|---|---|
| `src/timeline.ts` | `Second` (one bucket per second of trace time: open, high, low, close of `step.ms`; processed and sampled tokens split by kind) and `foldStep` |
| `src/strip.ts` | `stripModel(steps, columns)`: cells by column and lane |
| `src/table.ts` | `TABS`, `tableRows(store, tab)`, `tabCounts(store)` |
| `src/ui/draw.ts` | `drawStepChart`, `drawStrip`, against `Pen`, the subset of `CanvasRenderingContext2D` they use |
| `src/ui/StoreCanvas.tsx` | One canvas element that React keeps across renders, drawn by a loop on `requestAnimationFrame` when `store.version` or its size changed |
| `src/ui/Panels.tsx` | `StepChart` and `BatchStrip`: a heading with a colour key, and a `StoreCanvas` |
| `src/ui/RequestsTable.tsx`, `src/ui/window.ts` | The tabs and the windowed table, `visibleRange` and `ROW_HEIGHT` (20 px) |

**Store.** `TelemetryStore` gains `steps` (the newest 1,024 steps) and `seconds` (the newest 600 buckets), both cleared at `hello`. `lastStep` is now the newest of `steps`. Each `RequestRow` gains `cachedTokens` (from `req.admitted`), `firstTokenT`, `lastTokenT` (the newest step with the request as a row), `endT`, `tpotMs` (from `req.finished`) and `blocks`. A `kv.snapshot` sets `blocks` on every request in flight to the blocks it owns alone, 0 when absent; `blocks` is null before the first snapshot and after the request ends.

**Step chart, own Canvas 2D.** No chart library. The newest second sits at the right edge, one column of 7 px per second; a second with no step is a gap. Three panes, each scaled to its own largest value in view and labelled with it: candles of step latency (top half), processed tokens per second, sampled tokens per second. Each bar stacks prefill on decode. A candle whose latency closed above its open is `--rise`, else `--fall`. The newest bucket is still filling.

**Batch strip.** The newest steps that fit at 4 px per column, the newest at the right. Each request takes the lowest free lane at its first column in view and holds it until its last column; one empty column then separates it from the next request in that lane. A lane is at most 12 px high.

**Requests table.** Tabs: Running holds prefilling and running, Finished holds finished and aborted. Running, Waiting and Preempted list the oldest arrival first; Finished lists the newest end first. Columns: request, state, prompt, generated, TTFT, TPOT, blocks, cache hit (`cached_tokens / prompt_tokens`), age (`(endT ?? nowT) − arrivedT`). TPOT is the engine's for a finished request, and `(lastTokenT − firstTokenT) / (tokens − 1)` otherwise, null below 2 tokens. Rows are a fixed 20 px; only the rows in view plus 5 on each side render, with spacer rows for the rest of the scroll height.

These figures are for watching. They are never a claim (rule 8).

## Alternatives considered

- **lightweight-charts.** It brings a crosshair and zoom, but puts a TradingView link on the terminal, which 6.6 says must keep its own identity, and adds a dependency for one panel. The batch strip has to be hand-drawn anyway, so both canvases now share one loop and one test method.
- **uPlot.** Candles need a custom draw hook, so most of the drawing would be ours anyway.
- **One bar graph.** Processed tokens hide decode, sampled tokens hide prefill. The author chose both graphs, to prune one later.
- **One row per request in the strip.** A window of hundreds of steps sees dozens of requests; lanes need about `max_num_seqs` rows.
- **Blocks from `kv_tokens / block_size`.** It counts shared prefix blocks once per request and is wrong once prefix caching starts.
- **A virtual list library.** Fixed-height windowing is eight lines in `window.ts`.

## Consequences

- Section 6.6 now names Canvas 2D for the step chart, and T2 names the two bar graphs.
- A canvas is not tested in Node: `draw.test.ts` records the rectangles drawn through `Pen`. The `requestAnimationFrame` loop and the layout are proved in a browser. M2-T10 adds Playwright and the frame-rate budget.
- M3-T8 (T5) reuses `StoreCanvas` for the KV grid and the snapshot decoding in the store.
- No new dependency.
