# 0024: Prometheus metrics from the telemetry hub

**Status:** accepted (the author chose the recommended answer to each of four questions before any logic was written) · **Task:** M2-T5 (O2)

## Context

- O2 asks for `/metrics` on every engine: histograms for TTFT, TPOT, end-to-end time, step time by kind and queue wait; gauges for queue depth, batch size and KV blocks; counters for tokens, requests by finish reason, prefix-cache hits, preemptions, aborts and dropped telemetry. Percentiles come from histograms. Proof: after a run, the counters equal the load generator's totals.
- Rule 2: only the engine thread touches engine state, and `core` has no locks. Rule 5: the engine thread never blocks on metrics. Rule 6: no request id as a label.
- Every number O2 needs is already in a C2 event the engine emits (decision 0003), and the `Hub` drains them on the event loop's thread (decision 0023).
- `prometheus-client` 0.26.0 takes a `threading.Lock` per metric value (`prometheus_client/values.py`, `MutexValue`). `Counter("requests", ...)` exports `requests_total` and `requests_created` (`metrics.py`, `Counter._metric_init`).
- `bench/run.py` already saves `/metrics` to `metrics.txt` after a run when the server answers 200 (C5, decision 0013).
- Not built yet: KV blocks (M3), the router (M5). M3-T4 owns "memory metrics"; M5-T1 owns the router's (R8).

## Decision

1. **`Metrics`** (`ullm/telemetry/metrics.py`) owns its own `CollectorRegistry`, so two apps in one process count apart. `observe(events, dropped)` maps C2 events to metrics. `render(waiting=, running=)` sets the two gauges and returns the text exposition. Both run on the loop thread only, so the library's locks are never contended by the engine.
2. **`Hub(..., observe=)`** calls the observer with each non-empty drain, before the subscriptions. The observer sees every event the bus kept, whatever a slow tab keeps.
3. **`GET /metrics`** (`server/app.py`) calls `hub.flush()`, then renders with `EngineClient.health()`'s `waiting` and `running`. The engine emits a request's events before it delivers that turn's outputs, so after the flush the counters include every reply a client has read. Without the flush, a scrape right after a reply missed it in 3 runs of 3 (`tests/integration/test_metrics.py`).
4. **No hub, no `/metrics`.** `ullm serve --telemetry off` and `create_app(..., emit=...)` answer 404, as for `/telemetry/stream`. The "off" case stays the drop sink that claims row 25 measured.
5. **Names and labels.** Namespace `ullm_`. Labels: `kind` on `step_seconds`, `finish_reason` on `requests_total`, `reason` on `aborts_total`. Every label value from the C2 `Literal`s exists at zero from the start. No `replica` label: Prometheus adds `instance` per scrape target.
6. **What each metric reads.**

   | Metric | Event | Notes |
   |---|---|---|
   | `ttft_seconds` | `req.first_token.ttft_ms` | observed at the first token, so a request aborted later still counts |
   | `tpot_seconds` | `req.finished.timings.tpot_ms` | absent with one output token |
   | `e2e_seconds` | `req.finished.timings.e2e_ms` | finished requests only |
   | `step_seconds{kind}` | `step.ms` | |
   | `queue_wait_seconds` | `req.admitted.queue_ms` | |
   | `queue_depth`, `batch_size` | `EngineClient.health()` at scrape | the last `step` goes stale when the engine idles |
   | `prompt_tokens_total` | `step.tokens`, prefill steps | work done |
   | `generation_tokens_total` | one per `req.first_token`, `len(step.rids)` per decode step | work done; the first token is counted at `req.first_token` so a chunked prefill step (E13) that samples nothing stays right |
   | `prefix_cache_hit_tokens_total` | `req.admitted.cached_tokens` | 0 until M3-T5 |
   | `requests_total{finish_reason}` | `req.finished.reason` | |
   | `preemptions_total` | `req.preempted` | 0 until M3-T6 |
   | `aborts_total{reason}` | `req.aborted.reason` | the label is added to O2's list: it tells a client leaving from an error |
   | `telemetry_dropped_total` | the bus's `dropped` | events lost before every reader, so missing from the counters too |

7. **Token counters count work done**, as vLLM's do: an aborted request's prefill and sampled tokens count. After a run with no aborts, they equal the sum of `usage` over the replies.
8. **Buckets** are set around uLLM's M1 figures at concurrency 1 (TTFT p50 177.9 ms, TPOT p50 16.5 ms, `bench/results/m1/c1/summary.md`): a design choice, not a measurement. They are constants in `metrics.py`.
9. **Out of this task:** `kv_blocks{state}` and `kv_waste_ratio` (M3-T4), router metrics (M5-T1).

## Alternatives considered

- **Update metrics on the engine thread**, with a tee of `emit`. No hub dependency, and it works with telemetry off. Lost: it takes the library's lock on the engine thread, which a scrape can hold (rules 2 and 5).
- **Metrics as a `Subscription`.** Reuses the reader type. Lost: a subscription is bounded and drops its oldest events when its reader is slow, so the counters would undercount for a reason other than the bus.
- **Keep the hub running with `--telemetry off`**, for metrics alone. Lost: it changes what "off" measured in claims row 25.
- **Hand-written exposition.** No dependency. Lost: about 80 more lines to read and test, for a format the library already gets right.
- **Count tokens from `req.finished.usage`.** Equal to what clients saw exactly. Lost: an aborted request's work never shows, so throughput panels undercount under disconnects.

## Consequences

- One new dependency: `prometheus-client==0.26.0`, pure Python, no required dependencies.
- If the bus overflows, the counters undercount by the events lost; `ullm_telemetry_dropped_total` shows it.
- `TrustedHostMiddleware` allows only `127.0.0.1` and `localhost`. A Prometheus in Docker (M2-T11) sends another `Host` and gets 400; that task must settle how it scrapes.
- Every bench run against `ullm serve` now writes `metrics.txt`, which M2-T6 can read.
