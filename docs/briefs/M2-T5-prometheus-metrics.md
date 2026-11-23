# M2-T5: Prometheus metrics

**State:** in build. You read the Result when `/build` reports.
**Decision record:** [0024](../decisions/0024-prometheus-metrics.md) (the full design, for agents) · **Spec:** roadmap O2, task M2-T5

## The goal

Each `ullm serve` gives `GET /metrics` in the Prometheus text format. Prometheus and Grafana (M2-T11) can then show TTFT, TPOT, batch size and queue depth for a replica.

## The problem

- Nothing gives counters or histograms now. `bench/run.py` asks for `/metrics` after each run, and the server answers 404.
- Rule 5: the engine thread never waits for metrics. Rule 2: threads talk through queues, and `core` has no locks.
- Rule 6: no request ID is a label. Labels have few values: `kind`, `finish_reason`, `reason`.
- O2 asks for percentiles from histograms, never from averages.
- O2 also asks for proof: after a run, the counters equal the totals of the load generator.

## How it works

```
ENGINE THREAD                    API THREAD (asyncio loop)
Engine ──emit──▶ Bus ──drain──▶ Hub ──observe──▶ Metrics ──▶ GET /metrics
                                  └──put──▶ Subscription ──▶ /telemetry/stream, record()
EngineClient.health() ── waiting, running ──────────────────▶ gauges, read at scrape time
```

1. `Metrics`, in `src/ullm/telemetry/metrics.py`, owns one `CollectorRegistry` with every O2 engine metric. Its `observe` method reads C2 events and updates the counters and histograms. It runs on the API thread only.
2. `Hub`, in `src/ullm/telemetry/bus.py`, gives each drain to `Metrics` before the subscriptions. `Metrics` sees every event that the bus kept, also when a tab is slow.
3. `GET /metrics`, in `src/ullm/server/app.py`, flushes the hub first. Thus a reply that a client has read is always in the counters. Then it sets `queue_depth` and `batch_size` from `EngineClient.health()`, and it renders the registry.
4. `ullm serve --telemetry off` has no hub, so `/metrics` answers 404. The "off" case stays the base that claims row 25 measured.

The agent read the installed `prometheus-client` 0.26.0 source. `Counter("requests", ...)` adds the `_total` suffix itself. Each metric takes a lock, but only the API thread updates or reads it.

## What you get at the end

- `ullm serve --fake`, one chat request, then `curl http://127.0.0.1:8000/metrics` shows `ullm_requests_total{finish_reason="stop"}` and the histograms.
- Each `bench/run.py` run against uLLM writes `metrics.txt` into its result directory (C5).
- New tests prove these things:
  - each event type moves the correct metric by the correct amount
  - the hub gives `Metrics` every event, and the count of dropped events
  - a scrape just after a reply already counts that reply
  - `/metrics` answers 404 with `--telemetry off`
  - after a bench run on `ullm serve --fake`, the request and token counters equal the totals in `raw.jsonl`

**What you do not get yet:** M3-T4 brings `kv_blocks{state}` and `kv_waste_ratio`. M5-T1 brings the router metrics (R8). M2-T11 brings the Grafana dashboard.

## Where this puts the project

```
M2  T1 █  T2 █  T3 █  T4 █  T5 ▓  T6 ░  T7 █  T8 ░  T9 ░  T10 ░  T11 ░  T12 ░  T13 ░
```

This slice opens M2-T11 (Grafana) and gives M2-T6 a `/metrics` snapshot for each run.

## Choices the agent made

You chose the answer to each question below before the build. You can still change them before `/ship`.

1. **The hub feeds the metrics.** The other design updates metrics on the engine thread. That puts a lock on the engine thread, which rules 2 and 5 forbid.
2. **`--telemetry off` also turns off `/metrics`.** Thus the measured "off" case does not change.
3. **The `prometheus-client` library, pinned at 0.26.0.** It has no required dependencies. Text that the agent writes by hand gives more code to read and to test.
4. **Token counters count the work that the engine did.** `ullm_prompt_tokens_total` adds the tokens of each prefill step. `ullm_generation_tokens_total` adds each sampled token. Thus a request that a client stops also counts.
5. **Two counters read 0 until M3.** `ullm_prefix_cache_hit_tokens_total` and `ullm_preemptions_total` read real events, and no event gives them a value yet.
6. **Names start with `ullm_`.** This is the Prometheus convention for a namespace. No `replica` label, because Prometheus adds `instance` itself.
7. **Histogram buckets come from the M1 figures.** For example, the TPOT buckets are close together near 16.5 ms. The buckets are a design guess, not a measurement.

## What can go wrong

- If the bus drops events, the counters miss them. `ullm_telemetry_dropped_total` then shows how many events the bus lost.
- A Prometheus in Docker sends a `Host` header other than `127.0.0.1`. The server then refuses the scrape. M2-T11 must solve this.
- Each counter restarts at 0 when the server restarts. Prometheus `rate()` handles this.

## Words

| Word | Meaning |
|---|---|
| scrape | One `GET /metrics` from Prometheus. |
| counter | A number that only goes up, such as `ullm_requests_total`. |
| histogram | Counts of values in fixed buckets. Prometheus calculates percentiles from them. |
| gauge | A number that goes up and down, such as `ullm_queue_depth`. |

## Learn from this slice

The telemetry events already have each number that the metrics need. Thus the metrics read the same events as the terminal. The engine does no new work for them.

A scrape must agree with what the client saw. The engine puts an event on the bus before it sends the reply. Thus a flush at scrape time gives each reply that a client has read.

Test yourself: why does `/metrics` read `batch_size` from `EngineClient.health()`, and not from the last `step` event?

<details><summary>Answer</summary>

After the last request finishes, the engine does no more steps. The last `step` event still shows the old batch. `health()` shows the load after each turn of the engine, so an idle engine shows 0.

</details>

## Result

