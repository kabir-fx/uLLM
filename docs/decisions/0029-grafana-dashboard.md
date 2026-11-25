# 0029: Grafana dashboard for `ullm serve` on the Mac

**Status:** accepted (the author chose the panel scope; the agent chose the rest, open to change before the merge) · **Task:** M2-T11 (O3) · **Builds on:** 0024 (Prometheus metrics)

## Context

- O3 asks for `deploy/docker-compose.yml` with Prometheus scraping the host every 5 seconds and Grafana with one provisioned dashboard. Engines never run in Docker on the Mac: the Linux VM cannot reach the Metal GPU.
- `ullm serve` binds 127.0.0.1 only (rule 13), and `TrustedHostMiddleware` allowed `Host` values `127.0.0.1` and `localhost` only, so a scrape from Docker got 400 (0024, Consequences).
- Measured on Docker Desktop 28.0.4: a container reached a server on the Mac bound to 127.0.0.1 through `host.docker.internal`. The server saw peer `127.0.0.1` and `Host: host.docker.internal:8765`.
- Measured: `promtool check config` on Prometheus 3.7.3 refuses `http_headers: {Host: ...}` with "setting header \"Host\" is not allowed". A scrape cannot present itself as `localhost`.
- O3's panel list names three metrics that do not exist yet: KV usage (`kv_blocks`, M3-T4), breaker state and router load per replica (M5-T1).

## Decision

1. **`LOCAL_HOSTS` adds `host.docker.internal`**, for every path. The bind stays 127.0.0.1. A DNS-rebound page still arrives under its own name and gets 400 (`tests/integration/test_metrics.py`).
2. **`deploy/docker-compose.yml`**, project `ullm-observe`: `prom/prometheus:v3.7.3` and `grafana/grafana:12.3.0`, each pinned by digest (rule 12). Ports bind to 127.0.0.1: Prometheus 9090, Grafana 3000. `make grafana` starts it, `make grafana-down` stops it.
3. **`deploy/prometheus/prometheus.yml`**: one job, `ullm`, scrapes `host.docker.internal:8000` every 5 s. Port 8000 is `ullm serve`'s default.
4. **Grafana provisioning**: a Prometheus data source with uid `prometheus` and `timeInterval: 5s`, so `$__rate_interval` is at least 20 s, four scrapes. A file provider loads `deploy/grafana/dashboards/ullm.json`, which is also the home dashboard. Anonymous users view; no password is in the repo. Usage reporting and update checks are off.
5. **Panels, on metrics that exist now** (the author's choice): throughput (output and prompt tok/s), requests finished by reason, TTFT p50 and p99, TPOT p50 and p99, batch size and queue depth by `instance` (load per replica), prefix hit rate (cached over cached plus prefilled prompt tokens), preemptions. Percentiles use `histogram_quantile` over `_bucket` rates (O2).
6. **`tests/unit/test_dashboard.py`** fails when the dashboard queries a name `Metrics` does not export, or a latency without `histogram_quantile`.
7. **Later panels**: M3-T4 adds KV usage when `kv_blocks` exists, M5-T1 breaker state and router load per replica.

## Alternatives considered

- **A proxy container that rewrites `Host`** in front of the scrape. The server keeps its two names. Lost: a third container and its config for one header, and the proxy then needs the same `host.docker.internal` path anyway.
- **Accept `host.docker.internal` on `/metrics` only.** Narrower. Lost: `TrustedHostMiddleware` is app-wide, so it needs a hand-written middleware, and the name gives a rebound page nothing that `localhost` does not.
- **Prometheus on the Mac, not in Docker.** No `Host` problem. Lost: O3 names compose, and a local install is not pinned by the repo.
- **Panels for metrics not built yet.** All ten O3 panels now. Lost: the router's names are not settled, and panels on guessed names drift.

## Consequences

- Any local client may now send `Host: host.docker.internal`. It reaches the same 127.0.0.1 socket a `localhost` client does.
- With `ullm serve` stopped or `--telemetry off` (404), the target shows down and the panels show "No data".
- A server on a port other than 8000 needs the target edited.
- Measured once with `ullm serve --fake` and 6 closed-loop clients for 75 s: the target was up at 5 s, every panel returned data through Grafana, and Prometheus's `ullm_requests_total` (2,167) and `ullm_generation_tokens_total` (138,688) equalled the client's totals. No README figure comes from it.
