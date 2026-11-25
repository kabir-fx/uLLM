# 0029: Grafana dashboard for `ullm serve` on the Mac

**Status:** accepted (the author chose the panel scope and where the server accepts the Docker host name; the agent chose the rest, open to change before the merge) · **Task:** M2-T11 (O3) · **Builds on:** 0024 (Prometheus metrics)

## Context

- O3 asks for `deploy/docker-compose.yml` with Prometheus scraping the host every 5 seconds and Grafana with one provisioned dashboard. Engines never run in Docker on the Mac: the Linux VM cannot reach the Metal GPU.
- `ullm serve` binds 127.0.0.1 only (rule 13), and `TrustedHostMiddleware` allowed `Host` values `127.0.0.1` and `localhost` only, so a scrape from Docker got 400 (0024, Consequences).
- Measured on Docker Desktop 28.0.4: a container reached a server on the Mac bound to 127.0.0.1 through `host.docker.internal`. The server saw peer `127.0.0.1` and `Host: host.docker.internal:8765`.
- Measured: `promtool check config` on Prometheus 3.7.3 refuses `http_headers: {Host: ...}` with "setting header \"Host\" is not allowed". A scrape cannot present itself as `localhost`.
- Measured: `host.docker.internal` is not in the Mac's `/etc/hosts`, and `dscacheutil -q host -a name host.docker.internal` returns nothing. A browser on the Mac resolves it through the network's DNS, so a hostile resolver can answer it and then rebind it to 127.0.0.1. `localhost` never leaves the Mac.
- Measured: Grafana 12.3.0 with only anonymous Viewer on accepts any `Host`, and `POST /login` with `admin/admin` under `Host: evil.test` returned 200. A rebound page could sign in as server admin and add a data source.
- O3's panel list names three metrics that do not exist yet: KV usage (`kv_blocks`, M3-T4), breaker state and router load per replica (M5-T1).

## Decision

1. **`host.docker.internal` reaches `GET /metrics` only.** `_LocalHosts` (`server/app.py`) runs Starlette's `TrustedHostMiddleware` with `LOCAL_HOSTS` (`127.0.0.1`, `localhost`) on every request, except `GET /metrics` with a `Host` that fully matches `host.docker.internal[:port]`. Under that name every other path, `/metrics/` included, gets 400. The bind stays 127.0.0.1. `tests/integration/test_metrics.py` holds both sides.
2. **`deploy/docker-compose.yml`**, project `ullm-observe`: `prom/prometheus:v3.7.3` and `grafana/grafana:12.3.0`, each pinned by digest (rule 12). Ports bind to 127.0.0.1: Prometheus 9090, Grafana 3000. `make grafana` starts it, `make grafana-down` stops it.
3. **`deploy/prometheus/prometheus.yml`**: one job, `ullm`, scrapes `host.docker.internal:8000` every 5 s. Port 8000 is `ullm serve`'s default.
4. **Grafana provisioning**: a Prometheus data source with uid `prometheus` and `timeInterval: 5s`, so `$__rate_interval` is at least 20 s, four scrapes. A file provider loads `deploy/grafana/dashboards/ullm.json`, which is also the home dashboard. Anonymous users view. `GF_SERVER_ENFORCE_DOMAIN` with domain `127.0.0.1` redirects any other `Host`, so a rebound page gets a 301 it cannot read. The login form and basic auth are off, so no one signs in, and the default `admin/admin` is unusable. No password is in the repo. Usage reporting and update checks are off.
5. **Panels, on metrics that exist now** (the author's choice): throughput (output and prompt tok/s), requests finished by reason, TTFT p50 and p99, TPOT p50 and p99, batch size and queue depth by `instance` (load per replica), prefix hit rate (cached over cached plus prefilled prompt tokens), preemptions. Percentiles use `histogram_quantile` over `_bucket` rates (O2).
6. **`tests/unit/test_dashboard.py`** fails when the dashboard queries a name `Metrics` does not export, or a latency without `histogram_quantile`.
7. **Later panels**: M3-T4 adds KV usage when `kv_blocks` exists, M5-T1 breaker state and router load per replica.

## Alternatives considered

- **A proxy container that rewrites `Host`** in front of the scrape. The server keeps its two names. Lost: a third container and its config for one header, and the proxy then needs the same `host.docker.internal` path anyway.
- **Accept `host.docker.internal` on every path**, one more entry in `LOCAL_HOSTS`. One line. Lost: on a hostile network a rebound page could send prompts and read `/telemetry/stream`.
- **Accept it only with a `ullm serve` flag.** Closed by default. Lost: one more option, and a step to remember before `make grafana` shows data.
- **Prometheus on the Mac, not in Docker.** No `Host` problem. Lost: O3 names compose, and a local install is not pinned by the repo.
- **Panels for metrics not built yet.** All ten O3 panels now. Lost: the router's names are not settled, and panels on guessed names drift.

## Consequences

- On a hostile network, a rebound page can read `/metrics`: counts and latencies, no prompts or outputs. It cannot reach any other path.
- Prometheus on 127.0.0.1:9090 has no `Host` check and no auth, so a rebound page can read its stored metrics. Its admin and lifecycle APIs are off by default, so it is read-only.
- Grafana has no admin login. The dashboard and data source change only through the files in `deploy/grafana/`.
- With `ullm serve` stopped or `--telemetry off` (404), the target shows down and the panels show "No data". The prefix hit rate shows no data while idle, as 0 over 0.
- A server on a port other than 8000 needs the target edited.
- No figure is claimed from this task. The proof of counters against client totals sits in the pull request.
