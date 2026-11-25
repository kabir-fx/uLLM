# M2-T11: Grafana dashboard

**State:** in build. You read this brief while the agent builds the slice.
**Decision record:** [0029](../decisions/0029-grafana-dashboard.md) (the full design, for agents) · **Spec:** roadmap M2-T11 (O3)

## The goal

You run one command and get Prometheus and Grafana in Docker. Grafana shows one dashboard with the live metrics of `ullm serve` on your Mac.

## The problem

- `GET /metrics` exists (decision 0024), but nothing stores or draws it.
- Engines never run in Docker on the Mac, because the Linux VM cannot reach the Metal GPU (O3).
- `ullm serve` binds to 127.0.0.1 only (rule 13). A container must reach it through `host.docker.internal`.
- `TrustedHostMiddleware` refuses every `Host` but `127.0.0.1` and `localhost`. A scrape from Docker sends `Host: host.docker.internal:8000` and gets 400.
- Prometheus does not let a scrape set its own `Host` header.

## How it works

```
Mac:     ullm serve 127.0.0.1:8000 ◀── GET /metrics, Host: host.docker.internal:8000
                                          │ every 5 s
Docker:  Prometheus 127.0.0.1:9090 ───────┘
              ▲
         Grafana 127.0.0.1:3000 ── provisioned data source and dashboard
```

1. `src/ullm/server/app.py`: `LOCAL_HOSTS` adds `host.docker.internal`. The bind stays 127.0.0.1.
2. `deploy/docker-compose.yml`: Prometheus and Grafana, each image pinned by digest. Each port binds to 127.0.0.1.
3. `deploy/prometheus/prometheus.yml`: one job that scrapes `host.docker.internal:8000` every 5 s.
4. `deploy/grafana/provisioning/`: the Prometheus data source and the dashboard provider.
5. `deploy/grafana/dashboards/ullm.json`: the dashboard.
6. `make grafana` starts the stack. `make grafana-down` stops it.

The agent measured two facts before the design:

- A container reached a server on the Mac that binds 127.0.0.1 only. The server saw peer `127.0.0.1` and `Host: host.docker.internal:8765`.
- `promtool check config` on Prometheus 3.7.3 refused a `Host` entry in `http_headers`: "setting header \"Host\" is not allowed".

## What you get at the end

Run `uv run ullm serve --fake`, then `make grafana`, and open `http://127.0.0.1:3000/`. The dashboard shows throughput, finished requests, TTFT and TPOT p50 and p99, and batch size. It also shows queue depth, prefix hit rate, and preemptions. Batch size and queue depth show one line for each `instance`, so they show the load of each replica.

New tests:

- A request with `Host: host.docker.internal:8000` gets `/metrics`. A foreign host still gets 400.
- Each metric name in the dashboard is one that `Metrics` exports.

What you do not get yet:

- KV usage: M3-T4 adds the metric and its panel.
- Breaker state and router load per replica: M5-T1 adds the metrics and their panels.

## Where this puts the project

```
M2  T1 █  T2 █  T3 █  T4 █  T5 █  T6 █  T7 █  T8 █  T9 █  T10 █  T11 ▓  T12 ░  T13 ░
```

After this slice, M2-T12 stays open in Lane A.

## Choices the agent made

You chose the panel scope: only metrics that exist now. You can still change these choices before `/ship`:

- `host.docker.internal` joins `LOCAL_HOSTS` for all paths. A proxy container that rewrites `Host` is the other option. It adds a third container for one header.
- Grafana lets anonymous users view, and only on 127.0.0.1. No password goes in the repo.
- The scrape target is port 8000, the default of `ullm serve`.

## What can go wrong

- `ullm serve` is not running. The Prometheus target shows "down", and the panels show "No data".
- `ullm serve --telemetry off` answers 404 on `/metrics`. The target shows "down".
- A page that a DNS rebind brings to 127.0.0.1 still arrives under its own name, so the server refuses it.

## Words

| Word | Meaning |
|---|---|
| scrape | One `GET /metrics` from Prometheus. |
| provisioned | Grafana reads the item from a file at start, so you do not add it by hand. |
| `host.docker.internal` | The name that a Docker Desktop container uses for the Mac. |

## Learn from this slice

A histogram gives percentiles because Prometheus counts samples in buckets. `histogram_quantile` finds the bucket that holds the quantile and interpolates in it. A p99 is therefore only as exact as the bucket bounds near it.

A `rate` needs two samples in its window. With a 5 s scrape, a window of 30 s holds about six samples.

Test yourself: why does the dashboard not show an average TTFT?

<details><summary>Answer</summary>

O2 says that percentiles come from histograms, never averages. One slow request moves an average but hides in it. The p99 shows the slow tail.

</details>

## Result
