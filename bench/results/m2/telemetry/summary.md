# VALID: m2-telemetry, concurrency 8

- Baseline `ullm-off` and its twin; engines `ullm-on`. One server per chunk.
- 24 prompts and 8 warm-up requests a chunk; settle 600 s; seed 0.
- Commit `fab197c68f59` (dirty tree). Apple M4, 10 cores, 16 GB, macOS 27.2.1; mlx 0.32.4, mlx-lm 0.32.2. At the start: AC True, battery 54%, Low Power Mode False. Over 31 probes: charging in 31, lid open in 31.

## Gates

- **Conditions**: passed. 1534 clock samples, 31 probes on AC.
- **Liveness**: passed. 10 of 10 blocks, ended completed; 0 of 960 requests failed; 0 prompt lengths differ.
- **Memory**: passed. peak GB of limit: ullm-off 3.2/8, ullm-off-twin 3.2/8, ullm-on 3.2/8; swap never grew.
- **Twin**: passed. Peak 1.0009 (limit 0.005); Sustained 1.0052 (limit 0.03).

## Regimes

Peak 1, Sustained 8, Transition 1 blocks. Transition blocks are in raw.jsonl and in no table.

## Peak

Ratio to `ullm-off`: median [95% block-bootstrap interval], prompts/blocks. Below 1 is faster for TTFT, TPOT and E2E; above 1 is faster for output tok/s.

| Engine | TTFT | TPOT | E2E | Output tok/s |
|---|---|---|---|---|
| ullm-on | 1.013 no interval, 24/1, within noise | 1.005 no interval, 24/1, within noise | 1.006 no interval, 24/1, within noise | 0.999 no interval, 1/1, within noise |
| ullm-off-twin | 1.006 no interval, 24/1 | 1.001 no interval, 24/1 | 0.999 no interval, 24/1 | 1.024 no interval, 1/1 |
| floor | n/a | n/a | n/a | n/a |

Absolute, ms: p50, p95, p99 [95% block-bootstrap interval of p99].

| Engine | Requests | TTFT | TPOT | E2E | Clock mix |
|---|---|---|---|---|---|
| ullm-off | 24 | 228.6, 1171.7, 1372.3 | 29.8, 33.9, 34.2 | 4509.4, 7916.6, 8963.3 | 1470 MHz 99%, 338 MHz 1%, 1338 MHz 1% |
| ullm-off-twin | 24 | 227.9, 1096.5, 1334.0 | 29.9, 34.0, 34.1 | 4185.0, 7554.3, 8610.8 | 1470 MHz 99%, 1338 MHz 1%, 338 MHz 0% |
| ullm-on | 24 | 219.7, 1214.3, 1394.1 | 30.0, 33.9, 34.3 | 4538.1, 7962.5, 9018.6 | 1470 MHz 98%, 1338 MHz 1%, 338 MHz 1% |

Streams: a stall is a gap over 100 ms between chunks. The footprint is the session's peak.

| Engine | Largest gap (ms) | Stream time in stalls | Peak footprint (GB) |
|---|---|---|---|
| ullm-off | 1313 | 26.03% | 3.17 |
| ullm-off-twin | 1273 | 26.91% | 3.17 |
| ullm-on | 1334 | 26.70% | 3.17 |

## Sustained

Ratio to `ullm-off`: median [95% block-bootstrap interval], prompts/blocks. Below 1 is faster for TTFT, TPOT and E2E; above 1 is faster for output tok/s.

| Engine | TTFT | TPOT | E2E | Output tok/s |
|---|---|---|---|---|
| ullm-on | 1.006 [0.975, 1.027], 192/8, within noise | 1.002 [0.977, 1.014], 192/8, within noise | 1.005 [0.985, 1.017], 192/8, within noise | 0.993 [0.981, 1.017], 8/8, within noise |
| ullm-off-twin | 1.004 [0.968, 1.031], 192/8 | 1.005 [0.985, 1.015], 192/8 | 1.007 [0.992, 1.018], 192/8 | 0.988 [0.980, 1.006], 8/8 |
| floor | ±0.032 | ±0.015 | ±0.018 | ±0.020 |

Absolute, ms: p50, p95, p99 [95% block-bootstrap interval of p99].

| Engine | Requests | TTFT | TPOT | E2E | Clock mix |
|---|---|---|---|---|---|
| ullm-off | 192 | 358.4, 2067.8, 2559.4 [2375.0, 2673.9] | 36.6, 48.0, 51.6 [48.6, 52.5] | 5912.5, 11187.9, 13726.2 [11850.1, 14309.7] | 928 MHz 32%, 796 MHz 28%, 1056 MHz 14% |
| ullm-off-twin | 192 | 353.7, 2076.6, 2606.3 [2393.0, 2706.7] | 36.9, 49.0, 52.8 [49.8, 52.9] | 5930.0, 11402.3, 12587.7 [11926.3, 12700.8] | 796 MHz 27%, 928 MHz 25%, 618 MHz 17% |
| ullm-on | 192 | 364.0, 2218.0, 2546.4 [2402.6, 2563.6] | 36.9, 49.6, 57.2 [50.8, 61.2] | 5970.0, 11084.7, 13916.1 [11996.1, 14442.4] | 928 MHz 33%, 796 MHz 29%, 618 MHz 14% |

Streams: a stall is a gap over 100 ms between chunks. The footprint is the session's peak.

| Engine | Largest gap (ms) | Stream time in stalls | Peak footprint (GB) |
|---|---|---|---|
| ullm-off | 2461 | 32.91% | 3.17 |
| ullm-off-twin | 2359 | 33.23% | 3.17 |
| ullm-on | 2447 | 33.28% | 3.17 |
