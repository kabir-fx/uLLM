# DISCARDED: m2-c16, concurrency 16

- Baseline `mlx-lm-batched-no-cache` and its twin; engines `ullm`. One server per chunk.
- 30 prompts and 16 warm-up requests a chunk; settle 600 s; seed 0.
- Commit `c51200e20812` (dirty tree). Apple M4, 10 cores, 16 GB, macOS 27.2.1; mlx 0.32.4, mlx-lm 0.32.2. At the start: AC True, battery 50%, Low Power Mode False. Over 25 probes: charging in 25, lid open in 25.

## Gates

- **Conditions**: passed. 1590 clock samples, 25 probes on AC.
- **Liveness**: passed. 8 of 8 blocks, ended completed; 0 of 1104 requests failed; 0 prompt lengths differ.
- **Memory**: FAILED. swap grew 1188 MB above the first probe's.
- **Twin**: passed. Peak 1.0048 (limit 0.005); Sustained 1.0081 (limit 0.03).

## Regimes

Peak 1, Sustained 6, Transition 1 blocks. Transition blocks are in raw.jsonl and in no table.

## Peak

Ratio to `mlx-lm-batched-no-cache`: median [95% block-bootstrap interval], prompts/blocks. Below 1 is faster for TTFT, TPOT and E2E; above 1 is faster for output tok/s.

| Engine | TTFT | TPOT | E2E | Output tok/s |
|---|---|---|---|---|
| ullm | 0.345 no interval, 30/1, within noise | 1.113 no interval, 30/1, within noise | 0.932 no interval, 30/1, within noise | 1.065 no interval, 1/1, within noise |
| mlx-lm-batched-no-cache-twin | 0.959 no interval, 30/1 | 1.005 no interval, 30/1 | 0.992 no interval, 30/1 | 1.019 no interval, 1/1 |
| floor | n/a | n/a | n/a | n/a |

Absolute, ms: p50, p95, p99 [95% block-bootstrap interval of p99].

| Engine | Requests | TTFT | TPOT | E2E | Clock mix |
|---|---|---|---|---|---|
| mlx-lm-batched-no-cache | 30 | 3788.9, 4037.9, 4038.0 | 49.0, 66.9, 68.4 | 10368.0, 17855.8, 17929.5 | 1470 MHz 99%, 1338 MHz 1%, 338 MHz 0% |
| mlx-lm-batched-no-cache-twin | 30 | 3636.0, 3874.4, 3875.7 | 50.7, 63.1, 70.4 | 10312.9, 17619.3, 17777.2 | 1470 MHz 99%, 1338 MHz 1%, 338 MHz 0% |
| ullm | 30 | 318.5, 2667.6, 2794.2 | 56.7, 76.9, 87.6 | 9658.0, 16915.2, 17065.9 | 1470 MHz 98%, 1338 MHz 1%, 338 MHz 0% |

Streams: a stall is a gap over 100 ms between chunks. The footprint is the session's peak.

| Engine | Largest gap (ms) | Stream time in stalls | Peak footprint (GB) |
|---|---|---|---|
| mlx-lm-batched-no-cache | 819 | 23.07% | 11.56 |
| mlx-lm-batched-no-cache-twin | 698 | 23.64% | 11.61 |
| ullm | 2788 | 32.32% | 5.08 |

## Sustained

Ratio to `mlx-lm-batched-no-cache`: median [95% block-bootstrap interval], prompts/blocks. Below 1 is faster for TTFT, TPOT and E2E; above 1 is faster for output tok/s.

| Engine | TTFT | TPOT | E2E | Output tok/s |
|---|---|---|---|---|
| ullm | 0.368 [0.354, 0.384], 180/6 | 1.095 [1.037, 1.209], 180/6, within noise | 0.923 [0.854, 0.960], 180/6, within noise | 1.064 [1.000, 1.156], 6/6, within noise |
| mlx-lm-batched-no-cache-twin | 1.019 [0.899, 1.285], 180/6 | 1.008 [0.920, 1.266], 180/6 | 1.006 [0.906, 1.253], 180/6 | 0.996 [0.780, 1.081], 6/6 |
| floor | ±0.285 | ±0.266 | ±0.253 | ±0.220 |

Absolute, ms: p50, p95, p99 [95% block-bootstrap interval of p99].

| Engine | Requests | TTFT | TPOT | E2E | Clock mix |
|---|---|---|---|---|---|
| mlx-lm-batched-no-cache | 180 | 4105.5, 5741.3, 5975.5 [5168.1, 5975.7] | 57.7, 69.9, 73.5 [70.9, 73.7] | 10435.0, 21602.2, 23994.4 [21798.5, 24294.0] | 1170 MHz 27%, 1056 MHz 25%, 928 MHz 17% |
| mlx-lm-batched-no-cache-twin | 180 | 4194.2, 7297.0, 7650.9 [5156.0, 7652.4] | 58.5, 90.8, 97.4 [67.9, 99.8] | 11440.0, 22259.0, 28754.2 [21417.6, 28889.7] | 1170 MHz 24%, 796 MHz 16%, 1278 MHz 16% |
| ullm | 180 | 440.0, 3366.5, 3732.7 [3640.5, 3833.3] | 65.1, 92.3, 123.2 [91.8, 123.9] | 9522.4, 19796.8, 20818.8 [19937.0, 21436.8] | 1170 MHz 35%, 1056 MHz 25%, 1278 MHz 14% |

Streams: a stall is a gap over 100 ms between chunks. The footprint is the session's peak.

| Engine | Largest gap (ms) | Stream time in stalls | Peak footprint (GB) |
|---|---|---|---|
| mlx-lm-batched-no-cache | 1026 | 24.75% | 11.56 |
| mlx-lm-batched-no-cache-twin | 1403 | 25.66% | 11.61 |
| ullm | 3718 | 32.36% | 5.08 |
