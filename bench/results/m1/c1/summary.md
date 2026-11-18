# VALID: m1-c1, concurrency 1

- Baseline `mlx-lm-sequential-no-cache` and its twin; engines `ullm`, `mlx-lm-sequential`. Servers stay loaded.
- 12 prompts and 2 warm-up requests a chunk; settle 240 s; seed 0.
- Commit `41671f1d9d8c` (dirty tree). Apple M4, 10 cores, 16 GB, macOS 27.2.1; mlx 0.32.4, mlx-lm 0.32.2. At the start: AC True, battery 80%, Low Power Mode False. Over 85 probes: charging in 0, lid open in 85.

## Gates

- **Conditions**: passed. 3697 clock samples, 85 probes on AC.
- **Liveness**: passed. 21 of 21 blocks, ended completed; 0 of 1184 requests failed; 0 prompt lengths differ.
- **Memory**: passed. peak GB of limit: mlx-lm-sequential-no-cache 2.0/3, mlx-lm-sequential-no-cache-twin 2.0/3, ullm 2.0/3, mlx-lm-sequential 2.9/4; swap never grew.
- **Twin**: passed. Peak 0.9986 (limit 0.005); Sustained 0.9960 (limit 0.03).

## Regimes

Peak 2, Sustained 19, Transition 0 blocks. Transition blocks are in raw.jsonl and in no table.

## Peak

Ratio to `mlx-lm-sequential-no-cache`: median [95% block-bootstrap interval], prompts/blocks. Below 1 is faster for TTFT, TPOT and E2E; above 1 is faster for output tok/s.

| Engine | TTFT | TPOT | E2E | Output tok/s |
|---|---|---|---|---|
| ullm | 0.941 no interval, 24/2, within noise | 1.177 no interval, 24/2, within noise | 1.149 no interval, 24/2, within noise | 0.858 no interval, 2/2, within noise |
| mlx-lm-sequential | 1.019 no interval, 24/2, within noise | 1.007 no interval, 24/2, within noise | 1.009 no interval, 24/2, within noise | 0.989 no interval, 2/2, within noise |
| mlx-lm-sequential-no-cache-twin | 1.001 no interval, 24/2 | 0.999 no interval, 24/2 | 0.999 no interval, 24/2 | 1.000 no interval, 2/2 |
| floor | n/a | n/a | n/a | n/a |

Absolute, ms: p50, p95, p99 [95% block-bootstrap interval of p99].

| Engine | Requests | TTFT | TPOT | E2E | Clock mix |
|---|---|---|---|---|---|
| mlx-lm-sequential-no-cache | 24 | 188.3, 253.0, 256.9 | 14.0, 14.4, 14.4 | 1836.9, 4383.1, 4420.8 | 1470 MHz 99%, 1338 MHz 0%, 338 MHz 0% |
| mlx-lm-sequential-no-cache-twin | 24 | 188.9, 253.3, 258.0 | 14.1, 14.3, 14.3 | 1839.2, 4406.0, 4415.5 | 1470 MHz 100%, 1338 MHz 0%, 338 MHz 0% |
| ullm | 24 | 177.9, 242.7, 245.6 | 16.5, 17.4, 17.5 | 2110.1, 5200.0, 5329.3 | 1470 MHz 99%, 1338 MHz 0%, 338 MHz 0% |
| mlx-lm-sequential | 24 | 176.5, 257.9, 260.9 | 14.2, 14.4, 15.6 | 1864.3, 4434.5, 4439.6 | 1470 MHz 99%, 1338 MHz 1%, 1278 MHz 0% |

Streams: a stall is a gap over 100 ms between chunks. The footprint is the session's peak.

| Engine | Largest gap (ms) | Stream time in stalls | Peak footprint (GB) |
|---|---|---|---|
| mlx-lm-sequential-no-cache | 29 | 0.00% | 1.99 |
| mlx-lm-sequential-no-cache-twin | 29 | 0.00% | 1.98 |
| ullm | 20 | 0.00% | 2.02 |
| mlx-lm-sequential | 42 | 0.00% | 2.94 |

## Sustained

Ratio to `mlx-lm-sequential-no-cache`: median [95% block-bootstrap interval], prompts/blocks. Below 1 is faster for TTFT, TPOT and E2E; above 1 is faster for output tok/s.

| Engine | TTFT | TPOT | E2E | Output tok/s |
|---|---|---|---|---|
| ullm | 0.866 [0.847, 0.883], 228/19 | 1.167 [1.160, 1.172], 228/19 | 1.131 [1.125, 1.136], 228/19 | 0.877 [0.873, 0.884], 19/19 |
| mlx-lm-sequential | 1.022 [0.994, 1.042], 228/19, within noise | 1.003 [0.999, 1.006], 228/19, within noise | 1.004 [0.998, 1.010], 228/19, within noise | 0.995 [0.991, 1.005], 19/19, within noise |
| mlx-lm-sequential-no-cache-twin | 0.980 [0.965, 1.009], 228/19 | 0.996 [0.993, 0.999], 228/19 | 0.995 [0.990, 1.001], 228/19 | 1.009 [0.996, 1.013], 19/19 |
| floor | ±0.035 | ±0.007 | ±0.010 | ±0.013 |

Absolute, ms: p50, p95, p99 [95% block-bootstrap interval of p99].

| Engine | Requests | TTFT | TPOT | E2E | Clock mix |
|---|---|---|---|---|---|
| mlx-lm-sequential-no-cache | 228 | 252.8, 418.8, 485.3 [438.1, 546.3] | 15.3, 15.9, 16.9 [16.0, 18.6] | 2703.4, 4667.6, 4842.9 [4765.9, 4941.2] | 1056 MHz 33%, 928 MHz 31%, 1170 MHz 18% |
| mlx-lm-sequential-no-cache-twin | 228 | 250.8, 417.5, 494.6 [444.6, 525.2] | 15.3, 15.9, 17.9 [15.9, 24.2] | 2662.3, 4662.5, 4875.9 [4776.7, 5706.5] | 1056 MHz 36%, 928 MHz 28%, 1170 MHz 18% |
| ullm | 228 | 220.8, 370.5, 415.8 [396.4, 490.5] | 17.9, 19.1, 19.4 [19.1, 19.6] | 3028.0, 5384.6, 5809.0 [5629.6, 5847.6] | 1170 MHz 36%, 1056 MHz 29%, 1278 MHz 13% |
| mlx-lm-sequential | 228 | 258.1, 436.7, 547.6 [456.2, 942.1] | 15.4, 16.0, 17.1 [16.0, 18.4] | 2688.0, 4707.5, 4879.6 [4779.5, 5013.5] | 1056 MHz 33%, 928 MHz 29%, 1170 MHz 17% |

Streams: a stall is a gap over 100 ms between chunks. The footprint is the session's peak.

| Engine | Largest gap (ms) | Stream time in stalls | Peak footprint (GB) |
|---|---|---|---|
| mlx-lm-sequential-no-cache | 55 | 0.00% | 1.99 |
| mlx-lm-sequential-no-cache-twin | 102 | 0.02% | 1.98 |
| ullm | 40 | 0.00% | 2.02 |
| mlx-lm-sequential | 48 | 0.00% | 2.94 |

## Same completion length as `mlx-lm-sequential-no-cache`, prompt by prompt

- mlx-lm-sequential-no-cache-twin: 252 of 252 prompts
- ullm: 252 of 252 prompts
- mlx-lm-sequential: 243 of 252 prompts
