# VALID: m2-c8, concurrency 8

- Baseline `mlx-lm-batched-no-cache` and its twin; engines `ullm`, `mlx-lm-batched`. One server per chunk.
- 24 prompts and 8 warm-up requests a chunk; settle 600 s; seed 0.
- Commit `c51200e20812` (dirty tree). Apple M4, 10 cores, 16 GB, macOS 27.2.1; mlx 0.32.4, mlx-lm 0.32.2. At the start: AC True, battery 55%, Low Power Mode False. Over 41 probes: charging in 41, lid open in 41.

## Gates

- **Conditions**: passed. 2433 clock samples, 41 probes on AC.
- **Liveness**: passed. 10 of 10 blocks, ended completed; 0 of 1280 requests failed; 0 prompt lengths differ.
- **Memory**: passed. peak GB of limit: mlx-lm-batched-no-cache 6.6/8, mlx-lm-batched-no-cache-twin 6.9/8, ullm 3.2/6, mlx-lm-batched 9.8/11; swap never grew.
- **Twin**: passed. Sustained 1.0043 (limit 0.03).

## Regimes

Peak 0, Sustained 9, Transition 1 blocks. Transition blocks are in raw.jsonl and in no table.

## Sustained

Ratio to `mlx-lm-batched-no-cache`: median [95% block-bootstrap interval], prompts/blocks. Below 1 is faster for TTFT, TPOT and E2E; above 1 is faster for output tok/s.

| Engine | TTFT | TPOT | E2E | Output tok/s |
|---|---|---|---|---|
| ullm | 0.424 [0.375, 0.482], 216/9 | 1.008 [0.964, 1.038], 216/9, within noise | 0.891 [0.844, 0.921], 216/9 | 1.120 [1.086, 1.170], 9/9 |
| mlx-lm-batched | 0.995 [0.933, 1.026], 216/9, within noise | 0.983 [0.915, 1.007], 216/9, within noise | 0.982 [0.946, 1.007], 216/9, within noise | 1.007 [0.974, 1.090], 9/9, within noise |
| mlx-lm-batched-no-cache-twin | 1.013 [0.977, 1.051], 216/9 | 1.004 [0.989, 1.032], 216/9 | 1.010 [0.995, 1.034], 216/9 | 0.990 [0.960, 1.027], 9/9 |
| floor | ±0.051 | ±0.032 | ±0.034 | ±0.040 |

Absolute, ms: p50, p95, p99 [95% block-bootstrap interval of p99].

| Engine | Requests | TTFT | TPOT | E2E | Clock mix |
|---|---|---|---|---|---|
| mlx-lm-batched-no-cache | 216 | 1595.6, 4816.8, 4977.6 [4573.0, 4980.2] | 47.6, 69.8, 76.4 [72.1, 80.3] | 9339.0, 18137.4, 21983.8 [18650.5, 24146.9] | 618 MHz 46%, 338 MHz 29%, 796 MHz 13% |
| mlx-lm-batched-no-cache-twin | 216 | 1703.5, 5001.3, 5095.7 [4881.4, 5098.4] | 48.8, 68.6, 78.1 [70.0, 86.5] | 9430.4, 19051.3, 22250.7 [19445.2, 23978.4] | 618 MHz 45%, 338 MHz 30%, 796 MHz 14% |
| ullm | 216 | 597.3, 2900.9, 3379.9 [3119.8, 3538.2] | 50.7, 70.8, 76.6 [68.4, 84.9] | 8164.0, 16733.0, 20976.9 [17525.6, 21788.8] | 618 MHz 47%, 338 MHz 28%, 796 MHz 11% |
| mlx-lm-batched | 216 | 1426.4, 4851.1, 5068.6 [4550.5, 5070.8] | 47.1, 68.0, 71.5 [67.1, 73.4] | 9182.6, 18309.4, 21693.6 [18985.4, 24176.5] | 618 MHz 44%, 338 MHz 26%, 796 MHz 12% |

Streams: a stall is a gap over 100 ms between chunks. The footprint is the session's peak.

| Engine | Largest gap (ms) | Stream time in stalls | Peak footprint (GB) |
|---|---|---|---|
| mlx-lm-batched-no-cache | 1947 | 28.87% | 6.64 |
| mlx-lm-batched-no-cache-twin | 2168 | 28.94% | 6.93 |
| ullm | 3156 | 34.25% | 3.17 |
| mlx-lm-batched | 2007 | 28.76% | 9.82 |
