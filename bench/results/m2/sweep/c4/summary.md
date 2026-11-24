# VALID: m2-c4, concurrency 4

- Baseline `mlx-lm-batched-no-cache` and its twin; engines `ullm`, `mlx-lm-batched`. One server per chunk.
- 12 prompts and 4 warm-up requests a chunk; settle 600 s; seed 0.
- Commit `c51200e20812` (dirty tree). Apple M4, 10 cores, 16 GB, macOS 27.2.1; mlx 0.32.4, mlx-lm 0.32.2. At the start: AC True, battery 44%, Low Power Mode False. Over 85 probes: charging in 85, lid open in 85.

## Gates

- **Conditions**: passed. 3241 clock samples, 85 probes on AC.
- **Liveness**: passed. 21 of 21 blocks, ended completed; 0 of 1344 requests failed; 0 prompt lengths differ.
- **Memory**: passed. peak GB of limit: mlx-lm-batched-no-cache 4.1/8, mlx-lm-batched-no-cache-twin 4.1/8, ullm 2.5/6, mlx-lm-batched 5.8/11; swap never grew.
- **Twin**: passed. Peak 1.0046 (limit 0.005); Sustained 1.0082 (limit 0.03).

## Regimes

Peak 1, Sustained 19, Transition 1 blocks. Transition blocks are in raw.jsonl and in no table.

## Peak

Ratio to `mlx-lm-batched-no-cache`: median [95% block-bootstrap interval], prompts/blocks. Below 1 is faster for TTFT, TPOT and E2E; above 1 is faster for output tok/s.

| Engine | TTFT | TPOT | E2E | Output tok/s |
|---|---|---|---|---|
| ullm | 0.301 no interval, 12/1, within noise | 1.020 no interval, 12/1, within noise | 0.901 no interval, 12/1, within noise | 1.099 no interval, 1/1, within noise |
| mlx-lm-batched | 1.051 no interval, 12/1, within noise | 1.036 no interval, 12/1, within noise | 1.038 no interval, 12/1, within noise | 0.975 no interval, 1/1, within noise |
| mlx-lm-batched-no-cache-twin | 1.007 no interval, 12/1 | 1.005 no interval, 12/1 | 1.008 no interval, 12/1 | 1.027 no interval, 1/1 |
| floor | n/a | n/a | n/a | n/a |

Absolute, ms: p50, p95, p99 [95% block-bootstrap interval of p99].

| Engine | Requests | TTFT | TPOT | E2E | Clock mix |
|---|---|---|---|---|---|
| mlx-lm-batched-no-cache | 12 | 928.8, 1078.8, 1078.8 | 20.6, 22.9, 22.9 | 3270.5, 7161.7, 7161.7 | 1470 MHz 99%, 338 MHz 1%, 1338 MHz 1% |
| mlx-lm-batched-no-cache-twin | 12 | 903.7, 1124.4, 1124.4 | 20.0, 23.5, 23.5 | 3616.2, 7227.2, 7227.2 | 1470 MHz 99%, 338 MHz 1%, 1338 MHz 1% |
| ullm | 12 | 207.9, 707.4, 707.4 | 21.7, 27.3, 27.3 | 2998.9, 7108.1, 7108.1 | 1470 MHz 98%, 338 MHz 1%, 1338 MHz 1% |
| mlx-lm-batched | 12 | 618.8, 1009.7, 1009.7 | 22.0, 26.1, 26.1 | 3641.3, 7507.3, 7507.3 | 1470 MHz 98%, 338 MHz 1%, 1338 MHz 1% |

Streams: a stall is a gap over 100 ms between chunks. The footprint is the session's peak.

| Engine | Largest gap (ms) | Stream time in stalls | Peak footprint (GB) |
|---|---|---|---|
| mlx-lm-batched-no-cache | 500 | 9.50% | 4.14 |
| mlx-lm-batched-no-cache-twin | 503 | 8.60% | 4.14 |
| ullm | 634 | 14.38% | 2.52 |
| mlx-lm-batched | 413 | 9.98% | 5.84 |

## Sustained

Ratio to `mlx-lm-batched-no-cache`: median [95% block-bootstrap interval], prompts/blocks. Below 1 is faster for TTFT, TPOT and E2E; above 1 is faster for output tok/s.

| Engine | TTFT | TPOT | E2E | Output tok/s |
|---|---|---|---|---|
| ullm | 0.465 [0.424, 0.526], 228/19 | 1.044 [1.018, 1.080], 228/19 | 0.912 [0.887, 0.930], 228/19 | 1.093 [1.050, 1.117], 19/19 |
| mlx-lm-batched | 1.007 [0.990, 1.035], 228/19, within noise | 1.013 [0.997, 1.036], 228/19, within noise | 1.004 [0.985, 1.027], 228/19, within noise | 0.989 [0.959, 0.999], 19/19, within noise |
| mlx-lm-batched-no-cache-twin | 1.002 [0.984, 1.025], 228/19 | 1.008 [0.994, 1.017], 228/19 | 1.009 [0.993, 1.020], 228/19 | 0.987 [0.978, 1.004], 19/19 |
| floor | ±0.025 | ±0.017 | ±0.020 | ±0.022 |

Absolute, ms: p50, p95, p99 [95% block-bootstrap interval of p99].

| Engine | Requests | TTFT | TPOT | E2E | Clock mix |
|---|---|---|---|---|---|
| mlx-lm-batched-no-cache | 228 | 1514.0, 2798.1, 2983.2 [2735.6, 2983.8] | 30.5, 39.0, 44.0 [39.6, 52.0] | 6429.1, 11519.9, 12137.6 [11702.4, 13148.7] | 618 MHz 47%, 338 MHz 35%, 796 MHz 13% |
| mlx-lm-batched-no-cache-twin | 228 | 1549.5, 2830.2, 2998.4 [2815.2, 2999.0] | 31.1, 39.4, 41.6 [39.8, 45.2] | 6390.1, 11407.6, 12298.4 [11848.7, 13362.9] | 618 MHz 45%, 338 MHz 38%, 796 MHz 10% |
| ullm | 228 | 586.6, 1906.8, 2286.4 [2053.8, 2314.3] | 33.1, 42.5, 45.6 [43.8, 48.5] | 5628.7, 10742.1, 11558.0 [11071.8, 11771.5] | 618 MHz 51%, 338 MHz 36%, 796 MHz 8% |
| mlx-lm-batched | 228 | 1469.3, 3004.5, 3096.2 [2977.9, 3097.4] | 31.4, 38.9, 44.7 [40.3, 51.5] | 6411.8, 11511.1, 12453.8 [11977.0, 13099.0] | 618 MHz 45%, 338 MHz 36%, 796 MHz 12% |

Streams: a stall is a gap over 100 ms between chunks. The footprint is the session's peak.

| Engine | Largest gap (ms) | Stream time in stalls | Peak footprint (GB) |
|---|---|---|---|
| mlx-lm-batched-no-cache | 2105 | 18.83% | 4.14 |
| mlx-lm-batched-no-cache-twin | 2109 | 18.42% | 4.14 |
| ullm | 1869 | 24.44% | 2.52 |
| mlx-lm-batched | 1877 | 18.85% | 5.84 |
