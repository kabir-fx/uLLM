# VALID: m2-c2, concurrency 2

- Baseline `mlx-lm-batched-no-cache` and its twin; engines `ullm`, `mlx-lm-batched`. One server per chunk.
- 12 prompts and 2 warm-up requests a chunk; settle 600 s; seed 0.
- Commit `c51200e20812` (dirty tree). Apple M4, 10 cores, 16 GB, macOS 27.2.1; mlx 0.32.4, mlx-lm 0.32.2. At the start: AC True, battery 31%, Low Power Mode False. Over 85 probes: charging in 85, lid open in 85.

## Gates

- **Conditions**: passed. 3544 clock samples, 85 probes on AC.
- **Liveness**: passed. 21 of 21 blocks, ended completed; 0 of 1176 requests failed; 0 prompt lengths differ.
- **Memory**: passed. peak GB of limit: mlx-lm-batched-no-cache 2.8/8, mlx-lm-batched-no-cache-twin 2.6/8, ullm 2.0/6, mlx-lm-batched 3.7/11; swap never grew.
- **Twin**: passed. Sustained 1.0030 (limit 0.03).

## Regimes

Peak 0, Sustained 20, Transition 1 blocks. Transition blocks are in raw.jsonl and in no table.

## Sustained

Ratio to `mlx-lm-batched-no-cache`: median [95% block-bootstrap interval], prompts/blocks. Below 1 is faster for TTFT, TPOT and E2E; above 1 is faster for output tok/s.

| Engine | TTFT | TPOT | E2E | Output tok/s |
|---|---|---|---|---|
| ullm | 0.548 [0.505, 0.586], 240/20 | 1.003 [0.995, 1.027], 240/20, within noise | 0.919 [0.900, 0.938], 240/20 | 1.079 [1.053, 1.099], 20/20 |
| mlx-lm-batched | 1.008 [0.982, 1.031], 240/20, within noise | 1.010 [0.993, 1.025], 240/20, within noise | 1.008 [0.989, 1.024], 240/20, within noise | 0.991 [0.977, 1.008], 20/20, within noise |
| mlx-lm-batched-no-cache-twin | 0.997 [0.984, 1.018], 240/20 | 1.003 [0.990, 1.021], 240/20 | 1.004 [0.992, 1.016], 240/20 | 0.992 [0.972, 1.006], 20/20 |
| floor | ±0.018 | ±0.021 | ±0.016 | ±0.028 |

Absolute, ms: p50, p95, p99 [95% block-bootstrap interval of p99].

| Engine | Requests | TTFT | TPOT | E2E | Clock mix |
|---|---|---|---|---|---|
| mlx-lm-batched-no-cache | 240 | 889.9, 1582.0, 1953.4 [1616.1, 2330.8] | 21.8, 27.5, 31.3 [27.8, 33.7] | 4157.2, 7597.6, 8057.5 [7794.4, 8912.0] | 618 MHz 50%, 338 MHz 21%, 796 MHz 18% |
| mlx-lm-batched-no-cache-twin | 240 | 894.2, 1645.4, 1953.2 [1659.6, 2248.9] | 21.8, 26.6, 28.5 [27.0, 29.8] | 4214.0, 7621.1, 7902.0 [7692.8, 8269.2] | 618 MHz 44%, 796 MHz 22%, 338 MHz 19% |
| ullm | 240 | 465.1, 949.6, 1376.6 [1020.8, 1462.7] | 22.6, 27.0, 29.2 [28.0, 31.2] | 3871.1, 7293.1, 7940.5 [7339.6, 8545.2] | 618 MHz 53%, 338 MHz 21%, 796 MHz 18% |
| mlx-lm-batched | 240 | 931.8, 1571.4, 1820.4 [1577.0, 2012.5] | 22.3, 26.8, 28.6 [27.3, 30.1] | 4483.7, 7692.3, 8018.8 [7723.7, 8433.0] | 618 MHz 53%, 338 MHz 21%, 796 MHz 18% |

Streams: a stall is a gap over 100 ms between chunks. The footprint is the session's peak.

| Engine | Largest gap (ms) | Stream time in stalls | Peak footprint (GB) |
|---|---|---|---|
| mlx-lm-batched-no-cache | 1160 | 10.37% | 2.81 |
| mlx-lm-batched-no-cache-twin | 1084 | 9.84% | 2.60 |
| ullm | 994 | 12.54% | 2.04 |
| mlx-lm-batched | 1107 | 10.32% | 3.68 |
