# DISCARDED: m2-c16, concurrency 16

- Baseline `mlx-lm-batched-no-cache` and its twin; engines `ullm`. One server per chunk.
- 30 prompts and 16 warm-up requests a chunk; settle 600 s; seed 0.
- Commit `c51200e20812` (dirty tree). Apple M4, 10 cores, 16 GB, macOS 27.2.1; mlx 0.32.4, mlx-lm 0.32.2. At the start: AC True, battery 64%, Low Power Mode False. Over 25 probes: charging in 25, lid open in 25.

## Gates

- **Conditions**: passed. 2167 clock samples, 25 probes on AC.
- **Liveness**: passed. 8 of 8 blocks, ended completed; 0 of 1104 requests failed; 0 prompt lengths differ.
- **Memory**: FAILED. swap grew 326 MB above the first probe's.
- **Twin**: FAILED. Sustained twin TPOT median 1.0556, off by 5.56%.

## Regimes

Peak 0, Sustained 7, Transition 1 blocks. Transition blocks are in raw.jsonl and in no table.

## Sustained

Ratio to `mlx-lm-batched-no-cache`: median [95% block-bootstrap interval], prompts/blocks. Below 1 is faster for TTFT, TPOT and E2E; above 1 is faster for output tok/s.

| Engine | TTFT | TPOT | E2E | Output tok/s |
|---|---|---|---|---|
| ullm | 0.444 [0.387, 0.496], 210/7 | 1.110 [1.043, 1.251], 210/7, within noise | 0.946 [0.869, 1.032], 210/7, within noise | 1.023 [0.967, 1.142], 7/7, within noise |
| mlx-lm-batched-no-cache-twin | 1.013 [0.975, 1.050], 210/7 | 1.056 [1.010, 1.165], 210/7 | 1.040 [1.000, 1.110], 210/7 | 0.935 [0.891, 1.004], 7/7 |
| floor | ±0.050 | ±0.165 | ±0.110 | ±0.109 |

Absolute, ms: p50, p95, p99 [95% block-bootstrap interval of p99].

| Engine | Requests | TTFT | TPOT | E2E | Clock mix |
|---|---|---|---|---|---|
| mlx-lm-batched-no-cache | 210 | 6060.1, 9063.1, 9368.1 [8378.7, 9369.8] | 90.2, 117.5, 121.4 [117.5, 128.5] | 16335.4, 36099.6, 37871.3 [35950.0, 37948.4] | 618 MHz 60%, 796 MHz 19%, 338 MHz 12% |
| mlx-lm-batched-no-cache-twin | 210 | 6280.6, 8815.7, 8898.7 [8378.9, 8901.0] | 99.4, 129.1, 141.4 [133.1, 144.4] | 17471.3, 36850.7, 37812.4 [37309.3, 40384.2] | 618 MHz 65%, 338 MHz 17%, 796 MHz 14% |
| ullm | 210 | 1029.7, 5563.9, 6285.8 [5664.3, 6439.2] | 107.0, 138.4, 153.3 [141.8, 160.0] | 15333.2, 32758.1, 34968.6 [32848.2, 35405.8] | 618 MHz 65%, 796 MHz 15%, 338 MHz 14% |

Streams: a stall is a gap over 100 ms between chunks. The footprint is the session's peak.

| Engine | Largest gap (ms) | Stream time in stalls | Peak footprint (GB) |
|---|---|---|---|
| mlx-lm-batched-no-cache | 2428 | 36.52% | 11.56 |
| mlx-lm-batched-no-cache-twin | 3218 | 39.88% | 11.57 |
| ullm | 6111 | 37.57% | 5.08 |
