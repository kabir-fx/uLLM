# VALID: m2-c1, concurrency 1

- Baseline `mlx-lm-batched-no-cache` and its twin; engines `ullm`, `mlx-lm-batched`, `mlx-lm-sequential-no-cache`. One server per chunk.
- 12 prompts and 2 warm-up requests a chunk; settle 240 s; seed 0.
- Commit `c51200e20812` (dirty tree). Apple M4, 10 cores, 16 GB, macOS 27.2.1; mlx 0.32.4, mlx-lm 0.32.2. At the start: AC True, battery 18%, Low Power Mode False. Over 106 probes: charging in 106, lid open in 106.

## Gates

- **Conditions**: passed. 5045 clock samples, 106 probes on AC.
- **Liveness**: passed. 21 of 21 blocks, ended completed; 0 of 1470 requests failed; 0 prompt lengths differ.
- **Memory**: passed. peak GB of limit: mlx-lm-batched-no-cache 2.0/4, mlx-lm-batched-no-cache-twin 2.0/4, ullm 2.0/3, mlx-lm-batched 2.8/4, mlx-lm-sequential-no-cache 2.0/3; swap never grew.
- **Twin**: passed. Peak 1.0009 (limit 0.005); Sustained 0.9992 (limit 0.03).

## Regimes

Peak 3, Sustained 17, Transition 1 blocks. Transition blocks are in raw.jsonl and in no table.

## Peak

Ratio to `mlx-lm-batched-no-cache`: median [95% block-bootstrap interval], prompts/blocks. Below 1 is faster for TTFT, TPOT and E2E; above 1 is faster for output tok/s.

| Engine | TTFT | TPOT | E2E | Output tok/s |
|---|---|---|---|---|
| ullm | 0.794 [0.779, 0.843], 36/3 | 1.002 [1.001, 1.115], 36/3, within noise | 0.986 [0.975, 1.098], 36/3 | 1.035 [0.930, 1.043], 3/3 |
| mlx-lm-batched | 1.016 [1.015, 1.023], 36/3 | 1.002 [0.999, 1.006], 36/3, within noise | 1.003 [1.002, 1.007], 36/3 | 1.002 [0.995, 1.020], 3/3, within noise |
| mlx-lm-sequential-no-cache | 0.865 [0.863, 0.874], 36/3 | 0.993 [0.992, 0.996], 36/3 | 0.981 [0.979, 0.985], 36/3 | 1.035 [1.032, 1.044], 3/3 |
| mlx-lm-batched-no-cache-twin | 0.999 [0.998, 1.001], 36/3 | 1.001 [1.000, 1.002], 36/3 | 1.000 [1.000, 1.003], 36/3 | 1.014 [0.999, 1.026], 3/3 |
| floor | ±0.002 | ±0.002 | ±0.003 | ±0.026 |

Absolute, ms: p50, p95, p99 [95% block-bootstrap interval of p99].

| Engine | Requests | TTFT | TPOT | E2E | Clock mix |
|---|---|---|---|---|---|
| mlx-lm-batched-no-cache | 36 | 220.2, 296.7, 299.3 [288.7, 299.3] | 14.3, 15.6, 15.8 [15.4, 15.8] | 1809.7, 4457.3, 4460.3 [4042.2, 4460.3] | 1470 MHz 99%, 1338 MHz 1%, 338 MHz 0% |
| mlx-lm-batched-no-cache-twin | 36 | 215.1, 296.4, 321.9 [290.7, 321.9] | 14.2, 15.7, 16.0 [14.4, 16.0] | 1813.9, 4460.9, 4468.4 [4056.9, 4468.4] | 1470 MHz 99%, 1338 MHz 0%, 338 MHz 0% |
| ullm | 36 | 172.0, 246.4, 247.7 [239.4, 247.7] | 14.3, 16.2, 17.0 [14.5, 17.0] | 1896.2, 4616.8, 4970.5 [4008.2, 4970.5] | 1470 MHz 94%, 1338 MHz 2%, 1170 MHz 1% |
| mlx-lm-batched | 36 | 210.1, 303.6, 323.1 [296.1, 323.1] | 14.2, 15.8, 16.0 [14.5, 16.0] | 1844.1, 4490.4, 4501.9 [4464.3, 4501.9] | 1470 MHz 98%, 1338 MHz 1%, 1056 MHz 0% |
| mlx-lm-sequential-no-cache | 36 | 188.7, 256.6, 257.9 [252.6, 257.9] | 14.1, 15.3, 15.4 [14.3, 15.4] | 1853.0, 4398.8, 4412.4 [3985.9, 4412.4] | 1470 MHz 99%, 1338 MHz 1%, 1278 MHz 0% |

Streams: a stall is a gap over 100 ms between chunks. The footprint is the session's peak.

| Engine | Largest gap (ms) | Stream time in stalls | Peak footprint (GB) |
|---|---|---|---|
| mlx-lm-batched-no-cache | 46 | 0.00% | 2.03 |
| mlx-lm-batched-no-cache-twin | 42 | 0.00% | 2.03 |
| ullm | 28 | 0.00% | 1.96 |
| mlx-lm-batched | 42 | 0.00% | 2.77 |
| mlx-lm-sequential-no-cache | 45 | 0.00% | 1.96 |

## Sustained

Ratio to `mlx-lm-batched-no-cache`: median [95% block-bootstrap interval], prompts/blocks. Below 1 is faster for TTFT, TPOT and E2E; above 1 is faster for output tok/s.

| Engine | TTFT | TPOT | E2E | Output tok/s |
|---|---|---|---|---|
| ullm | 0.841 [0.793, 0.863], 204/17 | 1.014 [1.008, 1.024], 204/17 | 0.989 [0.981, 0.999], 204/17 | 1.006 [0.994, 1.020], 17/17 |
| mlx-lm-batched | 1.006 [0.954, 1.024], 204/17, within noise | 1.004 [0.998, 1.014], 204/17 | 1.003 [0.994, 1.013], 204/17, within noise | 0.993 [0.974, 0.996], 17/17 |
| mlx-lm-sequential-no-cache | 0.910 [0.894, 0.928], 204/17 | 0.995 [0.991, 1.001], 204/17 | 0.981 [0.973, 0.989], 204/17 | 1.012 [1.008, 1.035], 17/17 |
| mlx-lm-batched-no-cache-twin | 1.002 [0.982, 1.019], 204/17 | 0.999 [0.996, 1.003], 204/17 | 0.999 [0.996, 1.003], 204/17 | 1.001 [0.997, 1.005], 17/17 |
| floor | ±0.019 | ±0.004 | ±0.004 | ±0.005 |

Absolute, ms: p50, p95, p99 [95% block-bootstrap interval of p99].

| Engine | Requests | TTFT | TPOT | E2E | Clock mix |
|---|---|---|---|---|---|
| mlx-lm-batched-no-cache | 204 | 425.9, 819.0, 900.5 [851.8, 1047.4] | 16.6, 18.4, 19.0 [18.6, 19.3] | 3071.7, 5406.3, 5625.4 [5504.0, 6012.7] | 618 MHz 35%, 796 MHz 33%, 928 MHz 12% |
| mlx-lm-batched-no-cache-twin | 204 | 425.3, 787.2, 930.5 [812.2, 1034.6] | 16.6, 18.3, 18.7 [18.3, 20.1] | 3031.3, 5360.4, 5709.0 [5500.7, 6297.9] | 796 MHz 34%, 618 MHz 32%, 928 MHz 18% |
| ullm | 204 | 348.5, 715.5, 821.8 [731.8, 920.9] | 17.0, 18.7, 19.9 [18.7, 20.9] | 3035.8, 5426.5, 5721.9 [5516.1, 6216.0] | 618 MHz 32%, 796 MHz 30%, 928 MHz 14% |
| mlx-lm-batched | 204 | 405.2, 799.8, 871.7 [810.5, 1037.6] | 16.9, 18.2, 18.9 [18.2, 22.4] | 3148.1, 5385.5, 5714.4 [5429.3, 5928.1] | 796 MHz 31%, 618 MHz 30%, 928 MHz 16% |
| mlx-lm-sequential-no-cache | 204 | 397.0, 758.7, 845.3 [789.5, 971.5] | 16.5, 18.3, 18.5 [18.0, 19.2] | 3005.6, 5233.5, 5881.2 [5390.7, 6190.8] | 618 MHz 37%, 796 MHz 33%, 928 MHz 11% |

Streams: a stall is a gap over 100 ms between chunks. The footprint is the session's peak.

| Engine | Largest gap (ms) | Stream time in stalls | Peak footprint (GB) |
|---|---|---|---|
| mlx-lm-batched-no-cache | 85 | 0.00% | 2.03 |
| mlx-lm-batched-no-cache-twin | 67 | 0.00% | 2.03 |
| ullm | 45 | 0.00% | 1.96 |
| mlx-lm-batched | 54 | 0.00% | 2.77 |
| mlx-lm-sequential-no-cache | 69 | 0.00% | 1.96 |

## Same completion length as `mlx-lm-batched-no-cache`, prompt by prompt

- mlx-lm-batched-no-cache-twin: 252 of 252 prompts
- ullm: 202 of 252 prompts
- mlx-lm-batched: 241 of 252 prompts
- mlx-lm-sequential-no-cache: 202 of 252 prompts
