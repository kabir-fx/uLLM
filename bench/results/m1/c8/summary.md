# VALID: m1-c8, concurrency 8

- Baseline `mlx-lm-batched-no-cache` and its twin; engines `ullm`, `mlx-lm-batched`, `mlx-lm-sequential-no-cache`. One server per chunk.
- 24 prompts and 8 warm-up requests a chunk; settle 600 s; seed 0.
- Commit `41671f1d9d8c` (dirty tree). Apple M4, 10 cores, 16 GB, macOS 27.2.1; mlx 0.32.4, mlx-lm 0.32.2. At the start: AC True, battery 80%, Low Power Mode False. Over 51 probes: charging in 0, lid open in 51.

## Gates

- **Conditions**: passed. 3367 clock samples, 51 probes on AC.
- **Liveness**: passed. 10 of 10 blocks, ended completed; 0 of 1600 requests failed; 0 prompt lengths differ.
- **Memory**: passed. peak GB of limit: mlx-lm-batched-no-cache 6.8/11, mlx-lm-batched-no-cache-twin 6.8/11, ullm 2.0/3, mlx-lm-batched 9.7/14, mlx-lm-sequential-no-cache 1.9/3; swap never grew.
- **Twin**: passed. Peak 1.0005 (limit 0.005); Sustained 0.9890 (limit 0.03).

## Regimes

Peak 1, Sustained 9, Transition 0 blocks. Transition blocks are in raw.jsonl and in no table.

## Peak

Ratio to `mlx-lm-batched-no-cache`: median [95% block-bootstrap interval], prompts/blocks. Below 1 is faster for TTFT, TPOT and E2E; above 1 is faster for output tok/s.

| Engine | TTFT | TPOT | E2E | Output tok/s |
|---|---|---|---|---|
| ullm | 18.258 no interval, 24/1, within noise | 0.598 no interval, 24/1, within noise | 3.884 no interval, 24/1, within noise | 0.267 no interval, 1/1, within noise |
| mlx-lm-batched | 1.025 no interval, 24/1, within noise | 1.002 no interval, 24/1, within noise | 1.004 no interval, 24/1, within noise | 0.991 no interval, 1/1, within noise |
| mlx-lm-sequential-no-cache | 16.123 no interval, 24/1, within noise | 0.514 no interval, 24/1, within noise | 3.346 no interval, 24/1, within noise | 0.305 no interval, 1/1, within noise |
| mlx-lm-batched-no-cache-twin | 1.010 no interval, 24/1 | 1.001 no interval, 24/1 | 1.001 no interval, 24/1 | 0.999 no interval, 1/1 |
| floor | n/a | n/a | n/a | n/a |

Absolute, ms: p50, p95, p99 [95% block-bootstrap interval of p99].

| Engine | Requests | TTFT | TPOT | E2E | Clock mix |
|---|---|---|---|---|---|
| mlx-lm-batched-no-cache | 24 | 924.2, 1908.4, 1908.4 | 26.7, 32.8, 36.8 | 4939.3, 8255.2, 9419.8 | 1470 MHz 99%, 1338 MHz 0%, 338 MHz 0% |
| mlx-lm-batched-no-cache-twin | 24 | 962.2, 1926.8, 1927.3 | 26.6, 32.8, 36.9 | 4935.9, 8259.5, 9428.7 | 1470 MHz 99%, 1338 MHz 0%, 338 MHz 0% |
| ullm | 24 | 17169.8, 23033.5, 23567.9 | 16.2, 17.0, 17.1 | 19580.9, 25754.4, 25911.3 | 1470 MHz 99%, 1338 MHz 0%, 338 MHz 0% |
| mlx-lm-batched | 24 | 835.6, 1999.7, 2000.6 | 27.2, 32.4, 35.4 | 4916.2, 8650.5, 9699.5 | 1470 MHz 97%, 1338 MHz 2%, 338 MHz 0% |
| mlx-lm-sequential-no-cache | 24 | 15253.3, 19335.2, 20732.6 | 13.9, 14.3, 14.3 | 17383.1, 22436.6, 22482.1 | 1470 MHz 99%, 1338 MHz 0%, 338 MHz 0% |

Streams: a stall is a gap over 100 ms between chunks. The footprint is the session's peak.

| Engine | Largest gap (ms) | Stream time in stalls | Peak footprint (GB) |
|---|---|---|---|
| mlx-lm-batched-no-cache | 860 | 17.45% | 6.78 |
| mlx-lm-batched-no-cache-twin | 865 | 17.63% | 6.77 |
| ullm | 19 | 0.00% | 1.97 |
| mlx-lm-batched | 499 | 18.26% | 9.68 |
| mlx-lm-sequential-no-cache | 30 | 0.00% | 1.93 |

## Sustained

Ratio to `mlx-lm-batched-no-cache`: median [95% block-bootstrap interval], prompts/blocks. Below 1 is faster for TTFT, TPOT and E2E; above 1 is faster for output tok/s.

| Engine | TTFT | TPOT | E2E | Output tok/s |
|---|---|---|---|---|
| ullm | 19.234 [17.509, 22.562], 216/9 | 0.533 [0.517, 0.551], 216/9 | 3.328 [3.259, 3.536], 216/9 | 0.322 [0.299, 0.335], 9/9 |
| mlx-lm-batched | 0.969 [0.958, 1.024], 216/9, within noise | 0.992 [0.975, 1.011], 216/9, within noise | 0.983 [0.970, 1.007], 216/9, within noise | 1.007 [0.998, 1.034], 9/9, within noise |
| mlx-lm-sequential-no-cache | 17.163 [15.260, 20.035], 216/9 | 0.456 [0.442, 0.470], 216/9 | 3.013 [2.830, 3.120], 216/9 | 0.362 [0.336, 0.380], 9/9 |
| mlx-lm-batched-no-cache-twin | 0.988 [0.954, 1.025], 216/9 | 0.989 [0.975, 0.998], 216/9 | 0.986 [0.967, 0.994], 216/9 | 1.009 [1.000, 1.043], 9/9 |
| floor | ±0.046 | ±0.025 | ±0.033 | ±0.043 |

Absolute, ms: p50, p95, p99 [95% block-bootstrap interval of p99].

| Engine | Requests | TTFT | TPOT | E2E | Clock mix |
|---|---|---|---|---|---|
| mlx-lm-batched-no-cache | 216 | 999.4, 3082.8, 3333.8 [2975.3, 3334.6] | 31.8, 44.9, 50.3 [45.7, 56.5] | 6289.7, 11362.4, 13697.7 [11726.0, 14792.7] | 1170 MHz 24%, 1056 MHz 20%, 928 MHz 15% |
| mlx-lm-batched-no-cache-twin | 216 | 1057.9, 3255.4, 3588.7 [2834.1, 3590.6] | 31.5, 42.6, 49.5 [45.0, 53.2] | 6317.9, 11028.3, 13523.7 [11648.5, 14359.8] | 1170 MHz 23%, 1056 MHz 19%, 1278 MHz 17% |
| ullm | 216 | 18248.1, 26281.6, 27814.6 [25920.0, 28256.3] | 17.0, 18.2, 18.5 [18.3, 18.8] | 21535.6, 29697.7, 31481.0 [29302.0, 32799.6] | 1278 MHz 35%, 1170 MHz 32%, 1338 MHz 16% |
| mlx-lm-batched | 216 | 1098.4, 2996.6, 3207.2 [2833.4, 3207.7] | 31.9, 43.6, 47.3 [44.3, 50.5] | 6296.0, 11085.8, 13326.8 [11455.8, 14639.8] | 1170 MHz 26%, 1056 MHz 21%, 928 MHz 15% |
| mlx-lm-sequential-no-cache | 216 | 16475.6, 23104.3, 24017.2 [22219.1, 24863.2] | 14.7, 15.3, 15.7 [15.3, 16.0] | 18796.4, 25536.6, 27581.4 [25536.6, 28687.2] | 1170 MHz 41%, 1056 MHz 35%, 928 MHz 12% |

Streams: a stall is a gap over 100 ms between chunks. The footprint is the session's peak.

| Engine | Largest gap (ms) | Stream time in stalls | Peak footprint (GB) |
|---|---|---|---|
| mlx-lm-batched-no-cache | 1231 | 22.62% | 6.78 |
| mlx-lm-batched-no-cache-twin | 1017 | 22.36% | 6.77 |
| ullm | 38 | 0.00% | 1.97 |
| mlx-lm-batched | 1267 | 22.02% | 9.68 |
| mlx-lm-sequential-no-cache | 52 | 0.00% | 1.93 |
