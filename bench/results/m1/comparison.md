### Peak: m1-c1, concurrency 1, servers stay loaded, VALID

Ratio to `mlx-lm-sequential-no-cache`: median [95% block-bootstrap interval], prompts/blocks. Below 1 is faster for TTFT, TPOT and E2E; above 1 is faster for output tok/s.

| Engine | TTFT | TPOT | E2E | Output tok/s |
|---|---|---|---|---|
| ullm | 0.941 no interval, 24/2, within noise | 1.177 no interval, 24/2, within noise | 1.149 no interval, 24/2, within noise | 0.858 no interval, 2/2, within noise |
| mlx-lm-sequential | 1.019 no interval, 24/2, within noise | 1.007 no interval, 24/2, within noise | 1.009 no interval, 24/2, within noise | 0.989 no interval, 2/2, within noise |
| mlx-lm-sequential-no-cache-twin | 1.001 no interval, 24/2 | 0.999 no interval, 24/2 | 0.999 no interval, 24/2 | 1.000 no interval, 2/2 |
| floor | n/a | n/a | n/a | n/a |

### Peak: m1-c8, concurrency 8, one server per chunk, VALID

Ratio to `mlx-lm-batched-no-cache`: median [95% block-bootstrap interval], prompts/blocks. Below 1 is faster for TTFT, TPOT and E2E; above 1 is faster for output tok/s.

| Engine | TTFT | TPOT | E2E | Output tok/s |
|---|---|---|---|---|
| ullm | 18.258 no interval, 24/1, within noise | 0.598 no interval, 24/1, within noise | 3.884 no interval, 24/1, within noise | 0.267 no interval, 1/1, within noise |
| mlx-lm-batched | 1.025 no interval, 24/1, within noise | 1.002 no interval, 24/1, within noise | 1.004 no interval, 24/1, within noise | 0.991 no interval, 1/1, within noise |
| mlx-lm-sequential-no-cache | 16.123 no interval, 24/1, within noise | 0.514 no interval, 24/1, within noise | 3.346 no interval, 24/1, within noise | 0.305 no interval, 1/1, within noise |
| mlx-lm-batched-no-cache-twin | 1.010 no interval, 24/1 | 1.001 no interval, 24/1 | 1.001 no interval, 24/1 | 0.999 no interval, 1/1 |
| floor | n/a | n/a | n/a | n/a |

### Sustained: m1-c1, concurrency 1, servers stay loaded, VALID

Ratio to `mlx-lm-sequential-no-cache`: median [95% block-bootstrap interval], prompts/blocks. Below 1 is faster for TTFT, TPOT and E2E; above 1 is faster for output tok/s.

| Engine | TTFT | TPOT | E2E | Output tok/s |
|---|---|---|---|---|
| ullm | 0.866 [0.847, 0.883], 228/19 | 1.167 [1.160, 1.172], 228/19 | 1.131 [1.125, 1.136], 228/19 | 0.877 [0.873, 0.884], 19/19 |
| mlx-lm-sequential | 1.022 [0.994, 1.042], 228/19, within noise | 1.003 [0.999, 1.006], 228/19, within noise | 1.004 [0.998, 1.010], 228/19, within noise | 0.995 [0.991, 1.005], 19/19, within noise |
| mlx-lm-sequential-no-cache-twin | 0.980 [0.965, 1.009], 228/19 | 0.996 [0.993, 0.999], 228/19 | 0.995 [0.990, 1.001], 228/19 | 1.009 [0.996, 1.013], 19/19 |
| floor | ±0.035 | ±0.007 | ±0.010 | ±0.013 |

### Sustained: m1-c8, concurrency 8, one server per chunk, VALID

Ratio to `mlx-lm-batched-no-cache`: median [95% block-bootstrap interval], prompts/blocks. Below 1 is faster for TTFT, TPOT and E2E; above 1 is faster for output tok/s.

| Engine | TTFT | TPOT | E2E | Output tok/s |
|---|---|---|---|---|
| ullm | 19.234 [17.509, 22.562], 216/9 | 0.533 [0.517, 0.551], 216/9 | 3.328 [3.259, 3.536], 216/9 | 0.322 [0.299, 0.335], 9/9 |
| mlx-lm-batched | 0.969 [0.958, 1.024], 216/9, within noise | 0.992 [0.975, 1.011], 216/9, within noise | 0.983 [0.970, 1.007], 216/9, within noise | 1.007 [0.998, 1.034], 9/9, within noise |
| mlx-lm-sequential-no-cache | 17.163 [15.260, 20.035], 216/9 | 0.456 [0.442, 0.470], 216/9 | 3.013 [2.830, 3.120], 216/9 | 0.362 [0.336, 0.380], 9/9 |
| mlx-lm-batched-no-cache-twin | 0.988 [0.954, 1.025], 216/9 | 0.989 [0.975, 0.998], 216/9 | 0.986 [0.967, 0.994], 216/9 | 1.009 [1.000, 1.043], 9/9 |
| floor | ±0.046 | ±0.025 | ±0.033 | ±0.043 |
