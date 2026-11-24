Ratio to `mlx-lm-batched-no-cache`: median [95% block-bootstrap interval], prompts/blocks. Below 1 is faster for TTFT, TPOT and E2E; above 1 is faster for output tok/s. A gap inside its session's twin floor reads within noise.

- c1, VALID, one server per chunk
  - Commit `c51200e20812` (dirty tree). Apple M4, 10 cores, 16 GB, macOS 27.2.1; mlx 0.32.4, mlx-lm 0.32.2. At the start: AC True, battery 18%, Low Power Mode False. Over 106 probes: charging in 106, lid open in 106.
- c2, VALID, one server per chunk
  - Commit `c51200e20812` (dirty tree). Apple M4, 10 cores, 16 GB, macOS 27.2.1; mlx 0.32.4, mlx-lm 0.32.2. At the start: AC True, battery 31%, Low Power Mode False. Over 85 probes: charging in 85, lid open in 85.
- c4, VALID, one server per chunk
  - Commit `c51200e20812` (dirty tree). Apple M4, 10 cores, 16 GB, macOS 27.2.1; mlx 0.32.4, mlx-lm 0.32.2. At the start: AC True, battery 44%, Low Power Mode False. Over 85 probes: charging in 85, lid open in 85.
- c8, VALID, one server per chunk
  - Commit `c51200e20812` (dirty tree). Apple M4, 10 cores, 16 GB, macOS 27.2.1; mlx 0.32.4, mlx-lm 0.32.2. At the start: AC True, battery 55%, Low Power Mode False. Over 41 probes: charging in 41, lid open in 41.
- c16, DISCARDED, one server per chunk
  - Commit `c51200e20812` (dirty tree). Apple M4, 10 cores, 16 GB, macOS 27.2.1; mlx 0.32.4, mlx-lm 0.32.2. At the start: AC True, battery 50%, Low Power Mode False. Over 25 probes: charging in 25, lid open in 25.

### Peak: TTFT, ratio to `mlx-lm-batched-no-cache`

| Engine | c1 | c4 | c16, DISCARDED |
|---|---|---|---|
| ullm | 0.794 [0.779, 0.843], 36/3 | 0.301 no interval, 12/1, within noise | 0.345 no interval, 30/1, within noise |
| mlx-lm-batched | 1.016 [1.015, 1.023], 36/3 | 1.051 no interval, 12/1, within noise | not run |
| mlx-lm-sequential-no-cache | 0.865 [0.863, 0.874], 36/3 | not run | not run |
| mlx-lm-batched-no-cache-twin | 0.999 [0.998, 1.001], 36/3 | 1.007 no interval, 12/1 | 0.959 no interval, 30/1 |

### Peak: TPOT, ratio to `mlx-lm-batched-no-cache`

| Engine | c1 | c4 | c16, DISCARDED |
|---|---|---|---|
| ullm | 1.002 [1.001, 1.115], 36/3, within noise | 1.020 no interval, 12/1, within noise | 1.113 no interval, 30/1, within noise |
| mlx-lm-batched | 1.002 [0.999, 1.006], 36/3, within noise | 1.036 no interval, 12/1, within noise | not run |
| mlx-lm-sequential-no-cache | 0.993 [0.992, 0.996], 36/3 | not run | not run |
| mlx-lm-batched-no-cache-twin | 1.001 [1.000, 1.002], 36/3 | 1.005 no interval, 12/1 | 1.005 no interval, 30/1 |

### Peak: E2E, ratio to `mlx-lm-batched-no-cache`

| Engine | c1 | c4 | c16, DISCARDED |
|---|---|---|---|
| ullm | 0.986 [0.975, 1.098], 36/3 | 0.901 no interval, 12/1, within noise | 0.932 no interval, 30/1, within noise |
| mlx-lm-batched | 1.003 [1.002, 1.007], 36/3 | 1.038 no interval, 12/1, within noise | not run |
| mlx-lm-sequential-no-cache | 0.981 [0.979, 0.985], 36/3 | not run | not run |
| mlx-lm-batched-no-cache-twin | 1.000 [1.000, 1.003], 36/3 | 1.008 no interval, 12/1 | 0.992 no interval, 30/1 |

### Peak: Output tok/s, ratio to `mlx-lm-batched-no-cache`

| Engine | c1 | c4 | c16, DISCARDED |
|---|---|---|---|
| ullm | 1.035 [0.930, 1.043], 3/3 | 1.099 no interval, 1/1, within noise | 1.065 no interval, 1/1, within noise |
| mlx-lm-batched | 1.002 [0.995, 1.020], 3/3, within noise | 0.975 no interval, 1/1, within noise | not run |
| mlx-lm-sequential-no-cache | 1.035 [1.032, 1.044], 3/3 | not run | not run |
| mlx-lm-batched-no-cache-twin | 1.014 [0.999, 1.026], 3/3 | 1.027 no interval, 1/1 | 1.019 no interval, 1/1 |

### Sustained: TTFT, ratio to `mlx-lm-batched-no-cache`

| Engine | c1 | c2 | c4 | c8 | c16, DISCARDED |
|---|---|---|---|---|---|
| ullm | 0.841 [0.793, 0.863], 204/17 | 0.548 [0.505, 0.586], 240/20 | 0.465 [0.424, 0.526], 228/19 | 0.424 [0.375, 0.482], 216/9 | 0.368 [0.354, 0.384], 180/6 |
| mlx-lm-batched | 1.006 [0.954, 1.024], 204/17, within noise | 1.008 [0.982, 1.031], 240/20, within noise | 1.007 [0.990, 1.035], 228/19, within noise | 0.995 [0.933, 1.026], 216/9, within noise | not run |
| mlx-lm-sequential-no-cache | 0.910 [0.894, 0.928], 204/17 | not run | not run | not run | not run |
| mlx-lm-batched-no-cache-twin | 1.002 [0.982, 1.019], 204/17 | 0.997 [0.984, 1.018], 240/20 | 1.002 [0.984, 1.025], 228/19 | 1.013 [0.977, 1.051], 216/9 | 1.019 [0.899, 1.285], 180/6 |

### Sustained: TPOT, ratio to `mlx-lm-batched-no-cache`

| Engine | c1 | c2 | c4 | c8 | c16, DISCARDED |
|---|---|---|---|---|---|
| ullm | 1.014 [1.008, 1.024], 204/17 | 1.003 [0.995, 1.027], 240/20, within noise | 1.044 [1.018, 1.080], 228/19 | 1.008 [0.964, 1.038], 216/9, within noise | 1.095 [1.037, 1.209], 180/6, within noise |
| mlx-lm-batched | 1.004 [0.998, 1.014], 204/17 | 1.010 [0.993, 1.025], 240/20, within noise | 1.013 [0.997, 1.036], 228/19, within noise | 0.983 [0.915, 1.007], 216/9, within noise | not run |
| mlx-lm-sequential-no-cache | 0.995 [0.991, 1.001], 204/17 | not run | not run | not run | not run |
| mlx-lm-batched-no-cache-twin | 0.999 [0.996, 1.003], 204/17 | 1.003 [0.990, 1.021], 240/20 | 1.008 [0.994, 1.017], 228/19 | 1.004 [0.989, 1.032], 216/9 | 1.008 [0.920, 1.266], 180/6 |

### Sustained: E2E, ratio to `mlx-lm-batched-no-cache`

| Engine | c1 | c2 | c4 | c8 | c16, DISCARDED |
|---|---|---|---|---|---|
| ullm | 0.989 [0.981, 0.999], 204/17 | 0.919 [0.900, 0.938], 240/20 | 0.912 [0.887, 0.930], 228/19 | 0.891 [0.844, 0.921], 216/9 | 0.923 [0.854, 0.960], 180/6, within noise |
| mlx-lm-batched | 1.003 [0.994, 1.013], 204/17, within noise | 1.008 [0.989, 1.024], 240/20, within noise | 1.004 [0.985, 1.027], 228/19, within noise | 0.982 [0.946, 1.007], 216/9, within noise | not run |
| mlx-lm-sequential-no-cache | 0.981 [0.973, 0.989], 204/17 | not run | not run | not run | not run |
| mlx-lm-batched-no-cache-twin | 0.999 [0.996, 1.003], 204/17 | 1.004 [0.992, 1.016], 240/20 | 1.009 [0.993, 1.020], 228/19 | 1.010 [0.995, 1.034], 216/9 | 1.006 [0.906, 1.253], 180/6 |

### Sustained: Output tok/s, ratio to `mlx-lm-batched-no-cache`

| Engine | c1 | c2 | c4 | c8 | c16, DISCARDED |
|---|---|---|---|---|---|
| ullm | 1.006 [0.994, 1.020], 17/17 | 1.079 [1.053, 1.099], 20/20 | 1.093 [1.050, 1.117], 19/19 | 1.120 [1.086, 1.170], 9/9 | 1.064 [1.000, 1.156], 6/6, within noise |
| mlx-lm-batched | 0.993 [0.974, 0.996], 17/17 | 0.991 [0.977, 1.008], 20/20, within noise | 0.989 [0.959, 0.999], 19/19, within noise | 1.007 [0.974, 1.090], 9/9, within noise | not run |
| mlx-lm-sequential-no-cache | 1.012 [1.008, 1.035], 17/17 | not run | not run | not run | not run |
| mlx-lm-batched-no-cache-twin | 1.001 [0.997, 1.005], 17/17 | 0.992 [0.972, 1.006], 20/20 | 0.987 [0.978, 1.004], 19/19 | 0.990 [0.960, 1.027], 9/9 | 0.996 [0.780, 1.081], 6/6 |
