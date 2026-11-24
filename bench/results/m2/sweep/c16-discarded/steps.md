# Steps: m2-c16, concurrency 16, DISCARDED

`ullm` from `ullm-<block>.jsonl.gz`. A step counts when it starts and ends inside the client's measured span of its chunk, so warm-up requests never count. Step time includes packing the batch KV, the forward pass and sampling.

## Sustained: 7 blocks, 0 events dropped

| Figure | Value |
|---|---|
| Requests | 210 |
| Measured span in prefill, decode, outside steps | 29.7%, 69.5%, 0.8% |
| Prefill, ms per 1,000 prompt tokens, p50, p90 | 827.2, 1072.5 |
| Gap between steps with a request live, ms, p50, p90 | 0.38, 0.54 |
| Others' prefill as a share of a request's decode span, p50, p90 | 0.244, 0.410 |
| Rows in a decode step, mean over steps | 10.93 |

| Rows | Decode steps | Step ms, p50, p90 |
|---|---|---|
| 1 | 189 | 21.29, 23.59 |
| 2 | 217 | 24.43, 26.77 |
| 3 | 176 | 25.91, 30.20 |
| 4 | 153 | 29.38, 32.55 |
| 5 | 77 | 30.77, 49.82 |
| 6 | 114 | 30.71, 34.67 |
| 7 | 121 | 37.76, 51.72 |
| 8 | 71 | 38.70, 47.15 |
| 9 | 85 | 46.22, 53.94 |
| 10 | 75 | 55.30, 72.69 |
| 11 | 65 | 54.66, 85.39 |
| 12 | 47 | 45.57, 92.40 |
| 13 | 50 | 60.55, 90.92 |
| 14 | 57 | 56.60, 135.08 |
| 15 | 124 | 98.66, 112.67 |
| 16 | 1451 | 78.58, 93.51 |
