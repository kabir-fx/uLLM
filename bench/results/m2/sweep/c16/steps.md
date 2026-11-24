# Steps: m2-c16, concurrency 16, DISCARDED

`ullm` from `ullm-<block>.jsonl.gz`. A step counts when it starts and ends inside the client's measured span of its chunk, so warm-up requests never count. Step time includes packing the batch KV, the forward pass and sampling.

## Peak: 1 blocks, 0 events dropped

| Figure | Value |
|---|---|
| Requests | 30 |
| Measured span in prefill, decode, outside steps | 23.9%, 75.2%, 0.9% |
| Prefill, ms per 1,000 prompt tokens, p50, p90 | 374.2, 418.6 |
| Gap between steps with a request live, ms, p50, p90 | 0.21, 0.39 |
| Others' prefill as a share of a request's decode span, p50, p90 | 0.170, 0.301 |
| Rows in a decode step, mean over steps | 11.31 |

| Rows | Decode steps | Step ms, p50, p90 |
|---|---|---|
| 1 | 27 | 17.40, 19.47 |
| 2 | 15 | 17.73, 20.65 |
| 3 | 15 | 18.27, 19.85 |
| 4 | 15 | 20.25, 22.06 |
| 5 | 31 | 21.28, 22.72 |
| 6 | 26 | 21.23, 22.78 |
| 7 | 4 | 22.23, 47.27 |
| 8 | 6 | 22.07, 53.35 |
| 9 | 9 | 23.93, 54.92 |
| 10 | 16 | 26.41, 27.80 |
| 11 | 2 | 24.61, 60.73 |
| 12 | 4 | 25.43, 70.09 |
| 13 | 22 | 29.91, 30.74 |
| 14 | 9 | 29.27, 87.04 |
| 15 | 21 | 86.40, 97.58 |
| 16 | 197 | 41.30, 45.50 |

## Sustained: 6 blocks, 0 events dropped

| Figure | Value |
|---|---|
| Requests | 180 |
| Measured span in prefill, decode, outside steps | 27.4%, 71.6%, 1.1% |
| Prefill, ms per 1,000 prompt tokens, p50, p90 | 464.7, 590.9 |
| Gap between steps with a request live, ms, p50, p90 | 0.23, 0.47 |
| Others' prefill as a share of a request's decode span, p50, p90 | 0.205, 0.404 |
| Rows in a decode step, mean over steps | 10.87 |

| Rows | Decode steps | Step ms, p50, p90 |
|---|---|---|
| 1 | 180 | 16.01, 18.11 |
| 2 | 190 | 16.75, 17.25 |
| 3 | 157 | 17.66, 20.19 |
| 4 | 141 | 20.65, 23.82 |
| 5 | 65 | 20.48, 35.72 |
| 6 | 95 | 20.25, 27.11 |
| 7 | 99 | 21.94, 27.60 |
| 8 | 49 | 21.60, 47.65 |
| 9 | 80 | 24.41, 31.95 |
| 10 | 35 | 24.00, 60.84 |
| 11 | 63 | 25.71, 59.83 |
| 12 | 36 | 26.21, 68.43 |
| 13 | 49 | 34.99, 70.49 |
| 14 | 50 | 32.45, 90.12 |
| 15 | 108 | 78.77, 90.43 |
| 16 | 1260 | 46.11, 52.37 |
