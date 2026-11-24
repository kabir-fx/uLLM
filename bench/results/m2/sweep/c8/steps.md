# Steps: m2-c8, concurrency 8, VALID

`ullm` from `ullm-<block>.jsonl.gz`. A step counts when it starts and ends inside the client's measured span of its chunk, so warm-up requests never count. Step time includes packing the batch KV, the forward pass and sampling.

## Sustained: 9 blocks, 0 events dropped

| Figure | Value |
|---|---|
| Requests | 216 |
| Measured span in prefill, decode, outside steps | 33.4%, 66.0%, 0.6% |
| Prefill, ms per 1,000 prompt tokens, p50, p90 | 930.1, 1253.0 |
| Gap between steps with a request live, ms, p50, p90 | 0.16, 0.35 |
| Others' prefill as a share of a request's decode span, p50, p90 | 0.311, 0.429 |
| Rows in a decode step, mean over steps | 6.39 |

| Rows | Decode steps | Step ms, p50, p90 |
|---|---|---|
| 1 | 417 | 19.95, 24.14 |
| 2 | 390 | 23.88, 28.49 |
| 3 | 214 | 25.27, 31.35 |
| 4 | 235 | 24.01, 32.49 |
| 5 | 236 | 30.02, 33.90 |
| 6 | 224 | 27.10, 34.70 |
| 7 | 310 | 43.66, 67.90 |
| 8 | 3414 | 33.91, 45.19 |
