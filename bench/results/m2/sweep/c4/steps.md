# Steps: m2-c4, concurrency 4, VALID

`ullm` from `ullm-<block>.jsonl.gz`. A step counts when it starts and ends inside the client's measured span of its chunk, so warm-up requests never count. Step time includes packing the batch KV, the forward pass and sampling.

## Peak: 1 blocks, 0 events dropped

| Figure | Value |
|---|---|
| Requests | 12 |
| Measured span in prefill, decode, outside steps | 16.0%, 83.4%, 0.5% |
| Prefill, ms per 1,000 prompt tokens, p50, p90 | 372.9, 426.6 |
| Gap between steps with a request live, ms, p50, p90 | 0.07, 0.14 |
| Others' prefill as a share of a request's decode span, p50, p90 | 0.118, 0.239 |
| Rows in a decode step, mean over steps | 3.41 |

| Rows | Decode steps | Step ms, p50, p90 |
|---|---|---|
| 1 | 72 | 16.45, 17.23 |
| 2 | 30 | 16.87, 17.68 |
| 3 | 58 | 17.63, 27.67 |
| 4 | 410 | 18.44, 19.55 |

## Sustained: 19 blocks, 0 events dropped

| Figure | Value |
|---|---|
| Requests | 228 |
| Measured span in prefill, decode, outside steps | 29.2%, 70.6%, 0.3% |
| Prefill, ms per 1,000 prompt tokens, p50, p90 | 1092.2, 1352.1 |
| Gap between steps with a request live, ms, p50, p90 | 0.04, 0.16 |
| Others' prefill as a share of a request's decode span, p50, p90 | 0.218, 0.374 |
| Rows in a decode step, mean over steps | 3.41 |

| Rows | Decode steps | Step ms, p50, p90 |
|---|---|---|
| 1 | 1435 | 20.34, 23.54 |
| 2 | 645 | 22.13, 26.83 |
| 3 | 723 | 24.41, 36.82 |
| 4 | 7932 | 24.52, 32.06 |
