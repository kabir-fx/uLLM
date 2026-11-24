# Steps: m2-c1, concurrency 1, VALID

`ullm` from `ullm-<block>.jsonl.gz`. A step counts when it starts and ends inside the client's measured span of its chunk, so warm-up requests never count. Step time includes packing the batch KV, the forward pass and sampling.

## Peak: 3 blocks, 0 events dropped

| Figure | Value |
|---|---|
| Requests | 36 |
| Measured span in prefill, decode, outside steps | 6.7%, 92.8%, 0.5% |
| Prefill, ms per 1,000 prompt tokens, p50, p90 | 359.7, 412.7 |
| Gap between steps with a request live, ms, p50, p90 | 0.05, 0.06 |
| Others' prefill as a share of a request's decode span, p50, p90 | 0.000, 0.000 |
| Rows in a decode step, mean over steps | 1.00 |

| Rows | Decode steps | Step ms, p50, p90 |
|---|---|---|
| 1 | 5544 | 14.26, 16.14 |

## Sustained: 17 blocks, 0 events dropped

| Figure | Value |
|---|---|
| Requests | 204 |
| Measured span in prefill, decode, outside steps | 11.8%, 87.8%, 0.4% |
| Prefill, ms per 1,000 prompt tokens, p50, p90 | 827.3, 1144.6 |
| Gap between steps with a request live, ms, p50, p90 | 0.03, 0.12 |
| Others' prefill as a share of a request's decode span, p50, p90 | 0.000, 0.000 |
| Rows in a decode step, mean over steps | 1.00 |

| Rows | Decode steps | Step ms, p50, p90 |
|---|---|---|
| 1 | 32989 | 16.79, 19.51 |
