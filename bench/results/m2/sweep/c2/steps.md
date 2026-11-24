# Steps: m2-c2, concurrency 2, VALID

`ullm` from `ullm-<block>.jsonl.gz`. A step counts when it starts and ends inside the client's measured span of its chunk, so warm-up requests never count. Step time includes packing the batch KV, the forward pass and sampling.

## Sustained: 20 blocks, 0 events dropped

| Figure | Value |
|---|---|
| Requests | 240 |
| Measured span in prefill, decode, outside steps | 21.3%, 78.5%, 0.3% |
| Prefill, ms per 1,000 prompt tokens, p50, p90 | 1002.2, 1298.3 |
| Gap between steps with a request live, ms, p50, p90 | 0.03, 0.13 |
| Others' prefill as a share of a request's decode span, p50, p90 | 0.101, 0.226 |
| Rows in a decode step, mean over steps | 1.90 |

| Rows | Decode steps | Step ms, p50, p90 |
|---|---|---|
| 1 | 2127 | 18.46, 22.23 |
| 2 | 18162 | 19.40, 24.34 |
