# DISCARDED: m2-c16-cache, concurrency 16

- Baseline `mlx-lm-batched-no-cache` and its twin; engines `mlx-lm-batched`. One server per chunk.
- 30 prompts and 16 warm-up requests a chunk; settle 0 s; seed 0.
- Commit `c51200e20812` (dirty tree). Apple M4, 10 cores, 16 GB, macOS 27.2.1; mlx 0.32.4, mlx-lm 0.32.2. At the start: AC True, battery 65%, Low Power Mode False. Over 4 probes: charging in 4, lid open in 4.

## Gates

- **Conditions**: passed. 122 clock samples, 4 probes on AC.
- **Liveness**: FAILED. 0 of 8 blocks, ended failed request; 38 of 138 requests failed; 0 prompt lengths differ.
- **Memory**: FAILED. swap grew 144 MB above the first probe's.
- **Twin**: passed. no Peak or Sustained block.

## Regimes

Peak 0, Sustained 0, Transition 1 blocks. Transition blocks are in raw.jsonl and in no table.
