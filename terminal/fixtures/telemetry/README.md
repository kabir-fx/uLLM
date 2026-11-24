# Synthetic telemetry traces

Recorded telemetry for building and testing the terminal before the engine exists. Each file is gzip-compressed JSONL: one telemetry v1 frame per line, exactly as `/telemetry/stream` sends it (C2, decision 0003). Browsers can read them with `DecompressionStream("gzip")`.

**These are synthetic. They are not measurements of uLLM or of any machine.** They come from a small simulation of the scheduling rules in the roadmap (`src/ullm/telemetry/synthetic.py`), timed by a toy latency model chosen to look plausible. Every trace starts with a `hello` event whose `source` is `"fixture"` and whose `model` is `"synthetic"`. The terminal must label such data as simulated wherever it is shown, and no number from these files may appear in a README, write-up or resume.

| File | Shape | Exercises |
|---|---|---|
| `steady-chat.jsonl.gz` | Workload A: 10 closed-loop clients, prompts of 100–1000 tokens, outputs up to 300, batch cap 8, contiguous KV (no blocks), about 45 s | Request feed, ticker, step chart, batch strip, a short queue, two client aborts |
| `prefix-pressure.jsonl.gz` | Workloads B and C: open-loop arrivals, 75% of requests share a 2048-token system prompt, two very long prompts, a 1024-block pool, about 50 s | Everything above, plus `kv.snapshot` (owned, shared, Cached and free blocks), prefix-cache hits, evictions, preemptions and the long-prefill stall |
| `stress.jsonl.gz` | Open-loop arrivals at 4 a second, 64 requests running, a 2,304-block pool, 218 requests, about 70 s | The terminal's frame-rate budget (T7): about 2,300 blocks in each snapshot, and 200 rows in the finished tab and the feed |

Both traces pass `ullm.telemetry.check.check_trace`: frames are gap-free and ordered, every request follows the life cycle (arrived, admitted, first token, then finished or aborted, with preemptions in between), steps only include requests that can be in them, and each `req.finished` token count equals the tokens its steps sampled.

Regenerate with `make fixtures`. `make check` fails if a committed trace no longer matches the generator. The generator is deterministic on every platform (integer microseconds, no floating-point library calls), and the check compares decompressed text because gzip bytes can differ between zlib builds.
