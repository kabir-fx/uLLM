# 0004: Synthetic telemetry fixtures and the trace checker

**Status:** accepted · **Task:** M0-T5

## Context

Lane B builds the terminal against recorded telemetry before the engine exists (DEC-2, 3.5). The fixtures have to look like a real engine's stream, stay consistent with the protocol's meaning and not only its schema, and never be mistaken for measurements.

## Decision

1. **A small discrete-event simulation, not random events.** `ullm.telemetry.synthetic` follows the scheduling rules in E6: FCFS admission that stops at the first request that does not fit, one prefill per step, batched decode, and the newest request preempted first. With blocks on, it also follows E8, E10 and E11: a fixed pool, a shared prefix matched block by block up to the first miss, full blocks kept as Cached when released, LRU eviction, and preemption to the front of the queue. Timings come from a stated toy latency model. The goal is a stream whose shapes are right, not a performance model.
2. **Two scenarios.** `steady-chat` (workload A, contiguous KV, a queue at the batch cap) covers the M1 and M2 panels. `prefix-pressure` (workloads B and C, a 1024-block pool) covers the M3 panels: every block state, prefix hits, evictions, preemptions and the long-prefill stall. Tests assert that each feature stays covered, so tuning cannot silently remove one.
3. **Labelled as synthetic everywhere.** `hello.source` is `"fixture"` and `hello.model` is `"synthetic"`, `machine` is null, and the fixtures' README says that no number from them may be quoted. The terminal must label this data wherever it appears.
4. **Deterministic across platforms.** Time is integer microseconds, the only randomness is `Random(seed).random()`, and there are no floating-point library calls whose last bits could differ between macOS and Linux. Freshness is checked on the decompressed text, because gzip bytes depend on the zlib build.
5. **A trace checker, `ullm.telemetry.check`.** It checks what the schema cannot: gap-free and ordered frames, every request following the E4 life cycle, steps only containing requests that can be in them, and token counts in `req.finished` matching the tokens the steps sampled. The fixtures must pass it. From M2-T4, real recordings must pass it too, which makes it a test of the engine's telemetry as well. Its own tests feed it broken stories.
6. **Location.** `terminal/fixtures/telemetry/`, next to the code that consumes them. `make fixtures` regenerates them, and `make check` fails when they are stale.

## Alternatives considered

- **Record from `ullm sim`.** The most faithful option, but the simulator does not exist until M1-T3 and the recorder until M2-T4. Real recordings will be added beside the fixtures then; the fixtures stay as the deterministic test inputs.
- **Hand-written traces.** Small and readable, but too short to exercise the charts, and easy to make inconsistent.
- **Random events that pass the schema.** Cheap, but a request could decode before its prefill, and the terminal would be built against impossible data.

## Consequences

- O1's proof gains an item: recordings from the engine pass `check_trace`.
- The checker currently requires event times to never go backwards in stream order. M2-T4 decides whether the real bus guarantees that (the engine drains arrivals at the start of a step) or whether the rule needs a bounded tolerance, recorded then.
