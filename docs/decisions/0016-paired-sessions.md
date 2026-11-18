# 0016: Benchmarks as paired sessions, sorted by GPU cap

**Status:** proposed, the author's answers recorded below · **Task:** M1-T7 (B1, B4, B5, B6) · **Implements:** the author's `Fanless Mac Benchmarking - Final Approach.md` (repo root, not committed), item by item in "Conformance" · **Supersedes:** the run plan of 0014 (repeats, cooldowns, the clock check) and 0013's B5 plan check, once accepted

## Problem

The benchmark Mac is a fanless M4 Air. Under load its GPU holds 1470 MHz for about 400 to 600 s from cold, then the OS caps it, and a 120 s cooldown does not restore it. Decision 0014's plan (each target alone, three repeats, cooldowns, "no clock drop" as a check) failed every run of the first matrix, and a target's numbers depended on when it ran. macOS cannot pin the GPU clock.

The Final Approach replaces that plan: engines take turns on the same prompts in one continuous session, with a twin of the baseline as the noise floor, and blocks are sorted into Peak and Sustained afterwards from the clock log. This record is the build contract for it. A first version of this record had three faults this version fixes: its twin gate would have discarded spike 1, its concurrency-8 group could not fit in memory, and some of its evidence came from the thermal-label segmentation it rejects.

## Evidence

Two spikes in `bench/.scratch/thermal-spike/`: spike 1 (`run-20261118-141243`, before decision 0015) and spike 2 (`run-20261118-163540`, after). Every figure below was re-derived under the capped rule this record adopts, by the scripts in `bench/.scratch/thermal-spike/design-checks/`; each script's output is the `.out` file beside it. The memory figures come from the three `*_probe.py` scripts there, run on 18 November 2026 with desktop apps open. A server's footprint does not depend on other apps: MLX's cache limit and the wired limit come from the device, and `phys_footprint` counts compressed and swapped pages. Swap does depend on them, so swap figures from these probes say nothing about an idle Mac.

| Fact | Measured | Consequence |
|---|---|---|
| The capped rule reproduces the approach's sorting | Spike 1 outside Low Power Mode: 3 Peak, 5 Transition, 4 Sustained when the baseline and the twin must **each** be capped for half their busy time. Pooling the two gives 3, 4, 5. Spike 2: 7, 2, 7. Margins 0.3 and 0.5 give the same counts; 0.1 leaves spike 1 one Peak block | `CAP_MARGIN = 0.3`; the baseline and the twin each |
| Twins agree on TPOT in every regime | Spike 1: Peak 0.999 [0.9988, 1.0014] (36 prompts, 3 blocks), Sustained 0.999 [0.964, 1.057] (48, 4). Spike 2: Peak 0.998 [0.9709, 1.0001] (84, 7), Sustained 1.000 [0.998, 1.002] (84, 7) | The twin's TPOT ratio is the gate |
| Other twin metrics are noisier | Spike 1 Peak output tokens per second: 0.992 over 3 blocks. Spike 2 Peak end-to-end: interval [0.9728, 1.0000], which excludes 1.00 | A twin gate on every metric discards both spikes. The other metrics carry the twin's ratio as their floor |
| The interval test is fragile | Spike 2's Peak TPOT interval contains 1.00 by 0.0001 | The twin gate checks the median only |
| uLLM's ratio holds across regimes after 0015 | Spike 2 TPOT ratio 1.156 to 1.170 in Peak, 1.168 in Sustained. No gap between chunks over 100 ms, largest 36 ms. Spike 1: 662 such gaps, 25.8% of stream time | The paired ratio is the result |
| Sustained absolute numbers do not repeat | mlx-lm TPOT p50 in Sustained: 15.3 to 15.4 ms in spike 1, 14.4 ms in spike 2 | No absolute Sustained figure is a headline |
| A first request after a pause is slower | Spike 2: the first request's prefill cost per prompt token is 1.07 (uLLM), 1.11 and 1.14 (mlx-lm) times the median of the rest of its chunk, median over blocks. Spike 1, before 0015: 22 times for mlx-lm | Warm-up at the start of every chunk |
| The pressure label does not show the cap | Spike 2 block 5: the twin's chunk read Heavy in every sample, at 1464 MHz, requested 8.98, applied 8.995 | Sort on requested against applied, never on the label |
| The state fields | Each sample's `gpu.sw_requested_state` lists `P1` to `P15` and `gpu.sw_state` lists `SW_P1` to `SW_P15`, each with a share of the sample. `SW_P9` is the 1470 MHz step | Mean state index, requested and applied |
| The clock log aligns with the wall clock | Rebuilt from the first timestamp plus each sample's `elapsed_ns`, the clock ends 0.31 s and 0.77 s from the log's own last timestamp over 47 minutes | Join requests to samples on wall-clock time |
| The c1 servers fit loaded together | uLLM, mlx-lm sequential without prompt cache twice, and sequential with it, 12 longest prompts each, twice: 7.2 GB summed footprint, peaks 2.1, 2.0, 2.0 and 3.1 GB | c1 keeps its servers loaded |
| The c8 servers do not | Five c8 servers loaded, 24 longest prompts at concurrency 8 each: 29.3 GB summed, swap 0.4 to 18.8 GB. Batched `mlx_lm.server` alone: 7.0 GB (peak 8.4) without its prompt cache, 11.3 GB (peak 12.3) with it. Swap grew 94 and 111 MB per chunk with the cache, not at all without | c8 starts one server per chunk, the approach's fallback |
| Why batched mlx-lm holds so much | `BatchGenerator` clears MLX's buffer cache after each prefill step and every 512 decode steps (`mlx_lm/generate.py` lines 1230, 1238, 1824), and the server raises the wired limit to the recommended working set (`server.py` line 2080). Freed buffers pile up in between | A property of the baseline's defaults, reported, not tuned away |
| Swap on an idle Mac | Spike 2, idle, three servers loaded: 459 MB at the start, 427 MB at the end of 45 minutes, never higher. Spike 1, idle, before 0015: 0.5 to 7.3 GB, from uLLM | Swap does not grow on an idle Mac without a cause. Any growth discards |
| `powermetrics` and SIGINT | Spike 1 left a root `powermetrics` running after the harness sent `sudo` SIGINT. SIGTERM stopped it (the author's write-up) | Stop it with SIGTERM |

## Conformance with the Final Approach

| The approach says | This design | Deviation, and why |
|---|---|---|
| Fix uLLM's memory first | Decision 0015 | None. Spike 2 shows no stall |
| Engines take turns on the same prompts, rotating through every ordering | Blocks of chunks; orders from a Williams design | Every ordering of 4 engines is 24 orders, more than the 21 blocks. A Williams design puts each engine in each place, and after each other engine, equally often. For 3 engines it is every ordering, as the spikes ran |
| One continuous session, no cooldowns, sorted afterwards | `bench.session`; `bench.regimes` | None. The session settles first with `powermetrics` logging, as the spikes did for 240 s |
| Capped: applied state below requested, margin 0.3, fixed before the run | `regimes.capped`, `CAP_MARGIN` | None |
| Peak: no chunk has a capped sample. Sustained: the baseline's chunks, twin included, capped for half their busy time. Else Transition. Whole blocks | `regimes.classify` | "Twin included" is read as each of the two, which reproduces the approach's counts |
| A twin of the baseline | The session derives it from `baseline` | None. A group without a twin cannot be written |
| AC power, lid open, apps closed, battery and charging recorded | B5 refusal; `env.json`; a `Probe` after every chunk | The lid and open apps are recorded, not refused. B5 adds: no other `powermetrics` may run |
| One real completion per server as the liveness check | Warm-up requests are real completions; a failure ends the session | None |
| Footprint and swap at every block | A `Probe` after every chunk | Finer than asked: the engine that just ran is the only one whose memory can change |
| 12 prompts per chunk, stratified by prompt length, lists and seed committed | `plan.plan`; `plan.json` in the session | None |
| 2 warm-up requests per chunk | c1: 2. c8: 8 | c8 is untested. A fresh process at concurrency 8 meets batch 8 first in its warm-up (inferred) |
| Servers stay loaded if their footprint leaves room, else one at a time, said in the table | `Group.servers`: `resident` or `per_chunk`, named in every table | None. Measured: c1 fits, c8 does not |
| Discard if swap grows | Swap in use never above the first probe's | None. The first version allowed 256 MB from probes run with apps open. The author chose zero |
| Discard if a server's footprint passes a declared limit | `EngineSpec.footprint_gb`, on the peak | One limit per engine: batched mlx-lm holds 12 GB by default, uLLM about 2 |
| Discard if any request fails | Liveness gate | Also counts a reply whose prompt length differs from workload A's (claim 18) |
| Twin gate: interval contains 1.00, median within 0.5% Peak and 3% Sustained | TPOT median within the limits | The author chose the median alone: spike 2's Peak TPOT interval held 1.00 by 0.0001, and the floor rule already stops claims inside the twin's noise. The approach's floors are TPOT floors |
| Paired ratio per regime, Peak first; bootstrap over whole blocks | `paired.ratio` | None |
| Absolute numbers per regime: p50, p95, p99 with its interval, count, clock mix, conditions | `analysis` | None |
| The twin's ratio is the noise floor; no smaller engine gap is claimed | `paired.floor`, `paired.claimed` | None |
| Stream gaps: largest, and share of stream time in stalls | `analysis` | None |
| Never mix regimes in a row; never scale by clock | `analysis` | None |
| Low Power Mode as an optional third table | Not built | Switching it mid-session needs a root helper. Optional in the approach |
| Stop `powermetrics` with SIGTERM | `conditions.PowerMetrics.stop` | None |
| Commit the spike folder | `bench/results/m1-spikes/` | None |
| An mlx-lm sequential row at concurrency 8 | `mlx-lm-sequential-no-cache` in `m1-c8` | None |

Not in the approach, added: `caffeinate` holds the Mac awake for the session (the spikes ran under it), and every probe checks AC power and Low Power Mode, so B5 holds for the whole session and not only at its start.

## Usage

```
make bench-baselines
  # uv run python -m bench.session bench/groups/m1-c1.json --settle 240 --out bench/results/m1/c1
  # uv run python -m bench.session bench/groups/m1-c8.json --settle 600 --out bench/results/m1/c8
  # uv run python -m bench.analysis bench/results/m1/c1 bench/results/m1/c8 --compare > bench/results/m1/comparison.md

uv run python -m bench.session bench/groups/m1-c8.json --smoke --blocks 1 --out bench/.scratch/c8
uv run python -m bench.analysis bench/results/m1/c1          # rewrites summary.md
```

`bench.session` exits 0 when the session is VALID, 1 when it is DISCARDED, and 2 when it refuses to start. Make stops at the first non-zero exit, so a discarded c1 session never starts c8. While it runs, the session prints one line per chunk: block, engine, place, requests, failures, TTFT p50 and TPOT p50. `summary.md` opens with VALID, DISCARDED or SMOKE, then every gate with its numbers.

## Shape

### Files and records (`bench/schema.py`)

A group file names everything fixed before a session, so git records it before the first request:

```python
class EngineSpec(Frozen):
    name: str
    launch: str  # may hold {model} and {port}; run with shlex, no shell
    model: str | None  # the id to send; "{model}" is the pinned snapshot path
    extra_body: dict[str, JsonValue] = {}
    footprint_gb: float = Field(gt=0)  # limit on this server's peak footprint


class Gates(Frozen):
    twin_tpot_peak: float = Field(gt=0)  # largest |twin TPOT median ratio - 1| in Peak
    twin_tpot_sustained: float = Field(gt=0)


class Group(Frozen):
    name: str
    concurrency: int = Field(ge=1)
    servers: Literal["resident", "per_chunk"]
    prompts_per_chunk: int = Field(ge=2)
    warmup_per_chunk: int = Field(ge=1)
    seed: int
    baseline: EngineSpec  # always started twice: <name> and <name>-twin
    engines: tuple[EngineSpec, ...]  # names unique; none ends in -twin
    gates: Gates
```

The twin is not listed: the session derives it from `baseline`, limits included. Ports are derived too: 8100 plus the engine's index in `(baseline, twin, *engines)`.

`loadgen.closed_loop` returns `Sent`, the bench's record of one request, and no longer knows phases or repeats. `bench.run` turns each into the `RequestRecord` it writes now, so its files do not change.

```python
class Sent(Frozen):
    id: str
    sent_unix: float  # wall clock at send: the clock powermetrics stamps
    chunk_ms: tuple[float, ...]  # after the send, on perf_counter
    end_ms: float
    usage: Usage | None
    finish_reason: str | None
    error: str | None


class SessionRecord(Frozen):  # one line of raw.jsonl
    engine: str
    block: int  # -1: the opening liveness requests of a resident session
    pos: int  # the engine's place in the block's order; -1 with block -1
    phase: Literal["liveness", "warmup", "measure"]
    sent: Sent


class Probe(Frozen):  # one line of probes.jsonl, after every chunk
    at_unix: float
    block: int  # -1: before any server starts
    engine: str | None  # whose chunk just ended; None only on the first line
    footprint_bytes: int | None  # phys_footprint summed over its process group
    peak_bytes: int | None  # phys_footprint_peak, summed the same way
    swap_used_bytes: int
    power: Power  # gains charging and lid_open
```

A validator holds `engine is None` exactly when both footprints are `None`.

A session directory holds `group.json` (the group as run), `plan.json` (each block's prompt ids, the warm-up ids, the orders), `env.json` (the machine before any server starts), `raw.jsonl`, `probes.jsonl`, `powermetrics.plist.gz`, one `<engine>.log` per server, `session.json` (the served model per engine, start and end times, blocks run, and how it ended: `completed`, `failed request` or `interrupted`) and `summary.md`.

### `bench/target.py`

`Target`, `ready` and `port_in_use` move here from `bench/run.py` unchanged, so `bench.session` imports no private names from `bench.run`.

### `bench/plan.py`, pure

```python
@dataclass(frozen=True)
class Plan:
    blocks: tuple[tuple[WorkloadRequest, ...], ...]  # block b: one prompt from each of k strata
    warmup: tuple[WorkloadRequest, ...]  # the requests no block holds; never measured
    orders: tuple[tuple[str, ...], ...]

    def order(self, block: int) -> tuple[str, ...]: ...  # orders[block % len(orders)]


def plan(
    requests: Sequence[WorkloadRequest],
    engines: Sequence[str],
    k: int,
    warmup_per_chunk: int,
    seed: int,
) -> Plan: ...
def williams(names: Sequence[str]) -> tuple[tuple[str, ...], ...]: ...
```

Requests are sorted by `(prompt_tokens, id)`. With `n = len(requests) // k`, stratum `s` is the slice `[s*n, (s+1)*n)`, and the rest are the warm-up pool. `plan` raises `ValueError` when the pool holds fewer than `warmup_per_chunk`. Each stratum is shuffled with `seed`, block `b` takes the `b`-th prompt of every stratum, and each block is shuffled with `seed`. Every engine gets a block's prompts in the same order. Every chunk's warm-up is `warmup[:warmup_per_chunk]`. `williams` gives `n` orders for an even `n` and `2n` for an odd `n`.

| Group | Engines | Blocks | Warm-up pool | Orders |
|---|---|---|---|---|
| `m1-c1`, k 12, warm-up 2 | 4 | 21 | the 4 longest prompts | 4 |
| `m1-c8`, k 24, warm-up 8 | 5 | 10 | the 16 longest | 10 |

### `bench/session.py`

The only module that starts servers for a claim, and the only writer of a session directory. One asyncio task sends every request and writes every file.

1. Refuse, exit 2, unless B5 holds (`conditions.violations`), every derived port is free, and `--out` does not exist. `--smoke` skips B5 only.
2. Create `--out`, write `group.json`, `plan.json` and `env.json`. Start `caffeinate -ims -w <own pid>` and `powermetrics`. Write the first `Probe`. Wait `--settle` seconds.
3. `resident`: start every server, wait until each is ready, and send each one `warmup_per_chunk` liveness requests at the group's concurrency.
4. For each block, for each engine in `plan.order(block)`: under `per_chunk`, start the server and wait until it is ready. Send the warm-up requests, then the block's prompts, each through `closed_loop` at the group's concurrency. Write a `Probe` for that engine. Under `per_chunk`, stop the server. A failed request or a server that is not ready ends the session after this chunk.
5. In `finally`: stop every server (SIGINT, then SIGKILL after 30 s), stop `powermetrics` with SIGTERM, gzip its log, write `session.json`, and run the analysis.

`--blocks N` runs the first N blocks. A session with fewer blocks than its plan is never VALID.

### `bench/regimes.py`, pure except the reader

```python
@dataclass(frozen=True)
class GpuSample:
    end_unix: float
    elapsed_s: float
    busy: float  # 1 - idle_ratio
    requested: float | None  # mean P index requested, None when the GPU barely ran
    applied: float | None  # mean SW_P index applied
    steps: Mapping[int, float]  # MHz -> share of the sample


@dataclass(frozen=True)
class ChunkSpan:
    engine: str
    start_unix: float  # the first measured request's send
    end_unix: float  # the last measured request's end


class Regime(Enum):
    PEAK, SUSTAINED, TRANSITION


CAP_MARGIN = 0.3  # of one P state
SUSTAINED_SHARE = 0.5


def read_powermetrics(path: Path) -> list[GpuSample]: ...
def capped(sample: GpuSample) -> bool: ...
def capped_share(span: ChunkSpan, samples: Sequence[GpuSample]) -> float: ...
def classify(
    block: Sequence[ChunkSpan], samples: Sequence[GpuSample], deciders: tuple[str, str]
) -> Regime: ...
```

A sample is capped when `applied < requested - CAP_MARGIN`. A sample belongs to a span when its interval overlaps it, so a sample at a block boundary counts in both blocks. `capped_share` weights samples by busy time. Spans cover measured requests only: warm-up and server start-up are outside them. `deciders` is the baseline and its twin: `mlx_lm.server` asks for P9 whenever it is busy, while uLLM asked for about P8.4 in spike 2, so the cap shows reliably only on the baseline. The reader skips a cut-off last sample and rebuilds the clock from the first timestamp plus each `elapsed_ns`.

### `bench/paired.py`, pure

```python
@dataclass(frozen=True)
class Estimate:
    median: float
    low: float | None  # 95% bootstrap interval; None under MIN_BLOCKS blocks
    high: float | None
    prompts: int
    blocks: int


MIN_BLOCKS = 3
RESAMPLES = 10_000


def ratio(per_block: Mapping[int, Sequence[float]], seed: int) -> Estimate: ...
def floor(twin: Estimate) -> float | None: ...  # max(|low - 1|, |high - 1|)
def claimed(engine: Estimate, floor: float | None) -> bool: ...
```

`per_block` maps a block to its ratios: one per prompt for TTFT, TPOT and end-to-end time (engine over baseline, same prompt, same block), and one per block for output tokens per second. The bootstrap resamples whole blocks with a fixed seed and takes the median of the pooled ratios. `claimed` is true only when `|median - 1| > floor`. With no floor, from fewer than 3 blocks, no gap is claimed.

### `bench/analysis.py`

It gains `load_session` and the session summary. The CLI takes run and session directories alike and tells them apart by `group.json`.

**Gates**, each printed passed or failed with its numbers. One failure makes the session DISCARDED:

- **Conditions:** not smoke, the clock log parsed, and every probe on AC power with Low Power Mode off.
- **Liveness:** every planned block ran, no request failed, and no reply's prompt length differs from workload A's.
- **Memory:** every engine's largest `peak_bytes` is within its `footprint_gb`, and no probe's `swap_used_bytes` is above the first probe's.
- **Twin:** in Peak and in Sustained, each when it has a block, `|twin TPOT median ratio - 1|` is within its limit.

**Then**, Peak first, then Sustained:

- block counts per regime, Transition included;
- each engine's ratio to the baseline for TTFT, TPOT, end-to-end and output tokens per second: median, interval, prompts, blocks, and "within noise" when not claimed, then the twin's row, the floor;
- each engine's absolute p50, p95, and p99 with its interval, the requests counted, and the clock mix of its spans;
- each engine's largest gap between chunks, the share of stream time in gaps over 100 ms, and its peak footprint;
- at c1, how many prompts uLLM and the baseline answered with the same `completion_tokens` (0014's check).

`--compare` prints one table across sessions, Peak first, with each group's server lifecycle in its heading.

### `bench/conditions.py`

`violations(power, *, sudo, containers, powermetrics_running)` loses its `Plan` argument and gains a refusal when another `powermetrics` runs. `parse_power` reads `charging` from `pmset -g batt`, and `lid_open` comes from `AppleClamshellState` in `ioreg`. `PowerMetrics.stop` sends SIGTERM.

### Groups for M1-T7

Launch commands and request fields per target are those of decision 0014.

| Group | Servers | Concurrency | Baseline and twin | Other engines | Footprint limits (GB) |
|---|---|---|---|---|---|
| `m1-c1` | resident | 1 | `mlx-lm-sequential-no-cache` | `ullm`, `mlx-lm-sequential` | 3, 3, 4 |
| `m1-c8` | per chunk | 8 | `mlx-lm-batched-no-cache` | `ullm`, `mlx-lm-batched`, `mlx-lm-sequential-no-cache` | 11, 3, 14, 3 |

Both groups: seed 0, `twin_tpot_peak` 0.005, `twin_tpot_sustained` 0.03. Footprint limits are each engine's measured peak plus about a quarter, rounded up. The prompt cache on and off stays, as the author chose. With no repeated prompt, it measures the template prefix every prompt shares.

## Failure and operation

- **Runs twice:** `--out` must not exist, so a session never writes over another. Two sessions at once collide on a port or on the `powermetrics` refusal, and the second exits 2.
- **Crashes halfway:** `raw.jsonl` and `probes.jsonl` are flushed per line. `finally` stops servers and `powermetrics`, and `caffeinate` ends with the session. The analysis marks the session DISCARDED. If the session is SIGKILLed, servers and the root `powermetrics` survive. The next session then refuses on the port or the `powermetrics` check and names the fix.
- **Concurrent writers:** none. One task writes every file. Probes run between chunks.
- **Trust:** the group file is committed configuration, as trusted as the Makefile. Launch commands run without a shell. Servers bind 127.0.0.1. Replies go through `loadgen`'s reader, which turns an unreadable one into a failed request.
- **Seen working:** one line per chunk while it runs, `probes.jsonl` for memory, `summary.md` and the exit code after.

## Trade-offs accepted

- We accept a small Peak sample, the first minutes of each session, in exchange for no waiting. More Peak needs more sessions.
- We accept that Sustained absolute numbers vary between sessions. The paired ratio is the headline, and an absolute figure always carries its regime and clock mix.
- We accept two server lifecycles in one module, because c1 fits in memory and c8 does not. c1 keeps the shape the spikes proved. c8's twin shows whether per-chunk starts work.
- We accept that per-chunk servers start cold: each chunk begins after a server start, and its warm-up must absorb that.
- We accept closed-loop tails at c8: the last requests of a chunk run with fewer than 8 open. Every engine in a block has the same tail.
- We accept one more server per group, the twin, in exchange for a measured noise floor.
- We accept that batched mlx-lm holds up to 12 GB under its defaults. The table reports it and does not tune it away. If it pushes an idle Mac into swap, the c8 session is DISCARDED: a one-block smoke session on the idle Mac shows this before the full one.
- Transition blocks stay in `raw.jsonl` and are counted, but they are in no table.
- `bench.run` remains for `make bench-smoke`, a second way to measure. Moving `bench-smoke` onto a session would change the performance check of every pull request, so that is left for the author to choose.

## Alternatives considered

- **c8 servers loaded together, as the first version of this record had.** Lost: 29.3 GB of footprints on a 16 GB Mac, and the memory gate discards every session.
- **One server per chunk for every group.** It would make memory interference impossible by structure. Lost: c1 fits loaded together, and that is the shape the spikes proved.
- **Cap `mlx_lm.server`'s buffer cache at launch.** Lost: it is no longer the default batched mode that B4 asks for. It could come back as a separate, tuned row.
- **A twin gate on every metric.** Lost: it discards both spikes. TTFT and output tokens per second carry more noise than the TPOT floors the approach measured.
- **Keep `bench.run` per target and add an orchestrator.** Lost: the orchestrator and `bench.run` would both own server lifecycle, and pairing would join across directories.
- **Gate on the pressure label, or match clock mixes between chunks.** Lost on the spikes: the label read Heavy at full clock, and identical servers in one block differed by up to 0.46 in clock mix and still agreed.
- **Pin the clock with Low Power Mode.** It caps power, not the clock, and mlx-lm's TPOT moved from 25.8 to 22.4 ms inside one Low Power Mode phase.
- **Sort blocks by board temperature.** The sensors come from a private interface: a hint, never proof.

## Consequences

- Roadmap B5 changes: no cooldowns and no median of three. Engines take turns with a twin, and each figure states its regime. 3.6 asks for that edit in the same pull request.
- `bench.run` loses its B5 check and full-run mode, `bench/baselines.sh` and its test go, and claim 17 is rewritten for sessions.
- M2-T6 and M3-T9 reuse the session, the regimes and the paired ratios. A concurrency-16 group needs upstream candidate 4 handled first.
- Upstream candidate 5: batched `mlx_lm.server` holds 7 to 12 GB for a 0.6B model at concurrency 8.
- The five run directories in `bench/results/m1/` were measured by the plan this record replaces, and the uLLM ones before decision 0015. They are not results.
- Both spikes and the design checks are committed under `bench/results/m1-spikes/`, clock logs gzipped. A test runs `regimes.read_powermetrics` and `regimes.classify` on them and expects 3, 5, 4 and 7, 2, 7 outside Low Power Mode. That ties the code to the approach's own numbers. The figures in this record then cite committed results.

## Decisions by the author

Asked in this design session, answered by the author:

1. c8 runs one server per chunk.
2. The twin gate checks the TPOT median only.
3. Footprint limits per engine as in the groups table. Swap may not grow at all: the probes behind the first version's 256 MB ran with apps open, and sessions run on an idle Mac.
4. `m1-c8` includes `mlx-lm-sequential-no-cache`. Agent's default, not changed by the author.
5. c8 chunks hold 24 prompts and 8 warm-up requests. Agent's default, not changed by the author.
6. Settle 240 s before c1 and 600 s before c8. Agent's default, not changed by the author.
7. Commit both spikes and the design checks under `bench/results/m1-spikes/`.
8. The five old run directories are deleted only after the new sessions are committed, and only when the author says yes then.

## First step

The anchoring tests of `bench/regimes.py`: `capped` and `classify` against hand-made samples, including spike 1's block 5 (shares 0.52, 0.41, 0.71), which is Transition read each and Sustained read pooled.
