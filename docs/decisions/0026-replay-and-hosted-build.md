# 0026: Replay controls, the run picker and the hosted build

**Status:** accepted · **Task:** M2-T9 (T4, DEP3)

## Context

T4 asks for replay mode with play, pause, speed and a scrubber, and a run picker in place of the launcher. It also asks for a static build bundled with recorded runs, published at a public URL (DEP3), always labelled as a replay, with the machine and commit of the recording. DEP3: publishing needs the author's approval each time.

Facts this design rests on:

- `replaySource(label, load, { speed })` plays forward only, at a speed fixed by `?speed=` (`terminal/src/sources/replay.ts`, decision 0017).
- `TelemetryStore.ingest` resets the store on a frame whose first event is `hello` (`terminal/src/store.ts`). C2 starts every recording with `hello` (decision 0023), so sending a trace again from frame 0 rebuilds the store from nothing.
- The one recording of the real engine is `bench/results/m2/telemetry/ullm-on.jsonl.gz`: 19,843 bytes, 287 frames, about 25 s of workload A at concurrency 8, `--max-num-seqs 8`. Its `hello` has `source: "engine"` but `commit: null` and `machine: null`, because `ullm serve` fills neither. The session's `env.json` records the machine (Apple M4, 16 GB) and the commit (`fab197c`, dirty).
- GitHub Pages is not enabled on `kabir-fx/uLLM` (`gh api repos/kabir-fx/uLLM/pages`: 404). A project site is served under `/uLLM/`.
- `actions/upload-pages-artifact` v5.0.0 (`fc324d3`) takes a `path`. `actions/deploy-pages` v5.0.1 (`368f825`) deploys the artifact `github-pages` (their `action.yml` at those commits).
- `parseTrace` reads gzip and plain JSONL alike, so a host that does not set `Content-Encoding: gzip` on `.gz` files still works (decision 0017).

## Decision

**Replay controls.** `replaySource` returns a `ReplaySource`, a `TelemetrySource` with a controller:

```ts
export interface ReplaySource extends TelemetrySource {
  readonly playing: boolean;
  readonly speed: number;
  /** Trace length in ms, from the first frame to the last; 0 until the trace loads. */
  readonly duration: number;
  /** Trace ms from the first frame. */
  position(): number;
  play(): void; // at the end, starts again from 0
  pause(): void;
  setSpeed(speed: number): void;
  seek(ms: number): void; // clamped to [0, duration]
}
```

The clock is a base position and the wall time it was taken at: `position = base + (now − wall) × speed` while playing, `base` while paused. Pause, a new speed and a seek each take a new base, so no change makes the position jump. A seek forward sends the frames between the old and the new position at once. A seek back sends the trace again from frame 0 up to the new position, and the store rebuilds itself from the `hello`. Every frame is still sent at its own time from one base, so timer lateness never accumulates. `SourceStatus` gains `paused`. A pause pressed while the trace loads holds: the replay shows its first frame and waits.

**Runs.** `terminal/src/runs.ts` lists the bundled runs:

```ts
export interface Recording { chip: string; memoryGb: number; commit: string; dirty: boolean }
export interface Run { id: string; title: string; trace: string; recording: Recording | null }
export const RUNS: readonly Run[];
```

`recording` is null for synthetic fixtures. They already say SIMULATED through their `hello`. The engine recording is copied to `terminal/fixtures/runs/m2-telemetry-c8.jsonl.gz` so the build serves it beside the fixtures. `runs.test.ts` fails when the copy differs from `bench/results/m2/telemetry/ullm-on.jsonl.gz` byte for byte, or when a recording's machine and commit differ from its session's `env.json`, or when a run's trace is missing.

**Modes.** `sourceFromQuery(query, env)` takes Vite's `import.meta.env` (`MODE`, `BASE_URL`) whole, so the choice of the hosted mode is inside the tested function. It returns one of two shapes, so a replay has no launcher and a live stream has no replay bar:

```ts
export type Mode =
  | { kind: "live"; source: TelemetrySource; controls: Controls }
  | { kind: "replay"; source: ReplaySource; run: Run | null };
```

`?run=<id>` picks a bundled run, the first by default. `?trace=` still replays any trace locally, with `run: null`. `?speed=` sets the first speed. `controlsFromQuery` is folded into it. The hosted build (`vite build --mode hosted`, so `MODE` is `hosted`) ignores `?live=` and `?trace=`: the public page plays only its own runs.

**Terminal.** In replay mode the side column holds the replay panel in place of the launcher and the playground: a run picker, play and pause, a speed (0.25 to 8), and a scrubber with the position, with any step, so it reaches the end of a trace of any length. Picking a run loads `?run=<id>`. The header shows a REPLAY badge in replay mode, and the run's machine and commit, marked dirty when they were.

**Hosted build.** `vite build --mode hosted` sets `base: "./"`, so the page works under any path, and writes `terminal/dist-hosted/`. `make terminal-hosted` builds it, and `make check-ts` builds it too, so CI keeps it working. `.github/workflows/pages.yml` runs only on `workflow_dispatch`: it builds with `make terminal-hosted` and deploys with the two Pages actions, pinned by commit. The author enables Pages once (Settings, Pages, source: GitHub Actions) and publishes each time with `gh workflow run pages.yml`. Nothing publishes on a push or a merge.

## Alternatives considered

- **`ullm serve` fills `hello.commit` and `hello.machine`.** That fixes the cause for future recordings, but it is Lane A's code and needs a new real recording. The author chose the manifest. The null fields are reported to Lane A.
- **A script that generates `runs.ts` from `env.json`.** A test that compares them gives the same guarantee with less machinery.
- **Importing the recording from `bench/results/` with Vite's `?url`.** It keeps one copy, but the dev server would need `server.fs.allow` widened to the repo root. A 20 KB copy, held equal by a test, is plainer.
- **A seek that rebuilds the store from a snapshot.** Sending the trace again from `hello` reuses the store's own reset and costs one pass over at most a few thousand frames.
- **Publishing on every merge to `main`.** DEP3 asks for the author's approval for each publish.
- **A hosted build that keeps `?live=` and `?trace=`.** The public page could then show data that is not a bundled replay.

## Consequences

- M2-T10's Playwright smoke test can drive the hosted build with a bundled run.
- M3-T8 adds runs for workloads B and C by adding entries to `RUNS`, each with its session's `env.json`.
- The hosted URL, `https://kabir-fx.github.io/uLLM/`, exists only after the author enables Pages and runs the workflow.
- Lane A: `ullm serve` sends `commit: null` and `machine: null` in `hello`. A recording made outside a bench session has no machine or commit to show.
