/** Where and from what a real recording was made, from its bench session's `env.json`. */
export interface Recording {
  /** The recording's path in the repo; the bundled trace is a copy of it. */
  from: string;
  chip: string;
  memoryGb: number;
  commit: string;
  /** The working tree had uncommitted changes when it was recorded. */
  dirty: boolean;
}

export interface Run {
  id: string;
  title: string;
  /** The trace's path under the build's base. */
  trace: string;
  /** Null for synthetic fixtures; their hello labels them simulated (decision 0003). */
  recording: Recording | null;
}

/** The runs the replay picker offers; the first is the default (decision 0026). */
export const RUNS: readonly Run[] = [
  {
    id: "m2-telemetry-c8",
    title: "workload A · concurrency 8 · Qwen3-0.6B bf16",
    trace: "runs/m2-telemetry-c8.jsonl.gz",
    recording: {
      from: "bench/results/m2/telemetry/ullm-on.jsonl.gz",
      chip: "Apple M4",
      memoryGb: 16,
      commit: "fab197c68f596460cb5383e8c4dccc66a0bb8c6d",
      dirty: true,
    },
  },
  { id: "steady-chat", title: "steady chat (synthetic)", trace: "telemetry/steady-chat.jsonl.gz", recording: null },
  {
    id: "prefix-pressure",
    title: "prefix pressure (synthetic)",
    trace: "telemetry/prefix-pressure.jsonl.gz",
    recording: null,
  },
];
