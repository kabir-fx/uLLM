import type { TelemetrySource } from "./source";
import { RUNS, type Run } from "./runs";
import { liveSource } from "./sources/live";
import { replaySource, type ReplaySource } from "./sources/replay";
import { loadTrace } from "./trace";

export interface Controls {
  /** The engine server's origin, for the playground's chat completions. */
  api: string;
  /** The control process behind the launcher (decision 0025). */
  control: string;
}

/** A live stream has the launcher and the playground; a replay has its controls (decision 0026). */
export type Mode =
  | { kind: "live"; source: TelemetrySource; controls: Controls }
  | { kind: "replay"; source: ReplaySource; run: Run | null };

/** The fields of Vite's `import.meta.env` that pick the mode; `MODE` is `hosted` in the public build. */
export interface BuildEnv {
  MODE: string;
  BASE_URL: string;
}

/**
 * `?live=<url>` reads a stream, with `?control=` (default `http://127.0.0.1:8090`).
 * Otherwise `?run=<id>` replays a bundled run (default: the first), or `?trace=<url>` any
 * trace, at `?speed=`. The hosted build ignores `?live=` and `?trace=`.
 */
export function sourceFromQuery(query: string, { MODE, BASE_URL: base }: BuildEnv): Mode {
  const hosted = MODE === "hosted";
  const params = new URLSearchParams(query);
  const live = hosted ? null : params.get("live");
  if (live !== null) {
    const controls = { api: new URL(live).origin, control: params.get("control") ?? "http://127.0.0.1:8090" };
    return { kind: "live", source: liveSource(live), controls };
  }
  const speed = Number(params.get("speed") ?? "1");
  if (!(speed > 0 && Number.isFinite(speed))) throw new Error(`?speed must be a positive number, not ${params.get("speed")}`);
  const trace = hosted ? null : params.get("trace");
  if (trace !== null) return { kind: "replay", source: replaySource(trace, () => loadTrace(trace), { speed }), run: null };
  const id = params.get("run") ?? RUNS[0]!.id;
  const run = RUNS.find((r) => r.id === id);
  if (!run) throw new Error(`no bundled run ${id}`);
  const url = `${base}${run.trace}`;
  return { kind: "replay", source: replaySource(url, () => loadTrace(url), { speed }), run };
}
