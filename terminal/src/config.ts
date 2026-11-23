import type { TelemetrySource } from "./source";
import { liveSource } from "./sources/live";
import { replaySource } from "./sources/replay";
import { loadTrace } from "./trace";

/** `?live=<url>` reads a stream; `?trace=<url>` (default: a bundled fixture) replays one, at `?speed=`. */
export function sourceFromQuery(query: string, base: string): TelemetrySource {
  const params = new URLSearchParams(query);
  const live = params.get("live");
  if (live !== null) return liveSource(live);
  const speed = Number(params.get("speed") ?? "1");
  if (!(speed > 0 && Number.isFinite(speed))) throw new Error(`?speed must be a positive number, not ${params.get("speed")}`);
  const trace = params.get("trace") ?? `${base}telemetry/steady-chat.jsonl.gz`;
  return replaySource(trace, () => loadTrace(trace), { speed });
}

export interface Controls {
  /** The engine server's origin, for the playground's chat completions. */
  api: string;
  /** The control process behind the launcher (decision 0024). */
  control: string;
}

/** Live mode only: `?live=`'s origin and `?control=` (default `http://127.0.0.1:8090`). */
export function controlsFromQuery(query: string): Controls | null {
  const params = new URLSearchParams(query);
  const live = params.get("live");
  if (live === null) return null;
  return { api: new URL(live).origin, control: params.get("control") ?? "http://127.0.0.1:8090" };
}
