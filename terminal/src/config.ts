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
