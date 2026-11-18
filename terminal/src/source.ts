import type { Frame } from "./contracts/telemetry";
import type { SourceStatus } from "./store";

export interface Sink {
  frame(frame: Frame): void;
  status(status: SourceStatus): void;
}

/** Where frames come from: a recorded trace or a live stream (T1). The rest of the app cannot tell. */
export interface TelemetrySource {
  readonly label: string;
  /** Starts sending to `sink`. The returned function stops it; nothing reaches `sink` after. */
  start(sink: Sink): () => void;
}
