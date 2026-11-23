import type { Mode } from "../config";
import type { TelemetryStore } from "../store";

export function Header({ store, mode }: { store: TelemetryStore; mode: Mode | null }) {
  const { hello, status, missedFrames, droppedEvents } = store;
  // Decision 0003: fixture and sim data are labelled as simulated, always. Until a hello
  // says otherwise, the data is not known to come from the engine.
  const simulated = hello?.source !== "engine";
  const replay = mode?.kind === "replay";
  // A bundled run names its recording's machine and commit; engine hellos do not yet (decision 0026).
  const recording = mode?.kind === "replay" ? mode.run?.recording : null;
  const machine = recording ? { chip: recording.chip, memory_gb: recording.memoryGb } : hello?.machine;
  const commit = recording?.commit ?? hello?.commit;
  const label = mode?.source.label ?? "";
  return (
    <header className="header">
      <span className="brand">uLLM</span>
      {replay && (
        <span className="badge badge-replay" title="a recorded run, not live">
          REPLAY
        </span>
      )}
      {simulated && (
        <span className="badge badge-sim" title={`source: ${hello?.source ?? "unknown"}`}>
          SIMULATED
        </span>
      )}
      <span className="field">{hello?.model ?? "waiting for hello"}</span>
      {hello && <span className="field dim">{hello.replica}</span>}
      {machine && (
        <span className="field dim">
          {machine.chip} · {machine.memory_gb} GB
        </span>
      )}
      {commit && (
        <span className="field dim" title={commit}>
          {commit.slice(0, 7)}
          {recording?.dirty && " dirty"}
        </span>
      )}
      <span className="spacer" />
      {missedFrames > 0 && <span className="warn">{missedFrames} frames missed</span>}
      {droppedEvents > 0 && <span className="warn">{droppedEvents} events dropped</span>}
      <span className="field dim" title={label}>
        {label.split("/").at(-1)}
      </span>
      <span className={`status status-${status.kind}`}>
        {status.kind === "error" ? `error: ${status.message}` : status.kind}
      </span>
    </header>
  );
}
