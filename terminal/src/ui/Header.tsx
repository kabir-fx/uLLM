import type { TelemetryStore } from "../store";

export function Header({ store, label }: { store: TelemetryStore; label: string }) {
  const { hello, status, missedFrames, droppedEvents } = store;
  // Decision 0003: fixture and sim data are labelled as simulated, always. Until a hello
  // says otherwise, the data is not known to come from the engine.
  const simulated = hello?.source !== "engine";
  return (
    <header className="header">
      <span className="brand">uLLM</span>
      {simulated && (
        <span className="badge badge-sim" title={`source: ${hello?.source ?? "unknown"}`}>
          SIMULATED
        </span>
      )}
      <span className="field">{hello?.model ?? "waiting for hello"}</span>
      {hello && <span className="field dim">{hello.replica}</span>}
      {hello?.machine && (
        <span className="field dim">
          {hello.machine.chip} · {hello.machine.memory_gb} GB
        </span>
      )}
      {hello?.commit && <span className="field dim">{hello.commit.slice(0, 7)}</span>}
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
