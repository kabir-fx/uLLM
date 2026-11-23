import { useReducer } from "react";
import { RUNS, type Run } from "../runs";
import type { ReplaySource } from "../sources/replay";

const SPEEDS = [0.25, 0.5, 1, 2, 4, 8];

/** m:ss.s */
function clock(ms: number): string {
  const s = ms / 1000;
  return `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;
}

/** The run picker and the transport, in place of the launcher (T4). Picking a run loads its page. */
export function ReplayPanel({ source, run }: { source: ReplaySource; run: Run | null }) {
  // Play, pause and speed change the source, not the store; this re-renders for them.
  const [, rerender] = useReducer((n: number) => n + 1, 0);
  const act = (change: () => void) => {
    change();
    rerender();
  };
  const at = source.position();

  return (
    <section className="panel side-panel replay" aria-label="replay">
      <h2>
        replay <span className="dim">recorded, not live</span>
      </h2>
      <div className="controls">
        <label>
          run
          <select
            name="run"
            value={run?.id ?? ""}
            onChange={(e) => {
              const params = new URLSearchParams(location.search);
              params.delete("trace");
              params.set("run", e.target.value);
              location.search = params.toString();
            }}
          >
            {!run && <option value="">{source.label.split("/").at(-1)}</option>}
            {RUNS.map((r) => (
              <option key={r.id} value={r.id}>
                {r.title}
              </option>
            ))}
          </select>
        </label>
        <input
          type="range"
          name="position"
          aria-label="position"
          min={0}
          max={source.duration}
          step={100}
          value={at}
          onChange={(e) => act(() => source.seek(e.target.valueAsNumber))}
        />
        <div className="buttons">
          <button type="button" onClick={() => act(() => (source.playing ? source.pause() : source.play()))}>
            {source.playing ? "pause" : "play"}
          </button>
          <select
            name="speed"
            aria-label="speed"
            value={source.speed}
            onChange={(e) => act(() => source.setSpeed(Number(e.target.value)))}
          >
            {[...new Set([...SPEEDS, source.speed])].sort((a, b) => a - b).map((s) => (
              <option key={s} value={s}>
                {s}×
              </option>
            ))}
          </select>
        </div>
      </div>
      <p className="line">
        {clock(at)} / {clock(source.duration)}
      </p>
    </section>
  );
}
