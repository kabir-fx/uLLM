import type { Mode } from "../config";
import type { TelemetryStore } from "../store";
import { Feed } from "./Feed";
import { Header } from "./Header";
import { Launcher } from "./Launcher";
import { BatchStrip, StepChart } from "./Panels";
import { Playground } from "./Playground";
import { ReplayPanel } from "./ReplayPanel";
import { RequestsTable } from "./RequestsTable";
import { Ticker } from "./Ticker";
import { useStore } from "./useStore";

/** Live mode has the launcher and the playground; replay mode has the replay panel (decision 0026). */
export function App({ store, mode = null }: { store: TelemetryStore; mode?: Mode | null }) {
  useStore(store);
  return (
    <div className={mode ? "shell with-side" : "shell"}>
      <Header store={store} mode={mode} />
      <Ticker store={store} />
      <StepChart store={store} />
      <BatchStrip store={store} />
      <Feed store={store} />
      <RequestsTable store={store} />
      {mode && (
        <aside className="side">
          {mode.kind === "live" ? (
            <>
              <Launcher control={mode.controls.control} />
              <Playground api={mode.controls.api} />
            </>
          ) : (
            <ReplayPanel source={mode.source} run={mode.run} />
          )}
        </aside>
      )}
    </div>
  );
}
