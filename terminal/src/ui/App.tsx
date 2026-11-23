import type { Controls } from "../config";
import type { TelemetryStore } from "../store";
import { Feed } from "./Feed";
import { Header } from "./Header";
import { Launcher } from "./Launcher";
import { BatchStrip, StepChart } from "./Panels";
import { Playground } from "./Playground";
import { RequestsTable } from "./RequestsTable";
import { Ticker } from "./Ticker";
import { useStore } from "./useStore";

/** `controls` is live mode's: the launcher and the playground. In replay mode there are none. */
export function App({ store, label, controls }: { store: TelemetryStore; label: string; controls?: Controls | null }) {
  useStore(store);
  return (
    <div className={controls ? "shell with-side" : "shell"}>
      <Header store={store} label={label} />
      <Ticker store={store} />
      <StepChart store={store} />
      <BatchStrip store={store} />
      <Feed store={store} />
      <RequestsTable store={store} />
      {controls && (
        <aside className="side">
          <Launcher control={controls.control} />
          <Playground api={controls.api} />
        </aside>
      )}
    </div>
  );
}
