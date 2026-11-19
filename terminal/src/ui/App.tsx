import type { TelemetryStore } from "../store";
import { Feed } from "./Feed";
import { Header } from "./Header";
import { BatchStrip, StepChart } from "./Panels";
import { RequestsTable } from "./RequestsTable";
import { Ticker } from "./Ticker";
import { useStore } from "./useStore";

export function App({ store, label }: { store: TelemetryStore; label: string }) {
  useStore(store);
  return (
    <div className="shell">
      <Header store={store} label={label} />
      <Ticker store={store} />
      <StepChart store={store} />
      <BatchStrip store={store} />
      <Feed store={store} />
      <RequestsTable store={store} />
    </div>
  );
}
