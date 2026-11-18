import type { TelemetryStore } from "../store";
import { Feed } from "./Feed";
import { Header } from "./Header";
import { Ticker } from "./Ticker";
import { useStore } from "./useStore";

export function App({ store, label }: { store: TelemetryStore; label: string }) {
  useStore(store);
  return (
    <div className="shell">
      <Header store={store} label={label} />
      <Ticker store={store} />
      <Feed store={store} />
    </div>
  );
}
