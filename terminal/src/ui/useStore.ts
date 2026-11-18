import { useSyncExternalStore } from "react";
import type { TelemetryStore } from "../store";

/** Re-renders when the store has changed, at most once per store flush. */
export function useStore(store: TelemetryStore): void {
  useSyncExternalStore(store.subscribe, store.getVersion, store.getVersion);
}
