import { createRoot } from "react-dom/client";
import { sourceFromQuery } from "./config";
import { TelemetryStore } from "./store";
import { App } from "./ui/App";
import "./ui/tokens.css";
import "./ui/app.css";

const store = new TelemetryStore((callback) => requestAnimationFrame(callback));
const root = createRoot(document.getElementById("root")!);
try {
  const mode = sourceFromQuery(location.search, import.meta.env.BASE_URL, import.meta.env.MODE === "hosted");
  mode.source.start({ frame: (f) => store.ingest(f), status: (s) => store.setStatus(s) });
  root.render(<App store={store} mode={mode} />);
} catch (error) {
  store.setStatus({ kind: "error", message: (error as Error).message });
  root.render(<App store={store} />);
}
