import type { LoadSpec, LoadStatus, WorkloadInfo } from "./contracts/control";

/** The control process behind the launcher (`bench.control`, decision 0025). */
export interface ControlClient {
  workloads(): Promise<WorkloadInfo[]>;
  status(): Promise<LoadStatus>;
  /** Rejects with the server's reason: 409 while a run is open, 400 for a bad spec. */
  start(spec: LoadSpec): Promise<LoadStatus>;
  stop(): Promise<LoadStatus>;
}

export function controlClient(base: string): ControlClient {
  const root = base.replace(/\/+$/, "");
  async function call<T>(method: string, path: string, body?: LoadSpec): Promise<T> {
    let response: Response;
    try {
      response = await fetch(`${root}${path}`, {
        method,
        ...(body && { headers: { "content-type": "application/json" }, body: JSON.stringify(body) }),
      });
    } catch {
      throw new Error(`cannot reach the control process at ${root}`);
    }
    const payload: unknown = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = (payload as { detail?: unknown } | null)?.detail;
      throw new Error(`${response.status}: ${typeof detail === "string" ? detail : response.statusText}`);
    }
    return payload as T;
  }
  return {
    workloads: () => call("GET", "/workloads"),
    status: () => call("GET", "/load"),
    start: (spec) => call("POST", "/load", spec),
    stop: () => call("DELETE", "/load"),
  };
}
