// Generated from bench/control.schema.json by `make contracts`.
// Do not edit: change bench/control.py and regenerate.

export interface Control {
  [k: string]: unknown | undefined;
}
/**
 * This interface was referenced by `Control`'s JSON-Schema
 * via the `definition` "Closed".
 */
export interface Closed {
  concurrency: number;
  kind: "closed";
}
/**
 * This interface was referenced by `Control`'s JSON-Schema
 * via the `definition` "LoadSpec".
 */
export interface LoadSpec {
  arrival: Closed | Open;
  /**
   * replaces every request's max_tokens
   */
  max_tokens: number;
  /**
   * the workload's first N, in order
   */
  requests: number;
  /**
   * a name GET /workloads lists
   */
  workload: string;
}
/**
 * This interface was referenced by `Control`'s JSON-Schema
 * via the `definition` "Open".
 */
export interface Open {
  kind: "open";
  rate_per_s: number;
}
/**
 * This interface was referenced by `Control`'s JSON-Schema
 * via the `definition` "LoadStatus".
 */
export interface LoadStatus {
  errors: number;
  first_error: string | null;
  /**
   * replies without an error; counted when the run ends
   */
  ok: number;
  spec: LoadSpec | null;
  state: "idle" | "running" | "done" | "stopped" | "failed";
}
/**
 * This interface was referenced by `Control`'s JSON-Schema
 * via the `definition` "WorkloadInfo".
 */
export interface WorkloadInfo {
  name: string;
  requests: number;
}
