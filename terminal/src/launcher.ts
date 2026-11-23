import type { LoadSpec, LoadStatus } from "./contracts/control";

export interface LaunchForm {
  workload: string;
  arrival: "closed" | "open";
  concurrency: number;
  ratePerS: number;
  maxTokens: number;
  requests: number;
}

export function launchSpec(form: LaunchForm): LoadSpec {
  return {
    workload: form.workload,
    arrival: form.arrival === "closed" ? { kind: "closed", concurrency: form.concurrency } : { kind: "open", rate_per_s: form.ratePerS },
    max_tokens: form.maxTokens,
    requests: form.requests,
  };
}

export function statusLine(status: LoadStatus): string {
  const { state, spec } = status;
  if (state === "failed") return `failed: ${status.first_error}`;
  if (state === "done") {
    const first = status.first_error === null ? "" : `, first: ${status.first_error}`;
    return `done: ${status.ok} ok, ${status.errors} errors${first}`;
  }
  if (spec === null) return state;
  const arrival = spec.arrival.kind === "closed" ? `closed loop of ${spec.arrival.concurrency}` : `open loop at ${spec.arrival.rate_per_s}/s`;
  return `${state}: ${spec.requests} of ${spec.workload}, ${arrival}, max_tokens ${spec.max_tokens}`;
}
