/* eslint-disable */
// Generated from src/ullm/telemetry/schema/telemetry.v1.json by `make contracts`.
// Do not edit: change src/ullm/telemetry/events.py and regenerate.

/**
 * One flush of the telemetry stream: an SSE ``data:`` line, or one line of a trace file.
 */
export interface Frame {
  dropped: number;
  events: (
    | Hello
    | ReqArrived
    | ReqAdmitted
    | ReqFirstToken
    | ReqFinished
    | ReqAborted
    | ReqPreempted
    | Step
    | KvSnapshot
    | RouterDecision
  )[];
  replica: string;
  seq: number;
  t: number;
  v: 1;
}
/**
 * First event on every connection and in every recording.
 *
 * This interface was referenced by `Frame`'s JSON-Schema
 * via the `definition` "Hello".
 */
export interface Hello {
  commit: string | null;
  epoch_unix_ms: number;
  kv: KvInfo;
  limits: Limits;
  machine: Machine | null;
  model: string;
  replica: string;
  source: "engine" | "sim" | "fixture";
  t: number;
  type: "hello";
  version: string;
}
/**
 * KV capacity. Block fields stay null until blocks exist (M3).
 *
 * This interface was referenced by `Frame`'s JSON-Schema
 * via the `definition` "KvInfo".
 */
export interface KvInfo {
  block_size: number | null;
  capacity_tokens: number | null;
  num_blocks: number | null;
}
/**
 * This interface was referenced by `Frame`'s JSON-Schema
 * via the `definition` "Limits".
 */
export interface Limits {
  max_model_len: number;
  max_num_seqs: number;
  max_prefill_tokens: number;
}
/**
 * This interface was referenced by `Frame`'s JSON-Schema
 * via the `definition` "Machine".
 */
export interface Machine {
  chip: string;
  memory_gb: number;
}
/**
 * This interface was referenced by `Frame`'s JSON-Schema
 * via the `definition` "ReqArrived".
 */
export interface ReqArrived {
  max_tokens: number;
  prompt_tokens: number;
  rid: string;
  t: number;
  tenant: string | null;
  type: "req.arrived";
}
/**
 * This interface was referenced by `Frame`'s JSON-Schema
 * via the `definition` "ReqAdmitted".
 */
export interface ReqAdmitted {
  cached_tokens: number;
  queue_ms: number;
  rid: string;
  t: number;
  type: "req.admitted";
}
/**
 * This interface was referenced by `Frame`'s JSON-Schema
 * via the `definition` "ReqFirstToken".
 */
export interface ReqFirstToken {
  rid: string;
  t: number;
  ttft_ms: number;
  type: "req.first_token";
}
/**
 * This interface was referenced by `Frame`'s JSON-Schema
 * via the `definition` "ReqFinished".
 */
export interface ReqFinished {
  reason: "stop" | "length" | "tool_calls";
  rid: string;
  t: number;
  timings: Timings;
  type: "req.finished";
  usage: TokenUsage;
}
/**
 * This interface was referenced by `Frame`'s JSON-Schema
 * via the `definition` "Timings".
 */
export interface Timings {
  e2e_ms: number;
  queue_ms: number;
  tpot_ms: number | null;
  ttft_ms: number;
}
/**
 * This interface was referenced by `Frame`'s JSON-Schema
 * via the `definition` "TokenUsage".
 */
export interface TokenUsage {
  cached_tokens: number;
  completion_tokens: number;
  prompt_tokens: number;
}
/**
 * This interface was referenced by `Frame`'s JSON-Schema
 * via the `definition` "ReqAborted".
 */
export interface ReqAborted {
  completion_tokens: number;
  reason: "client_disconnect" | "shutdown" | "error";
  rid: string;
  t: number;
  type: "req.aborted";
}
/**
 * This interface was referenced by `Frame`'s JSON-Schema
 * via the `definition` "ReqPreempted".
 */
export interface ReqPreempted {
  blocks_freed: number;
  completion_tokens: number;
  rid: string;
  t: number;
  type: "req.preempted";
}
/**
 * One engine step. ``t`` is its end, after ``mx.eval``; it started at ``t - ms``.
 *
 * This interface was referenced by `Frame`'s JSON-Schema
 * via the `definition` "Step".
 */
export interface Step {
  kind: "prefill" | "decode";
  kv_tokens: number;
  ms: number;
  n: number;
  rids: string[];
  running: number;
  t: number;
  tokens: number;
  type: "step";
  waiting: number;
}
/**
 * The state of every KV block, compactly (from M3).
 *
 * This interface was referenced by `Frame`'s JSON-Schema
 * via the `definition` "KvSnapshot".
 */
export interface KvSnapshot {
  num_blocks: number;
  /**
   * @maxItems 253
   */
  owners: string[];
  states: string;
  t: number;
  type: "kv.snapshot";
}
/**
 * Where the router sent a request, and why (from M5).
 *
 * This interface was referenced by `Frame`'s JSON-Schema
 * via the `definition` "RouterDecision".
 */
export interface RouterDecision {
  kind: "prefix" | "least_load" | "round_robin";
  matched_messages: number;
  replica: string;
  retry: boolean;
  rid: string;
  t: number;
  type: "router.decision";
}
