import type { Frame, Hello, Step } from "../contracts/telemetry";

export type Event = Frame["events"][number];

export function hello(over: Partial<Hello> = {}): Hello {
  return {
    type: "hello",
    t: 0,
    replica: "r1",
    model: "m",
    source: "engine",
    version: "test",
    commit: null,
    epoch_unix_ms: 0,
    limits: { max_num_seqs: 8, max_prefill_tokens: 8192, max_model_len: 16384 },
    kv: { block_size: null, num_blocks: null, capacity_tokens: 1000 },
    machine: null,
    ...over,
  };
}

export function step(t: number, kind: Step["kind"], rids: string[], over: Partial<Step> = {}): Step {
  return {
    type: "step",
    t,
    n: 0,
    kind,
    rids,
    tokens: kind === "decode" ? rids.length : 100,
    ms: 10,
    waiting: 0,
    running: rids.length,
    kv_tokens: 0,
    ...over,
  };
}

export function arrived(t: number, rid: string): Event {
  return { type: "req.arrived", t, rid, prompt_tokens: 100, max_tokens: 50, tenant: null };
}

export function admitted(t: number, rid: string): Event {
  return { type: "req.admitted", t, rid, queue_ms: 0, cached_tokens: 0 };
}

export function firstToken(t: number, rid: string, ttft_ms: number): Event {
  return { type: "req.first_token", t, rid, ttft_ms };
}

export function finished(t: number, rid: string, tokens: number, tpot_ms: number | null): Event {
  return {
    type: "req.finished",
    t,
    rid,
    reason: "stop",
    usage: { prompt_tokens: 100, completion_tokens: tokens, cached_tokens: 0 },
    timings: { queue_ms: 0, ttft_ms: 1, tpot_ms, e2e_ms: 1 },
  };
}

/** Frames numbered from `seq`, each at the time of its last event. */
export function frames(groups: Event[][], seq = 0): Frame[] {
  return groups.map((events, i) => ({
    v: 1,
    replica: "r1",
    seq: seq + i,
    t: Math.max(0, ...events.map((e) => e.t)),
    dropped: 0,
    events,
  }));
}
