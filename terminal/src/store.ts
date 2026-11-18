import type { Frame, Hello, Step } from "./contracts/telemetry";
import { Ring } from "./ring";

export type RequestState = "waiting" | "prefilling" | "running" | "preempted" | "finished" | "aborted";

export interface RequestRow {
  rid: string;
  state: RequestState;
  arrivedT: number;
  promptTokens: number;
  maxTokens: number;
  /** Sampled so far, counted from steps; the engine's own count once the request ends. */
  tokens: number;
  ttftMs: number | null;
}

export type SourceStatus =
  | { kind: "connecting" }
  | { kind: "open" }
  | { kind: "ended" }
  | { kind: "error"; message: string };

/** A value at a trace time, for the ticker's windows. */
export interface Sample {
  t: number;
  value: number;
}

/** Requests that ended and stay in the feed. Requests in flight are always kept. */
export const FEED_ROWS = 200;
const STEP_SAMPLES = 4096;
const REQUEST_SAMPLES = 4096;

/**
 * Everything the terminal shows, folded from telemetry frames. Events never reach React
 * one by one (T1): subscribers hear at most once per `schedule` callback, an animation
 * frame in the browser, and then read the store directly.
 */
export class TelemetryStore {
  hello: Hello | null = null;
  status: SourceStatus = { kind: "connecting" };
  lastStep: Step | null = null;
  /** The newest frame's `t`: the ticker's windows end here, so replay speed changes nothing. */
  nowT = 0;
  missedFrames = 0;
  droppedEvents = 0;
  /** Tokens sampled by each step. */
  readonly sampled = new Ring<Sample>(STEP_SAMPLES);
  readonly ttfts = new Ring<Sample>(REQUEST_SAMPLES);
  readonly tpots = new Ring<Sample>(REQUEST_SAMPLES);
  version = 0;

  private readonly requests = new Map<string, RequestRow>();
  private readonly ended: string[] = [];
  private lastSeq: number | null = null;
  private readonly listeners = new Set<() => void>();
  private notifyPending = false;

  constructor(private readonly schedule: (callback: () => void) => void) {}

  ingest(frame: Frame): void {
    // The contract sends hello first on every connection, and seq restarts with it.
    if (frame.events[0]?.type === "hello") this.reset();
    if (this.lastSeq !== null && frame.seq > this.lastSeq + 1) {
      this.missedFrames += frame.seq - this.lastSeq - 1;
    }
    this.lastSeq = frame.seq;
    this.droppedEvents += frame.dropped;
    this.nowT = Math.max(this.nowT, frame.t);
    for (const event of frame.events) this.apply(event);
    this.changed();
  }

  setStatus(status: SourceStatus): void {
    this.status = status;
    this.changed();
  }

  request(rid: string): RequestRow | undefined {
    return this.requests.get(rid);
  }

  /** Newest arrival first, at most `FEED_ROWS` rows. */
  feed(): RequestRow[] {
    return [...this.requests.values()].reverse().slice(0, FEED_ROWS);
  }

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  getVersion = (): number => this.version;

  private apply(event: Frame["events"][number]): void {
    switch (event.type) {
      case "hello":
        this.hello = event;
        return;
      case "req.arrived":
        this.requests.set(event.rid, {
          rid: event.rid,
          state: "waiting",
          arrivedT: event.t,
          promptTokens: event.prompt_tokens,
          maxTokens: event.max_tokens,
          tokens: 0,
          ttftMs: null,
        });
        return;
      case "req.admitted":
        this.update(event.rid, (r) => (r.state = "prefilling"));
        return;
      case "req.first_token":
        this.ttfts.push({ t: event.t, value: event.ttft_ms });
        this.update(event.rid, (r) => (r.ttftMs = event.ttft_ms));
        return;
      case "req.preempted":
        this.update(event.rid, (r) => (r.state = "preempted"));
        return;
      case "req.finished":
        if (event.timings.tpot_ms !== null) this.tpots.push({ t: event.t, value: event.timings.tpot_ms });
        this.end(event.rid, "finished", event.usage.completion_tokens);
        return;
      case "req.aborted":
        this.end(event.rid, "aborted", event.completion_tokens);
        return;
      case "step":
        this.applyStep(event);
        return;
      default:
        // kv.snapshot and router.decision have no panel yet; unknown types are ignored (C2).
        return;
    }
  }

  private applyStep(step: Step): void {
    this.lastStep = step;
    // One token per row: a prefill samples its first token, a decode row its next one.
    this.sampled.push({ t: step.t, value: step.rids.length });
    for (const rid of step.rids) {
      this.update(rid, (r) => {
        r.tokens += 1;
        r.state = "running";
      });
    }
  }

  private end(rid: string, state: "finished" | "aborted", tokens: number): void {
    const row = this.requests.get(rid);
    if (row === undefined) return;
    row.state = state;
    row.tokens = tokens;
    this.ended.push(rid);
    if (this.ended.length > FEED_ROWS) this.requests.delete(this.ended.shift()!);
  }

  private update(rid: string, change: (row: RequestRow) => void): void {
    const row = this.requests.get(rid);
    if (row !== undefined) change(row);
  }

  private reset(): void {
    this.hello = null;
    this.lastStep = null;
    this.nowT = 0;
    this.missedFrames = 0;
    this.droppedEvents = 0;
    this.sampled.clear();
    this.ttfts.clear();
    this.tpots.clear();
    this.requests.clear();
    this.ended.length = 0;
    this.lastSeq = null;
  }

  private changed(): void {
    this.version++;
    if (this.notifyPending) return;
    this.notifyPending = true;
    this.schedule(() => {
      this.notifyPending = false;
      for (const listener of this.listeners) listener();
    });
  }
}
