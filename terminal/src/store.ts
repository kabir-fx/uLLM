import type { Frame, Hello, KvSnapshot, Step } from "./contracts/telemetry";
import { Ring } from "./ring";
import { foldStep, type Second } from "./timeline";

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
  cachedTokens: number | null;
  firstTokenT: number | null;
  /** The end of the newest step that sampled a token for this request. */
  lastTokenT: number | null;
  endT: number | null;
  /** The engine's TPOT, known once the request finishes. */
  tpotMs: number | null;
  /** Blocks this request owns alone in the newest `kv.snapshot`; null before one, or once it ended. */
  blocks: number | null;
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
const STEPS = 1024;
const SECONDS = 600;
/** `BLOCK_OWNED_BASE` in `ullm/telemetry/events.py`: state byte 3 + i means owned by `owners[i]` alone. */
const OWNED_BASE = 3;

/**
 * Everything the terminal shows, folded from telemetry frames. Events never reach React
 * one by one (T1): subscribers hear at most once per `schedule` callback, an animation
 * frame in the browser, and then read the store directly.
 */
export class TelemetryStore {
  hello: Hello | null = null;
  status: SourceStatus = { kind: "connecting" };
  /** The newest frame's `t`: the ticker's windows end here, so replay speed changes nothing. */
  nowT = 0;
  missedFrames = 0;
  droppedEvents = 0;
  /** Tokens sampled by each step. */
  readonly sampled = new Ring<Sample>(STEP_SAMPLES);
  readonly ttfts = new Ring<Sample>(REQUEST_SAMPLES);
  readonly tpots = new Ring<Sample>(REQUEST_SAMPLES);
  readonly steps = new Ring<Step>(STEPS);
  readonly seconds = new Ring<Second>(SECONDS);
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

  get lastStep(): Step | null {
    return this.steps.newest() ?? null;
  }

  request(rid: string): RequestRow | undefined {
    return this.requests.get(rid);
  }

  /** Every request kept, oldest arrival first. */
  rows(): IterableIterator<RequestRow> {
    return this.requests.values();
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
          cachedTokens: null,
          firstTokenT: null,
          lastTokenT: null,
          endT: null,
          tpotMs: null,
          blocks: null,
        });
        return;
      case "req.admitted":
        this.update(event.rid, (r) => {
          r.state = "prefilling";
          r.cachedTokens = event.cached_tokens;
        });
        return;
      case "req.first_token":
        this.ttfts.push({ t: event.t, value: event.ttft_ms });
        this.update(event.rid, (r) => {
          r.ttftMs = event.ttft_ms;
          r.firstTokenT = event.t;
        });
        return;
      case "req.preempted":
        this.update(event.rid, (r) => (r.state = "preempted"));
        return;
      case "req.finished":
        if (event.timings.tpot_ms !== null) this.tpots.push({ t: event.t, value: event.timings.tpot_ms });
        this.update(event.rid, (r) => (r.tpotMs = event.timings.tpot_ms));
        this.end(event.rid, "finished", event.usage.completion_tokens, event.t);
        return;
      case "req.aborted":
        this.end(event.rid, "aborted", event.completion_tokens, event.t);
        return;
      case "step":
        this.applyStep(event);
        return;
      case "kv.snapshot":
        this.applySnapshot(event);
        return;
      default:
        // router.decision has no panel yet; unknown types are ignored (C2).
        return;
    }
  }

  private applyStep(step: Step): void {
    this.steps.push(step);
    foldStep(this.seconds, step);
    // One token per row: a prefill samples its first token, a decode row its next one.
    this.sampled.push({ t: step.t, value: step.rids.length });
    for (const rid of step.rids) {
      this.update(rid, (r) => {
        r.tokens += 1;
        r.state = "running";
        r.lastTokenT = step.t;
      });
    }
  }

  private applySnapshot(snapshot: KvSnapshot): void {
    const owned = new Array<number>(snapshot.owners.length).fill(0);
    for (const byte of atob(snapshot.states)) {
      const owner = byte.charCodeAt(0) - OWNED_BASE;
      if (owner >= 0) owned[owner]! += 1;
    }
    const index = new Map(snapshot.owners.map((rid, i) => [rid, i]));
    for (const row of this.requests.values()) {
      if (row.endT === null) row.blocks = owned[index.get(row.rid) ?? -1] ?? 0;
    }
  }

  private end(rid: string, state: "finished" | "aborted", tokens: number, t: number): void {
    const row = this.requests.get(rid);
    if (row === undefined) return;
    row.state = state;
    row.tokens = tokens;
    row.endT = t;
    row.blocks = null;
    this.ended.push(rid);
    if (this.ended.length > FEED_ROWS) this.requests.delete(this.ended.shift()!);
  }

  private update(rid: string, change: (row: RequestRow) => void): void {
    const row = this.requests.get(rid);
    if (row !== undefined) change(row);
  }

  private reset(): void {
    this.hello = null;
    this.steps.clear();
    this.seconds.clear();
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
