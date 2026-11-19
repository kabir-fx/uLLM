import type { Step } from "./contracts/telemetry";
import type { Ring } from "./ring";

export interface Split {
  prefill: number;
  decode: number;
}

/** The steps that ended in one second of trace time: a candle of step latency, and token counts. */
export interface Second {
  second: number;
  open: number;
  high: number;
  low: number;
  close: number;
  steps: number;
  /** Tokens run through the model: the prompt for a prefill, one per row for a decode. */
  processed: Split;
  /** Tokens sampled: one per row. */
  sampled: Split;
}

export function foldStep(seconds: Ring<Second>, step: Step): void {
  const second = Math.floor(step.t / 1_000);
  let bucket = seconds.newest();
  if (bucket === undefined || bucket.second !== second) {
    bucket = {
      second,
      open: step.ms,
      high: step.ms,
      low: step.ms,
      close: step.ms,
      steps: 0,
      processed: { prefill: 0, decode: 0 },
      sampled: { prefill: 0, decode: 0 },
    };
    seconds.push(bucket);
  }
  bucket.high = Math.max(bucket.high, step.ms);
  bucket.low = Math.min(bucket.low, step.ms);
  bucket.close = step.ms;
  bucket.steps += 1;
  bucket.processed[step.kind] += step.tokens;
  bucket.sampled[step.kind] += step.rids.length;
}
