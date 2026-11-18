import type { Sample, TelemetryStore } from "./store";
import type { Ring } from "./ring";

export const RATE_WINDOW_MS = 1_000;
export const LATENCY_WINDOW_MS = 60_000;

export interface Ticker {
  tokensPerSecond: number;
  ttft: { p50: number; p99: number; n: number } | null;
  tpot: { p50: number; n: number } | null;
  batch: number | null;
  batchCap: number | null;
  queue: number | null;
  kvTokens: number | null;
  kvCapacity: number | null;
}

/** Figures for the stats ticker, over windows that end at the store's trace time. */
export function tickerStats(store: TelemetryStore): Ticker {
  const rate = within(store.sampled, store.nowT, RATE_WINDOW_MS).reduce((a, b) => a + b, 0);
  const ttfts = sorted(within(store.ttfts, store.nowT, LATENCY_WINDOW_MS));
  const tpots = sorted(within(store.tpots, store.nowT, LATENCY_WINDOW_MS));
  const step = store.lastStep;
  return {
    tokensPerSecond: (rate * 1_000) / RATE_WINDOW_MS,
    ttft: ttfts.length ? { p50: percentile(ttfts, 0.5), p99: percentile(ttfts, 0.99), n: ttfts.length } : null,
    tpot: tpots.length ? { p50: percentile(tpots, 0.5), n: tpots.length } : null,
    batch: step?.running ?? null,
    batchCap: store.hello?.limits.max_num_seqs ?? null,
    queue: step?.waiting ?? null,
    kvTokens: step?.kv_tokens ?? null,
    kvCapacity: store.hello?.kv.capacity_tokens ?? null,
  };
}

/** Nearest rank: the smallest value with at least `q` of the values at or below it. */
export function percentile(sortedValues: readonly number[], q: number): number {
  const rank = Math.max(1, Math.ceil(q * sortedValues.length));
  return sortedValues[rank - 1]!;
}

function within(ring: Ring<Sample>, now: number, windowMs: number): number[] {
  const values: number[] = [];
  for (const sample of ring.newestFirst()) {
    if (sample.t <= now - windowMs) break;
    values.push(sample.value);
  }
  return values;
}

function sorted(values: number[]): number[] {
  return values.sort((a, b) => a - b);
}
