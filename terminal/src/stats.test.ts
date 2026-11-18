import { describe, expect, it } from "vitest";
import { percentile, tickerStats } from "./stats";
import { TelemetryStore } from "./store";
import * as ev from "./testing/events";

function storeOf(groups: ev.Event[][]): TelemetryStore {
  const store = new TelemetryStore(() => {});
  ev.frames(groups).forEach((f) => store.ingest(f));
  return store;
}

describe("percentile", () => {
  it("takes the nearest rank", () => {
    const sorted = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    expect(percentile(sorted, 0.5)).toBe(5);
    expect(percentile(sorted, 0.99)).toBe(10);
    expect(percentile([7], 0.99)).toBe(7);
  });
});

describe("tickerStats", () => {
  it("counts the tokens sampled in the last second of trace time", () => {
    const store = storeOf([
      [ev.hello()],
      [ev.step(500, "decode", ["a", "b", "c"])],
      [ev.step(1500, "decode", ["a", "b"]), ev.step(2400, "decode", ["a", "b"])],
    ]);
    expect(store.nowT).toBe(2400);
    expect(tickerStats(store).tokensPerSecond).toBe(4);
  });

  it("takes TTFT and TPOT from the requests of the last 60 seconds", () => {
    const old = [ev.arrived(0, "old"), ev.firstToken(1000, "old", 900), ev.finished(1000, "old", 2, 900)];
    const recent = [10, 20, 30, 40].flatMap((ms, i) => [
      ev.arrived(61_000, `r${i}`),
      ev.firstToken(61_000 + ms, `r${i}`, ms),
      ev.finished(61_500, `r${i}`, 5, ms / 10),
    ]);
    const stats = tickerStats(storeOf([[ev.hello(), ...old], recent]));
    expect(stats.ttft).toEqual({ p50: 20, p99: 40, n: 4 });
    expect(stats.tpot).toEqual({ p50: 2, n: 4 });
  });

  it("has no latency figures before the first request", () => {
    const stats = tickerStats(storeOf([[ev.hello()]]));
    expect(stats.ttft).toBeNull();
    expect(stats.tpot).toBeNull();
  });

  it("reads batch, queue and KV from the newest step, with caps from hello", () => {
    const store = storeOf([
      [ev.hello()],
      [ev.step(5, "decode", ["a", "b", "c"], { waiting: 4, kv_tokens: 250 })],
    ]);
    expect(tickerStats(store)).toMatchObject({
      batch: 3,
      batchCap: 8,
      queue: 4,
      kvTokens: 250,
      kvCapacity: 1000,
    });
  });
});
