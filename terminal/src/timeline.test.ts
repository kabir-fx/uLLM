import { describe, expect, it } from "vitest";
import { Ring } from "./ring";
import * as ev from "./testing/events";
import { foldStep, type Second } from "./timeline";

function fold(...steps: ReturnType<typeof ev.step>[]): Second[] {
  const seconds = new Ring<Second>(8);
  steps.forEach((s) => foldStep(seconds, s));
  return [...seconds];
}

describe("foldStep", () => {
  it("folds the steps of one second into one candle of step latency", () => {
    const [only, ...rest] = fold(
      ev.step(1_100, "decode", ["a"], { ms: 20 }),
      ev.step(1_200, "decode", ["a"], { ms: 50 }),
      ev.step(1_300, "decode", ["a"], { ms: 5 }),
      ev.step(1_999, "decode", ["a"], { ms: 30 }),
    );
    expect(rest).toEqual([]);
    expect(only).toMatchObject({ second: 1, open: 20, high: 50, low: 5, close: 30, steps: 4 });
  });

  it("starts a new bucket at each second of trace time, by the step's end", () => {
    const seconds = fold(
      ev.step(999, "decode", ["a"], { ms: 1 }),
      ev.step(1_000, "decode", ["a"], { ms: 2 }),
      ev.step(3_500, "decode", ["a"], { ms: 3 }),
    );
    expect(seconds.map((s) => [s.second, s.open])).toEqual([
      [0, 1],
      [1, 2],
      [3, 3],
    ]);
  });

  it("counts processed and sampled tokens apart for prefill and decode", () => {
    const [s] = fold(
      ev.step(100, "prefill", ["a"], { tokens: 500 }),
      ev.step(200, "decode", ["a", "b", "c"], { tokens: 3 }),
      ev.step(300, "prefill", ["d"], { tokens: 250 }),
    );
    expect(s?.processed).toEqual({ prefill: 750, decode: 3 });
    expect(s?.sampled).toEqual({ prefill: 2, decode: 3 });
  });
});
