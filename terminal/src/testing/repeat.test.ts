import { describe, expect, it } from "vitest";
import { TelemetryStore } from "../store";
import * as ev from "./events";
import { repeatTrace } from "./repeat";

const trace = ev.frames([
  [ev.hello()],
  [ev.arrived(100, "a")],
  [ev.step(200, "prefill", ["a"]), ev.snapshot(200, ["a"], [3, 0])],
  [ev.finished(300, "a", 1, null)],
]);

describe("repeatTrace", () => {
  it("plays the trace again with one hello, gap-free seq, later times and fresh request ids", () => {
    const long = repeatTrace(trace, 3);
    expect(long).toHaveLength(12);
    expect(long.map((f) => f.seq)).toEqual([...Array(12).keys()]);
    expect(long.flatMap((f) => f.events).filter((e) => e.type === "hello")).toHaveLength(1);
    // One period is the trace's span plus its first frame gap: 300 + 100 ms.
    expect(long.map((f) => f.t)).toEqual([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]);
    expect(long[9]!.events[0]).toMatchObject({ type: "req.arrived", t: 900, rid: "a.2" });
    expect(long[10]!.events).toEqual([
      expect.objectContaining({ type: "step", t: 1000, rids: ["a.2"] }),
      expect.objectContaining({ type: "kv.snapshot", t: 1000, owners: ["a.2"] }),
    ]);
  });

  it("feeds a store that never resets", () => {
    const store = new TelemetryStore(() => {});
    repeatTrace(trace, 3).forEach((f) => store.ingest(f));
    expect([...store.rows()].map((r) => r.rid)).toEqual(["a", "a.1", "a.2"]);
    expect(store.missedFrames).toBe(0);
  });

  it("leaves the trace it repeats unchanged", () => {
    const before = structuredClone(trace);
    repeatTrace(trace, 2);
    expect(trace).toEqual(before);
  });
});
