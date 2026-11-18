import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Frame } from "../contracts/telemetry";
import type { SourceStatus } from "../store";
import * as ev from "../testing/events";
import { replaySource } from "./replay";

const trace = ev.frames([[ev.hello()], [ev.arrived(100, "a")], [ev.arrived(100, "b")], [ev.arrived(1000, "c")]]);

function record() {
  const frames: Frame[] = [];
  const statuses: SourceStatus["kind"][] = [];
  return { frames, statuses, sink: { frame: (f: Frame) => frames.push(f), status: (s: SourceStatus) => statuses.push(s.kind) } };
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

describe("replaySource", () => {
  it("sends each frame at its own time from the start of the trace", async () => {
    const out = record();
    replaySource("t", async () => trace).start(out.sink);
    await vi.advanceTimersByTimeAsync(0);
    expect(out.frames.map((f) => f.seq)).toEqual([0]);
    await vi.advanceTimersByTimeAsync(100);
    expect(out.frames.map((f) => f.seq)).toEqual([0, 1, 2]);
    await vi.advanceTimersByTimeAsync(899);
    expect(out.frames).toHaveLength(3);
    await vi.advanceTimersByTimeAsync(1);
    expect(out.frames).toHaveLength(4);
    expect(out.statuses).toEqual(["connecting", "open", "ended"]);
  });

  it("divides the delays by the speed", async () => {
    const out = record();
    replaySource("t", async () => trace, { speed: 4 }).start(out.sink);
    await vi.advanceTimersByTimeAsync(250);
    expect(out.frames).toHaveLength(4);
  });

  it("sends nothing after stop", async () => {
    const out = record();
    const stop = replaySource("t", async () => trace).start(out.sink);
    await vi.advanceTimersByTimeAsync(100);
    stop();
    await vi.advanceTimersByTimeAsync(10_000);
    expect(out.frames).toHaveLength(3);
    expect(out.statuses).toEqual(["connecting", "open"]);
  });

  it("sends nothing when stopped while the trace loads", async () => {
    const out = record();
    const stop = replaySource("t", async () => trace).start(out.sink);
    stop();
    await vi.advanceTimersByTimeAsync(10_000);
    expect(out.frames).toEqual([]);
  });

  it("reports a trace that fails to load", async () => {
    const out = record();
    const statuses: SourceStatus[] = [];
    replaySource("t", async () => {
      throw new Error("trace line 2: bad");
    }).start({ ...out.sink, status: (s) => statuses.push(s) });
    await vi.advanceTimersByTimeAsync(0);
    expect(statuses.at(-1)).toEqual({ kind: "error", message: "trace line 2: bad" });
  });
});
