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

  it("reports an error and stops when the sink throws on a frame", async () => {
    const statuses: SourceStatus[] = [];
    let delivered = 0;
    replaySource("t", async () => trace).start({
      frame: () => {
        delivered++;
        throw new TypeError("bad event");
      },
      status: (s) => statuses.push(s),
    });
    await vi.advanceTimersByTimeAsync(10_000);
    expect(delivered).toBe(1);
    expect(statuses.at(-1)).toEqual({ kind: "error", message: "bad event" });
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

describe("replaySource controls", () => {
  async function started(speed = 1) {
    const out = record();
    const replay = replaySource("t", async () => trace, { speed });
    replay.start(out.sink);
    await vi.advanceTimersByTimeAsync(0);
    return { out, replay };
  }
  const seqs = (frames: Frame[]) => frames.map((f) => f.seq);

  it("knows the trace length once it loads", async () => {
    const replay = replaySource("t", async () => trace);
    expect(replay.duration).toBe(0);
    replay.start(record().sink);
    await vi.advanceTimersByTimeAsync(0);
    expect(replay.duration).toBe(1000);
  });

  it("holds the position while paused and plays on from it", async () => {
    const { out, replay } = await started();
    await vi.advanceTimersByTimeAsync(50);
    replay.pause();
    expect(replay.playing).toBe(false);
    expect(replay.position()).toBe(50);
    await vi.advanceTimersByTimeAsync(10_000);
    expect(seqs(out.frames)).toEqual([0]);
    expect(replay.position()).toBe(50);
    replay.play();
    await vi.advanceTimersByTimeAsync(49);
    expect(seqs(out.frames)).toEqual([0]);
    await vi.advanceTimersByTimeAsync(1);
    expect(seqs(out.frames)).toEqual([0, 1, 2]);
    expect(out.statuses).toEqual(["connecting", "open", "paused", "open"]);
  });

  it("changes speed without moving the position", async () => {
    const { out, replay } = await started();
    await vi.advanceTimersByTimeAsync(100);
    replay.setSpeed(4);
    expect(replay.speed).toBe(4);
    expect(replay.position()).toBe(100);
    await vi.advanceTimersByTimeAsync(224);
    expect(out.frames).toHaveLength(3);
    await vi.advanceTimersByTimeAsync(1);
    expect(out.frames).toHaveLength(4);
  });

  it("sends the frames between on a seek forward, and keeps playing from there", async () => {
    const { out, replay } = await started();
    replay.seek(500);
    expect(seqs(out.frames)).toEqual([0, 1, 2]);
    expect(replay.position()).toBe(500);
    await vi.advanceTimersByTimeAsync(500);
    expect(seqs(out.frames)).toEqual([0, 1, 2, 3]);
  });

  it("sends the trace again from its hello on a seek back", async () => {
    const { out, replay } = await started();
    await vi.advanceTimersByTimeAsync(1000);
    replay.seek(100);
    expect(seqs(out.frames)).toEqual([0, 1, 2, 3, 0, 1, 2]);
    expect(out.statuses.at(-1)).toBe("paused");
  });

  it("seeks while paused without playing", async () => {
    const { out, replay } = await started();
    replay.pause();
    replay.seek(100);
    await vi.advanceTimersByTimeAsync(10_000);
    expect(seqs(out.frames)).toEqual([0, 1, 2]);
    expect(replay.playing).toBe(false);
  });

  it("clamps a seek to the trace and ends at its end", async () => {
    const { out, replay } = await started();
    replay.seek(99_999);
    expect(replay.position()).toBe(1000);
    expect(out.frames).toHaveLength(4);
    expect(replay.playing).toBe(false);
    expect(out.statuses.at(-1)).toBe("ended");
    replay.seek(-5);
    expect(replay.position()).toBe(0);
    expect(seqs(out.frames)).toEqual([0, 1, 2, 3, 0]);
  });

  it("starts again from the beginning when played at the end", async () => {
    const { out, replay } = await started(10);
    await vi.advanceTimersByTimeAsync(100);
    expect(out.statuses.at(-1)).toBe("ended");
    replay.play();
    expect(seqs(out.frames)).toEqual([0, 1, 2, 3, 0]);
    expect(replay.playing).toBe(true);
    await vi.advanceTimersByTimeAsync(100);
    expect(seqs(out.frames)).toEqual([0, 1, 2, 3, 0, 1, 2, 3]);
  });

  it("keeps a pause pressed while the trace loads", async () => {
    const out = record();
    const replay = replaySource("t", async () => trace);
    replay.start(out.sink);
    replay.pause();
    expect(replay.playing).toBe(false);
    await vi.advanceTimersByTimeAsync(10_000);
    expect(seqs(out.frames)).toEqual([0]);
    expect(out.statuses).toEqual(["connecting", "paused"]);
    replay.play();
    await vi.advanceTimersByTimeAsync(100);
    expect(seqs(out.frames)).toEqual([0, 1, 2]);
  });

  it("plays when play follows a pause while the trace loads", async () => {
    const out = record();
    const replay = replaySource("t", async () => trace);
    replay.start(out.sink);
    replay.pause();
    replay.play();
    expect(replay.playing).toBe(true);
    await vi.advanceTimersByTimeAsync(10_000);
    expect(seqs(out.frames)).toEqual([0, 1, 2, 3]);
    expect(out.statuses).toEqual(["connecting", "open", "ended"]);
  });

  it("does nothing after stop", async () => {
    const out = record();
    const replay = replaySource("t", async () => trace);
    const stop = replay.start(out.sink);
    await vi.advanceTimersByTimeAsync(0);
    stop();
    replay.seek(1000);
    replay.play();
    await vi.advanceTimersByTimeAsync(10_000);
    expect(seqs(out.frames)).toEqual([0]);
  });
});
