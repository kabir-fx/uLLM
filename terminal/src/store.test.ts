import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import type { Frame } from "./contracts/telemetry";
import { FEED_ROWS, TelemetryStore } from "./store";
import * as ev from "./testing/events";
import { parseTrace } from "./trace";

const fixture = (name: string) =>
  parseTrace(
    new Uint8Array(readFileSync(new URL(`../fixtures/telemetry/${name}`, import.meta.url))),
  );

const manual = () => {
  const pending: (() => void)[] = [];
  return { schedule: (cb: () => void) => pending.push(cb), flush: () => pending.splice(0).forEach((cb) => cb()) };
};

function storeOf(frames: Frame[]): TelemetryStore {
  const store = new TelemetryStore(() => {});
  frames.forEach((f) => store.ingest(f));
  return store;
}

describe("TelemetryStore on the fixtures", () => {
  it.each([
    ["steady-chat.jsonl.gz", 72, 2],
    ["prefix-pressure.jsonl.gz", 71, 2],
  ])("ends %s with every request finished or aborted", async (name, nFinished, nAborted) => {
    const store = storeOf(await fixture(name));
    const states = store.feed().map((r) => r.state);
    expect(states.filter((s) => s === "finished")).toHaveLength(nFinished);
    expect(states.filter((s) => s === "aborted")).toHaveLength(nAborted);
    expect(store.missedFrames).toBe(0);
    expect(store.hello?.source).toBe("fixture");
  });
});

describe("TelemetryStore life cycle", () => {
  it("moves a request through waiting, prefilling, running and finished", () => {
    const store = storeOf(ev.frames([[ev.hello(), ev.arrived(1, "a")]]));
    const state = () => store.feed()[0]?.state;
    expect(state()).toBe("waiting");
    store.ingest(ev.frames([[ev.admitted(2, "a")]], 1)[0]!);
    expect(state()).toBe("prefilling");
    store.ingest(ev.frames([[ev.step(3, "prefill", ["a"]), ev.firstToken(3, "a", 2)]], 2)[0]!);
    expect(state()).toBe("running");
    expect(store.feed()[0]?.ttftMs).toBe(2);
    store.ingest(ev.frames([[ev.finished(4, "a", 1, null)]], 3)[0]!);
    expect(state()).toBe("finished");
  });

  it("counts one token per prefill row and per decode row", () => {
    const store = storeOf(
      ev.frames([
        [ev.hello(), ev.arrived(1, "a"), ev.arrived(1, "b"), ev.admitted(1, "a"), ev.admitted(1, "b")],
        [ev.step(2, "prefill", ["a"]), ev.step(3, "prefill", ["b"])],
        [ev.step(4, "decode", ["a", "b"]), ev.step(5, "decode", ["a", "b"]), ev.step(6, "decode", ["b"])],
      ]),
    );
    const tokens = Object.fromEntries(store.feed().map((r) => [r.rid, r.tokens]));
    expect(tokens).toEqual({ a: 3, b: 4 });
  });

  it("takes the engine's count when a request ends", () => {
    const store = storeOf(
      ev.frames([[ev.hello(), ev.arrived(1, "a"), ev.admitted(1, "a")], [ev.finished(9, "a", 7, 3)]]),
    );
    expect(store.feed()[0]?.tokens).toBe(7);
  });

  it("ignores events for a request it never saw arrive", () => {
    const store = storeOf(ev.frames([[ev.hello(), ev.step(1, "decode", ["x"]), ev.finished(2, "x", 5, 1)]]));
    expect(store.feed()).toEqual([]);
    expect(store.lastStep?.rids).toEqual(["x"]);
  });

  it("ignores event types it does not know", () => {
    const unknown = { type: "future.event", t: 1 } as unknown as Frame["events"][number];
    const store = storeOf(ev.frames([[ev.hello(), unknown, ev.arrived(1, "a")]]));
    expect(store.feed()).toHaveLength(1);
  });
});

describe("TelemetryStore frames", () => {
  it("counts the frames missing between two seq numbers and the dropped events", () => {
    const [f0, f1, f2] = ev.frames([[ev.hello()], [], []]);
    const store = storeOf([f0!, { ...f1!, dropped: 3 }, { ...f2!, seq: 5, dropped: 2 }]);
    expect(store.missedFrames).toBe(3);
    expect(store.droppedEvents).toBe(5);
  });

  it("starts a new session at hello", () => {
    const store = storeOf([
      ...ev.frames([[ev.hello(), ev.arrived(1, "a")], [], []]),
      ...ev.frames([[ev.hello({ model: "other" }), ev.arrived(1, "b")]]),
    ]);
    expect(store.feed().map((r) => r.rid)).toEqual(["b"]);
    expect(store.hello?.model).toBe("other");
    expect(store.missedFrames).toBe(0);
  });

  it("tracks trace time as the newest frame's t", () => {
    const store = storeOf(ev.frames([[ev.hello()], [ev.arrived(250, "a")]]));
    expect(store.nowT).toBe(250);
  });
});

describe("TelemetryStore feed", () => {
  it("lists the newest arrival first and keeps only the newest ended rows", () => {
    const n = FEED_ROWS + 50;
    const groups: ev.Event[][] = [[ev.hello()]];
    for (let i = 0; i < n; i++) {
      const rid = `r${i}`;
      groups.push([ev.arrived(i, rid), ev.admitted(i, rid), ev.finished(i, rid, 1, null)]);
    }
    groups.push([ev.arrived(n, "live")]);
    const feed = storeOf(ev.frames(groups)).feed();
    expect(feed).toHaveLength(FEED_ROWS);
    expect(feed[0]?.rid).toBe("live");
    expect(feed[1]?.rid).toBe(`r${n - 1}`);
  });

  it("never drops a request that is still in flight", () => {
    const groups: ev.Event[][] = [[ev.hello(), ev.arrived(0, "old")]];
    for (let i = 1; i <= FEED_ROWS + 10; i++) groups.push([ev.arrived(i, `r${i}`), ev.finished(i, `r${i}`, 1, null)]);
    const store = storeOf(ev.frames(groups));
    expect(store.request("old")?.state).toBe("waiting");
  });
});

describe("TelemetryStore notifications", () => {
  it("tells subscribers once per scheduled flush, however many frames arrived", () => {
    const clock = manual();
    const store = new TelemetryStore(clock.schedule);
    let calls = 0;
    store.subscribe(() => calls++);
    for (const f of ev.frames([[ev.hello()], [], []])) store.ingest(f);
    expect(calls).toBe(0);
    clock.flush();
    expect(calls).toBe(1);
    const v = store.version;
    store.ingest(ev.frames([[]], 3)[0]!);
    expect(store.version).toBeGreaterThan(v);
  });

  it("stops telling a subscriber after it unsubscribes", () => {
    const clock = manual();
    const store = new TelemetryStore(clock.schedule);
    let calls = 0;
    store.subscribe(() => calls++)();
    store.ingest(ev.frames([[ev.hello()]])[0]!);
    clock.flush();
    expect(calls).toBe(0);
  });
});
