import { describe, expect, it } from "vitest";
import type { Frame } from "./contracts/telemetry";
import { FEED_ROWS, TelemetryStore } from "./store";
import { tabAfterKey, tabCounts, tableRows } from "./table";
import * as ev from "./testing/events";
import { fixtureFrames } from "./testing/fixtures";

function storeOf(frames: Frame[]): TelemetryStore {
  const store = new TelemetryStore(() => {});
  frames.forEach((f) => store.ingest(f));
  return store;
}

const preempted = (t: number, rid: string): ev.Event => ({ type: "req.preempted", t, rid, completion_tokens: 1, blocks_freed: 1 });
const aborted = (t: number, rid: string): ev.Event => ({ type: "req.aborted", t, rid, reason: "client_disconnect", completion_tokens: 2 });

/** a running, b prefilling, c waiting, d preempted, e finished, f aborted. */
function everyState(): TelemetryStore {
  const rids = ["a", "b", "c", "d", "e", "f"];
  return storeOf(
    ev.frames([
      [ev.hello(), ...rids.map((rid, i) => ev.arrived(i, rid))],
      ["a", "b", "d", "e", "f"].map((rid) => ev.admitted(10, rid)),
      [ev.step(20, "prefill", ["a"]), ev.step(21, "prefill", ["d"]), ev.step(22, "prefill", ["e"]), ev.step(23, "prefill", ["f"])],
      [preempted(30, "d"), ev.finished(31, "e", 1, null), aborted(32, "f")],
    ]),
  );
}

describe("tableRows", () => {
  it("puts each request under one tab, prefilling with running and aborted with finished", () => {
    const store = everyState();
    const rids = (tab: Parameters<typeof tableRows>[1]) => tableRows(store, tab).map((r) => r.rid);
    expect(rids("running")).toEqual(["a", "b"]);
    expect(rids("waiting")).toEqual(["c"]);
    expect(rids("preempted")).toEqual(["d"]);
    expect(rids("finished")).toEqual(["f", "e"]);
    expect(tabCounts(store)).toEqual({ running: 2, waiting: 1, finished: 2, preempted: 1 });
  });

  it("lists requests in flight oldest arrival first, ended ones newest end first", () => {
    const store = storeOf(
      ev.frames([
        [ev.hello(), ev.arrived(1, "x"), ev.arrived(2, "y"), ev.arrived(3, "z")],
        [ev.finished(9, "y", 1, null), ev.finished(10, "x", 1, null)],
      ]),
    );
    expect(tableRows(store, "waiting").map((r) => r.rid)).toEqual(["z"]);
    expect(tableRows(store, "finished").map((r) => r.rid)).toEqual(["x", "y"]);
  });

  it("derives TPOT from step times while running and takes the engine's once finished", () => {
    const store = storeOf(
      ev.frames([
        [ev.hello(), ev.arrived(0, "a"), ev.admitted(0, "a")],
        [ev.step(100, "prefill", ["a"]), ev.firstToken(100, "a", 100)],
        [ev.step(120, "decode", ["a"]), ev.step(160, "decode", ["a"])],
      ]),
    );
    expect(tableRows(store, "running")[0]?.tpotMs).toBe(30);
    store.ingest(ev.frames([[ev.finished(170, "a", 3, 33)]], 3)[0]!);
    expect(tableRows(store, "finished")[0]?.tpotMs).toBe(33);
  });

  it("has no TPOT before a second token", () => {
    const store = storeOf(
      ev.frames([[ev.hello(), ev.arrived(0, "a"), ev.admitted(0, "a"), ev.step(100, "prefill", ["a"]), ev.firstToken(100, "a", 100)]]),
    );
    expect(tableRows(store, "running")[0]?.tpotMs).toBeNull();
  });

  it("gives the cache-hit share of the prompt, and the age up to now or to the end", () => {
    const store = storeOf(
      ev.frames([
        [ev.hello(), ev.arrived(100, "a"), ev.arrived(100, "b"), { ...ev.admitted(150, "a"), cached_tokens: 25 } as ev.Event],
        [ev.finished(400, "b", 1, null)],
        [ev.arrived(1_000, "c")],
      ]),
    );
    const [a] = tableRows(store, "running");
    const [b] = tableRows(store, "finished");
    expect(a).toMatchObject({ cacheHit: 0.25, ageMs: 900 });
    expect(b).toMatchObject({ cacheHit: null, ageMs: 300 });
  });

  it("covers every request of a fixture in exactly one tab", async () => {
    const store = storeOf(await fixtureFrames("prefix-pressure.jsonl.gz"));
    const counts = tabCounts(store);
    expect(counts.running + counts.waiting + counts.finished + counts.preempted).toBe([...store.rows()].length);
  });

  it("lists every request in flight and the newest ended ones, beyond the feed's rows", () => {
    const groups: ev.Event[][] = [[ev.hello()]];
    for (let i = 0; i < FEED_ROWS + 30; i++) groups.push([ev.arrived(i, `done${i}`), ev.finished(i, `done${i}`, 1, null)]);
    for (let i = 0; i < 40; i++) groups.push([ev.arrived(1_000 + i, `live${i}`)]);
    const store = storeOf(ev.frames(groups));
    expect(store.feed()).toHaveLength(FEED_ROWS);
    expect(tabCounts(store)).toEqual({ running: 0, waiting: 40, finished: FEED_ROWS, preempted: 0 });
    expect(tableRows(store, "finished")[0]?.rid).toBe(`done${FEED_ROWS + 29}`);
  });
});

describe("tabAfterKey", () => {
  it("moves with the arrows, wrapping at each end, and jumps with Home and End (WAI-ARIA tabs)", () => {
    expect(tabAfterKey("running", "ArrowRight")).toBe("waiting");
    expect(tabAfterKey("preempted", "ArrowRight")).toBe("running");
    expect(tabAfterKey("running", "ArrowLeft")).toBe("preempted");
    expect(tabAfterKey("finished", "Home")).toBe("running");
    expect(tabAfterKey("waiting", "End")).toBe("preempted");
  });

  it("ignores other keys", () => {
    expect(tabAfterKey("running", "Enter")).toBeNull();
  });
});
