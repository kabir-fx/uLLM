import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { TelemetryStore } from "../store";
import * as ev from "../testing/events";
import { fixtureFrames } from "../testing/fixtures";
import type { Mode } from "../config";
import { RUNS, type Run } from "../runs";
import type { TelemetrySource } from "../source";
import type { ReplaySource } from "../sources/replay";
import { App } from "./App";

async function fixtureStore(name: string): Promise<TelemetryStore> {
  const store = new TelemetryStore(() => {});
  (await fixtureFrames(name)).forEach((f) => store.ingest(f));
  store.setStatus({ kind: "ended" });
  return store;
}

const rows = (html: string) => (html.match(/<tr data-rid=/g) ?? []).length;

describe("App", () => {
  it("labels fixture data as simulated and lists every request", async () => {
    const html = renderToStaticMarkup(<App store={await fixtureStore("steady-chat.jsonl.gz")} />);
    expect(html).toContain("SIMULATED");
    expect(rows(html)).toBe(74);
    expect(html).toContain('data-state="aborted"');
  });

  it("lays out the step chart, the batch strip and the requests table beside the feed", async () => {
    const html = renderToStaticMarkup(<App store={await fixtureStore("steady-chat.jsonl.gz")} />);
    expect(html).toMatch(/<section class="panel chart"[^>]*aria-label="step chart"[\s\S]*?<canvas/);
    expect(html).toMatch(/<section class="panel strip"[^>]*aria-label="batch strip"[\s\S]*?<canvas/);
    expect(html).toMatch(/aria-label="requests"[\s\S]*data-tab="finished"[^>]*>finished.*?74/);
  });

  it("labels sim data as simulated", () => {
    const store = new TelemetryStore(() => {});
    store.ingest(ev.frames([[ev.hello({ source: "sim" })]])[0]!);
    expect(renderToStaticMarkup(<App store={store} />)).toContain("SIMULATED");
  });

  it("labels data as simulated until a hello says it comes from the engine", () => {
    const store = new TelemetryStore(() => {});
    store.ingest(ev.frames([[ev.arrived(1, "a")]])[0]!);
    expect(renderToStaticMarkup(<App store={store} />)).toContain("SIMULATED");
  });

  it("does not label engine data as simulated", () => {
    const store = new TelemetryStore(() => {});
    store.ingest(ev.frames([[ev.hello({ source: "engine" })]])[0]!);
    expect(renderToStaticMarkup(<App store={store} />)).not.toContain("SIMULATED");
  });

  it("shows the ticker figures from the store", () => {
    const store = new TelemetryStore(() => {});
    ev.frames([
      [ev.hello(), ev.arrived(0, "a"), ev.admitted(0, "a")],
      [ev.step(500, "prefill", ["a"], { waiting: 2, kv_tokens: 250 }), ev.firstToken(500, "a", 123.4)],
    ]).forEach((f) => store.ingest(f));
    const html = renderToStaticMarkup(<App store={store} />);
    expect(html).toMatch(/data-stat="batch"[^>]*>.*1.*\/.*8/);
    expect(html).toMatch(/data-stat="queue"[^>]*>.*2/);
    expect(html).toMatch(/data-stat="kv"[^>]*>.*25%/);
    expect(html).toMatch(/data-stat="ttft"[^>]*>.*123/);
  });

  it("warns about missed frames and dropped events", () => {
    const store = new TelemetryStore(() => {});
    const [f0, f1] = ev.frames([[ev.hello()], []]);
    store.ingest(f0!);
    store.ingest({ ...f1!, seq: 3, dropped: 4 });
    const html = renderToStaticMarkup(<App store={store} />);
    expect(html).toContain("2 frames missed");
    expect(html).toContain("4 events dropped");
  });

  it("shows a source error", () => {
    const store = new TelemetryStore(() => {});
    store.setStatus({ kind: "error", message: "HTTP 503" });
    expect(renderToStaticMarkup(<App store={store} />)).toContain("HTTP 503");
  });
});

describe("App modes", () => {
  const controls = { api: "http://127.0.0.1:8000", control: "http://127.0.0.1:8090" };
  const idle: TelemetrySource = { label: "x", start: () => () => {} };
  const liveMode: Mode = { kind: "live", source: idle, controls };
  const replayMode = (run: Run | null, at = 0): Mode => {
    const source: ReplaySource = {
      label: "./runs/r.jsonl.gz",
      start: () => () => {},
      playing: true,
      speed: 2,
      duration: 25_000,
      position: () => at,
      play() {},
      pause() {},
      setSpeed() {},
      seek() {},
    };
    return { kind: "replay", source, run };
  };
  const real = RUNS.find((r) => r.recording !== null)!;
  const synthetic = RUNS.find((r) => r.recording === null)!;
  const engine = () => {
    const store = new TelemetryStore(() => {});
    store.ingest(ev.frames([[ev.hello({ source: "engine" })]])[0]!);
    return store;
  };

  it("shows the launcher and the playground in live mode, and no replay label", () => {
    const html = renderToStaticMarkup(<App store={engine()} mode={liveMode} />);
    expect(html).toContain('class="shell with-side"');
    expect(html).toMatch(/<aside class="side">[\s\S]*aria-label="load launcher"[\s\S]*aria-label="playground"/);
    expect(html).toContain("not a benchmark");
    expect(html).not.toContain("REPLAY");
    expect(html).not.toContain('aria-label="replay"');
  });

  it("shows the replay panel in place of the launcher in replay mode", () => {
    const html = renderToStaticMarkup(<App store={engine()} mode={replayMode(real, 12_300)} />);
    expect(html).toContain('class="shell with-side"');
    expect(html).toMatch(/<aside class="side">[\s\S]*aria-label="replay"/);
    expect(html).not.toContain("load launcher");
    expect(html).not.toContain("playground");
    for (const run of RUNS) expect(html).toContain(`<option value="${run.id}"`);
    expect(html).toMatch(new RegExp(`<option value="${real.id}" selected`));
    expect(html).toMatch(/<input[^>]*type="range"[^>]*max="25000"[^>]*value="12300"/);
    expect(html).toContain("0:12.3 / 0:25.0");
    expect(html).toMatch(/<button[^>]*>pause<\/button>/);
    expect(html).toMatch(/<option value="2" selected/);
  });

  it("labels a replay as a replay, with the machine and commit of its recording", () => {
    const html = renderToStaticMarkup(<App store={engine()} mode={replayMode(real)} />);
    expect(html).toContain("REPLAY");
    expect(html).not.toContain("SIMULATED");
    expect(html).toContain("Apple M4 · 16 GB");
    expect(html).toContain("fab197c dirty");
  });

  it("labels a synthetic run as a replay and as simulated", () => {
    const store = new TelemetryStore(() => {});
    store.ingest(ev.frames([[ev.hello({ source: "fixture" })]])[0]!);
    const html = renderToStaticMarkup(<App store={store} mode={replayMode(synthetic)} />);
    expect(html).toContain("REPLAY");
    expect(html).toContain("SIMULATED");
  });

  it("labels a trace that is no bundled run as a replay, with the machine and commit its hello gives", () => {
    const store = new TelemetryStore(() => {});
    const hello = ev.hello({ source: "engine", commit: "0123456789", machine: { chip: "Apple M2", memory_gb: 24 } });
    store.ingest(ev.frames([[hello]])[0]!);
    const html = renderToStaticMarkup(<App store={store} mode={replayMode(null)} />);
    expect(html).toContain("REPLAY");
    expect(html).toContain("Apple M2 · 24 GB");
    expect(html).toContain("0123456");
    expect(html).toContain('<option value="" selected');
  });

  it("shows no side column without a mode", () => {
    const html = renderToStaticMarkup(<App store={new TelemetryStore(() => {})} />);
    expect(html).toContain('class="shell"');
  });
});
