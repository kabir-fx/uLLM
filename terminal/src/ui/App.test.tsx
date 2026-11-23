import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { TelemetryStore } from "../store";
import * as ev from "../testing/events";
import { fixtureFrames } from "../testing/fixtures";
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
    const html = renderToStaticMarkup(<App store={await fixtureStore("steady-chat.jsonl.gz")} label="steady" />);
    expect(html).toContain("SIMULATED");
    expect(rows(html)).toBe(74);
    expect(html).toContain('data-state="aborted"');
  });

  it("lays out the step chart, the batch strip and the requests table beside the feed", async () => {
    const html = renderToStaticMarkup(<App store={await fixtureStore("steady-chat.jsonl.gz")} label="steady" />);
    expect(html).toMatch(/<section class="panel chart"[^>]*aria-label="step chart"[\s\S]*?<canvas/);
    expect(html).toMatch(/<section class="panel strip"[^>]*aria-label="batch strip"[\s\S]*?<canvas/);
    expect(html).toMatch(/aria-label="requests"[\s\S]*data-tab="finished"[^>]*>finished.*?74/);
  });

  it("labels sim data as simulated", () => {
    const store = new TelemetryStore(() => {});
    store.ingest(ev.frames([[ev.hello({ source: "sim" })]])[0]!);
    expect(renderToStaticMarkup(<App store={store} label="x" />)).toContain("SIMULATED");
  });

  it("labels data as simulated until a hello says it comes from the engine", () => {
    const store = new TelemetryStore(() => {});
    store.ingest(ev.frames([[ev.arrived(1, "a")]])[0]!);
    expect(renderToStaticMarkup(<App store={store} label="x" />)).toContain("SIMULATED");
  });

  it("does not label engine data as simulated", () => {
    const store = new TelemetryStore(() => {});
    store.ingest(ev.frames([[ev.hello({ source: "engine" })]])[0]!);
    expect(renderToStaticMarkup(<App store={store} label="x" />)).not.toContain("SIMULATED");
  });

  it("shows the ticker figures from the store", () => {
    const store = new TelemetryStore(() => {});
    ev.frames([
      [ev.hello(), ev.arrived(0, "a"), ev.admitted(0, "a")],
      [ev.step(500, "prefill", ["a"], { waiting: 2, kv_tokens: 250 }), ev.firstToken(500, "a", 123.4)],
    ]).forEach((f) => store.ingest(f));
    const html = renderToStaticMarkup(<App store={store} label="x" />);
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
    const html = renderToStaticMarkup(<App store={store} label="x" />);
    expect(html).toContain("2 frames missed");
    expect(html).toContain("4 events dropped");
  });

  it("shows a source error", () => {
    const store = new TelemetryStore(() => {});
    store.setStatus({ kind: "error", message: "HTTP 503" });
    expect(renderToStaticMarkup(<App store={store} label="x" />)).toContain("HTTP 503");
  });
});

describe("App controls", () => {
  const controls = { api: "http://127.0.0.1:8000", control: "http://127.0.0.1:8090" };

  it("shows the launcher and the playground in live mode", () => {
    const html = renderToStaticMarkup(<App store={new TelemetryStore(() => {})} label="x" controls={controls} />);
    expect(html).toContain('class="shell with-side"');
    expect(html).toMatch(/<aside class="side">[\s\S]*aria-label="load launcher"[\s\S]*aria-label="playground"/);
    expect(html).toContain("not a benchmark");
  });

  it("shows neither in replay mode", () => {
    const html = renderToStaticMarkup(<App store={new TelemetryStore(() => {})} label="x" controls={null} />);
    expect(html).toContain('class="shell"');
    expect(html).not.toContain("load launcher");
    expect(html).not.toContain("playground");
  });
});
