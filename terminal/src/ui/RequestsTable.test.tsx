import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { TelemetryStore } from "../store";
import * as ev from "../testing/events";
import { RequestsTable } from "./RequestsTable";

const rows = (html: string) => (html.match(/<tr data-row=/g) ?? []).length;

function storeWith(groups: ev.Event[][]): TelemetryStore {
  const store = new TelemetryStore(() => {});
  ev.frames(groups).forEach((f) => store.ingest(f));
  return store;
}

describe("RequestsTable", () => {
  it("shows every tab with its count and opens on Running", () => {
    const store = storeWith([[ev.hello(), ev.arrived(0, "a"), ev.arrived(0, "b"), ev.admitted(1, "a")]]);
    const html = renderToStaticMarkup(<RequestsTable store={store} />);
    expect(html).toMatch(/data-tab="running"[^>]*aria-selected="true"[^>]*>running.*?1/);
    expect(html).toMatch(/data-tab="waiting"[^>]*aria-selected="false"[^>]*>waiting.*?1/);
    expect(html).toMatch(/data-tab="finished"[^>]*>finished.*?0/);
    expect(html).toMatch(/data-tab="preempted"[^>]*>preempted.*?0/);
    expect(rows(html)).toBe(1);
    expect(html).toContain('data-row="a"');
  });

  it("renders a row's figures, with a dash for what is not known yet", () => {
    const store = storeWith([
      [ev.hello(), ev.arrived(0, "a"), { ...ev.admitted(0, "a"), cached_tokens: 50 } as ev.Event],
      [ev.step(100, "prefill", ["a"]), ev.firstToken(100, "a", 100)],
      [ev.step(120, "decode", ["a"]), ev.step(140, "decode", ["a"])],
    ]);
    const cells = renderToStaticMarkup(<RequestsTable store={store} />)
      .match(/<tr data-row="a">(.*?)<\/tr>/)![1]!
      .match(/<td[^>]*>(.*?)<\/td>/g)!
      .map((td) => td.replace(/<[^>]+>/g, ""));
    expect(cells).toEqual(["a", "running", "100", "3", "100 ms", "20 ms", "—", "50%", "140 ms"]);
  });

  it("renders only the rows near the view, and keeps the full scroll height", () => {
    const groups: ev.Event[][] = [[ev.hello()]];
    for (let i = 0; i < 500; i++) groups.push([ev.arrived(i, `r${i}`), ev.admitted(i, `r${i}`)]);
    const html = renderToStaticMarkup(<RequestsTable store={storeWith(groups)} />);
    expect(rows(html)).toBeGreaterThan(10);
    expect(rows(html)).toBeLessThan(60);
    expect(html).toContain('data-row="r0"');
    expect(html).toMatch(/<tr style="height:\d+px" aria-hidden="true">/);
  });
});
