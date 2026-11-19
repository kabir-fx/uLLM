import { useEffect, useRef, useState } from "react";
import type { TelemetryStore } from "../store";
import { TABS, tabCounts, tableRows, type Tab } from "../table";
import { int, ms, percent } from "./format";
import { ROW_HEIGHT, visibleRange } from "./window";

/** Used until the scroll box is measured, and when rendering without a DOM. */
const DEFAULT_VIEWPORT = 600;

export function RequestsTable({ store }: { store: TelemetryStore }) {
  const [tab, setTab] = useState<Tab>("running");
  const [scrollTop, setScrollTop] = useState(0);
  const [viewport, setViewport] = useState(DEFAULT_VIEWPORT);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = box.current!;
    const observer = new ResizeObserver(() => setViewport(el.clientHeight));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const rows = tableRows(store, tab);
  const counts = tabCounts(store);
  const { start, end } = visibleRange(scrollTop, viewport, rows.length);
  const choose = (next: Tab) => {
    setTab(next);
    setScrollTop(0);
    box.current?.scrollTo({ top: 0 });
  };

  return (
    <section className="requests" aria-label="requests">
      <div className="tabs" role="tablist">
        {TABS.map((t) => (
          <button key={t} role="tab" data-tab={t} aria-selected={t === tab} onClick={() => choose(t)}>
            {t} <span className="dim">{counts[t]}</span>
          </button>
        ))}
      </div>
      <div className="scroll" ref={box} onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)}>
        <table>
          <thead>
            <tr>
              <th>request</th>
              <th>state</th>
              <th className="num">prompt</th>
              <th className="num">generated</th>
              <th className="num">TTFT</th>
              <th className="num">TPOT</th>
              <th className="num">blocks</th>
              <th className="num">cache hit</th>
              <th className="num">age</th>
            </tr>
          </thead>
          <tbody>
            {start > 0 && <tr style={{ height: start * ROW_HEIGHT }} aria-hidden="true" />}
            {rows.slice(start, end).map((r) => (
              <tr data-row={r.rid} key={r.rid}>
                <td>{r.rid}</td>
                <td>
                  <span className={`pill pill-${r.state}`}>{r.state}</span>
                </td>
                <td className="num">{int(r.promptTokens)}</td>
                <td className="num">{int(r.tokens)}</td>
                <td className="num">{ms(r.ttftMs)}</td>
                <td className="num">{ms(r.tpotMs)}</td>
                <td className="num">{int(r.blocks)}</td>
                <td className="num">{percent(r.cacheHit)}</td>
                <td className="num">{ms(r.ageMs)}</td>
              </tr>
            ))}
            {end < rows.length && <tr style={{ height: (rows.length - end) * ROW_HEIGHT }} aria-hidden="true" />}
          </tbody>
        </table>
      </div>
    </section>
  );
}
