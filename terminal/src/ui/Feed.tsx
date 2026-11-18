import type { TelemetryStore } from "../store";
import { int, ms } from "./format";

export function Feed({ store }: { store: TelemetryStore }) {
  return (
    <section className="feed" aria-label="request feed">
      <table>
        <thead>
          <tr>
            <th>request</th>
            <th>state</th>
            <th className="num">prompt</th>
            <th className="num">tokens</th>
            <th className="num">TTFT</th>
          </tr>
        </thead>
        <tbody>
          {store.feed().map((r) => (
            <tr data-rid={r.rid} key={r.rid}>
              <td>{r.rid}</td>
              <td>
                <span className={`pill pill-${r.state}`} data-state={r.state}>
                  {r.state}
                </span>
              </td>
              <td className="num">{int(r.promptTokens)}</td>
              <td className="num">
                {int(r.tokens)} <span className="dim">/ {int(r.maxTokens)}</span>
              </td>
              <td className="num">{ms(r.ttftMs)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
