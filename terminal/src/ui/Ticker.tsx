import { tickerStats } from "../stats";
import type { TelemetryStore } from "../store";
import { int, ms } from "./format";

export function Ticker({ store }: { store: TelemetryStore }) {
  const s = tickerStats(store);
  const kvShare = s.kvTokens !== null && s.kvCapacity ? `${Math.round((100 * s.kvTokens) / s.kvCapacity)}%` : null;
  return (
    <section className="ticker" aria-label="stats">
      <Stat id="tps" name="tok/s" value={int(s.tokensPerSecond)} note="last 1 s" />
      <Stat id="ttft" name="TTFT p50" value={ms(s.ttft?.p50)} note={s.ttft ? `p99 ${ms(s.ttft.p99)} · n ${s.ttft.n}` : "last 60 s"} />
      <Stat id="tpot" name="TPOT p50" value={ms(s.tpot?.p50)} note={s.tpot ? `n ${s.tpot.n}` : "last 60 s"} />
      <Stat id="batch" name="batch" value={`${int(s.batch)} / ${int(s.batchCap)}`} note="running / max" />
      <Stat id="queue" name="queue" value={int(s.queue)} note="waiting" />
      <Stat id="kv" name="KV" value={kvShare ?? int(s.kvTokens)} note={s.kvCapacity ? `${int(s.kvTokens)} of ${int(s.kvCapacity)} tokens` : "tokens"} />
    </section>
  );
}

function Stat({ id, name, value, note }: { id: string; name: string; value: string; note: string }) {
  return (
    <div className="stat" data-stat={id}>
      <div className="stat-name">{name}</div>
      <div className="stat-value">{value}</div>
      <div className="stat-note">{note}</div>
    </div>
  );
}
