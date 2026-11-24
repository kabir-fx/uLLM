import type { RequestRow, RequestState, TelemetryStore } from "./store";

export const TABS = ["running", "waiting", "finished", "preempted"] as const;
export type Tab = (typeof TABS)[number];

/** The WAI-ARIA tabs pattern: the arrows move and wrap, Home and End jump; null for any other key. */
export function tabAfterKey(tab: Tab, key: string): Tab | null {
  const i = TABS.indexOf(tab);
  const n = TABS.length;
  switch (key) {
    case "ArrowRight":
      return TABS[(i + 1) % n]!;
    case "ArrowLeft":
      return TABS[(i + n - 1) % n]!;
    case "Home":
      return TABS[0];
    case "End":
      return TABS[n - 1]!;
    default:
      return null;
  }
}

const TAB_OF: Record<RequestState, Tab> = {
  waiting: "waiting",
  prefilling: "running",
  running: "running",
  preempted: "preempted",
  finished: "finished",
  aborted: "finished",
};

export interface TableRow {
  rid: string;
  state: RequestState;
  promptTokens: number;
  tokens: number;
  ttftMs: number | null;
  tpotMs: number | null;
  blocks: number | null;
  /** Cached prompt tokens as a share of the prompt, from 0 to 1. */
  cacheHit: number | null;
  ageMs: number;
}

/** In-flight tabs list the oldest arrival first, as the scheduler serves them. Finished lists the newest end first. */
export function tableRows(store: TelemetryStore, tab: Tab): TableRow[] {
  const rows = [...store.rows()].filter((r) => TAB_OF[r.state] === tab);
  if (tab === "finished") rows.sort((a, b) => b.endT! - a.endT!);
  return rows.map((r) => ({
    rid: r.rid,
    state: r.state,
    promptTokens: r.promptTokens,
    tokens: r.tokens,
    ttftMs: r.ttftMs,
    tpotMs: r.state === "finished" ? r.tpotMs : observedTpot(r),
    blocks: r.blocks,
    cacheHit: r.cachedTokens !== null && r.promptTokens > 0 ? r.cachedTokens / r.promptTokens : null,
    ageMs: (r.endT ?? store.nowT) - r.arrivedT,
  }));
}

export function tabCounts(store: TelemetryStore): Record<Tab, number> {
  const counts: Record<Tab, number> = { running: 0, waiting: 0, finished: 0, preempted: 0 };
  for (const r of store.rows()) counts[TAB_OF[r.state]] += 1;
  return counts;
}

/** The engine's definition (`ullm/core/request.py`): time from the first token to the newest, per token after the first. */
function observedTpot(r: RequestRow): number | null {
  if (r.firstTokenT === null || r.lastTokenT === null || r.tokens < 2) return null;
  return (r.lastTokenT - r.firstTokenT) / (r.tokens - 1);
}
