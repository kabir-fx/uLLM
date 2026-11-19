import type { Step } from "./contracts/telemetry";
import type { Ring } from "./ring";

export interface StripCell {
  column: number;
  lane: number;
  kind: Step["kind"];
}

export interface StripModel {
  lanes: number;
  cells: StripCell[];
}

/**
 * The newest `columns` steps, one per column, the newest at the right. Each request takes the
 * lowest free lane when it first shows and holds it until its last column in view; one empty
 * column then separates it from the next request in that lane. At most `max_num_seqs` requests
 * share a step, so the strip needs about that many lanes however many requests pass through.
 */
export function stripModel(steps: Ring<Step>, columns: number): StripModel {
  const visible = [...steps].slice(-columns);
  const offset = columns - visible.length;
  const lastColumn = new Map<string, number>();
  visible.forEach((step, i) => step.rids.forEach((rid) => lastColumn.set(rid, offset + i)));

  const laneOf = new Map<string, number>();
  const freeFrom: number[] = [];
  const cells: StripCell[] = [];
  visible.forEach((step, i) => {
    const column = offset + i;
    for (const rid of step.rids) {
      let lane = laneOf.get(rid);
      if (lane === undefined) {
        lane = freeFrom.findIndex((from) => from <= column);
        if (lane === -1) lane = freeFrom.length;
        freeFrom[lane] = lastColumn.get(rid)! + 2;
        laneOf.set(rid, lane);
      }
      cells.push({ column, lane, kind: step.kind });
    }
  });
  return { lanes: freeFrom.length, cells };
}
