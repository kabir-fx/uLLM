import { describe, expect, it } from "vitest";
import type { Step } from "./contracts/telemetry";
import { Ring } from "./ring";
import { stripModel } from "./strip";
import * as ev from "./testing/events";

function ring(steps: Step[]): Ring<Step> {
  const r = new Ring<Step>(64);
  steps.forEach((s) => r.push(s));
  return r;
}

/** One string per lane, one character per column: P prefill, D decode, . empty. */
function picture(steps: Step[], columns: number): string[] {
  const model = stripModel(ring(steps), columns);
  const lanes = Array.from({ length: model.lanes }, () => Array<string>(columns).fill("."));
  for (const c of model.cells) lanes[c.lane]![c.column] = c.kind === "prefill" ? "P" : "D";
  return lanes.map((l) => l.join(""));
}

describe("stripModel", () => {
  it("puts the newest step in the rightmost column", () => {
    expect(picture([ev.step(1, "prefill", ["a"]), ev.step(2, "decode", ["a"])], 4)).toEqual(["..PD"]);
  });

  it("keeps a request in one lane, and gives requests in one step their own lanes", () => {
    const steps = [
      ev.step(1, "prefill", ["a"]),
      ev.step(2, "prefill", ["b"]),
      ev.step(3, "decode", ["a", "b"]),
      ev.step(4, "decode", ["b", "a"]),
    ];
    expect(picture(steps, 4)).toEqual(["P.DD", ".PDD"]);
  });

  it("gives a free lane to a later request, after one empty column", () => {
    const steps = [
      ev.step(1, "prefill", ["a"]),
      ev.step(2, "decode", ["a"]),
      ev.step(3, "prefill", ["b"]),
      ev.step(4, "prefill", ["c"]),
      ev.step(5, "decode", ["b", "c"]),
    ];
    expect(picture(steps, 5)).toEqual(["PD.PD", "..P.D"]);
  });

  it("holds the lane of a request that leaves and comes back inside the window", () => {
    const steps = [
      ev.step(1, "prefill", ["a"]),
      ev.step(2, "prefill", ["b"]),
      ev.step(3, "decode", ["b"]),
      ev.step(4, "prefill", ["c"]),
      ev.step(5, "prefill", ["a"]),
    ];
    expect(picture(steps, 5)).toEqual(["P...P", ".PD..", "...P."]);
  });

  it("shows only the newest steps that fit", () => {
    const steps = [ev.step(1, "prefill", ["old"]), ev.step(2, "prefill", ["a"]), ev.step(3, "decode", ["a"])];
    expect(picture(steps, 2)).toEqual(["PD"]);
  });
});
