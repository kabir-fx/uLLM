import { describe, expect, it } from "vitest";
import { visibleRange } from "./window";

describe("visibleRange", () => {
  it("covers the rows in view plus the overscan on each side", () => {
    expect(visibleRange(400, 200, 1_000, 20, 5)).toEqual({ start: 15, end: 35 });
  });

  it("clamps to the first and the last row", () => {
    expect(visibleRange(0, 200, 1_000, 20, 5)).toEqual({ start: 0, end: 15 });
    expect(visibleRange(19_900, 200, 1_000, 20, 5)).toEqual({ start: 990, end: 1_000 });
    expect(visibleRange(0, 200, 3, 20, 5)).toEqual({ start: 0, end: 3 });
  });

  it("counts a row cut by the bottom edge as in view", () => {
    expect(visibleRange(10, 200, 1_000, 20, 0)).toEqual({ start: 0, end: 11 });
  });
});
