import { describe, expect, it } from "vitest";
import { Ring } from "./ring";

describe("Ring", () => {
  it("keeps items in push order until it is full", () => {
    const ring = new Ring<number>(3);
    ring.push(1);
    ring.push(2);
    expect([...ring]).toEqual([1, 2]);
    expect(ring.length).toBe(2);
  });

  it("writes over the oldest item when full", () => {
    const ring = new Ring<number>(3);
    for (const n of [1, 2, 3, 4, 5]) ring.push(n);
    expect([...ring]).toEqual([3, 4, 5]);
    expect(ring.length).toBe(3);
  });

  it("gives the newest item first from newestFirst", () => {
    const ring = new Ring<number>(3);
    for (const n of [1, 2, 3, 4]) ring.push(n);
    expect([...ring.newestFirst()]).toEqual([4, 3, 2]);
    expect(ring.newest()).toBe(4);
  });

  it("is empty after clear", () => {
    const ring = new Ring<number>(2);
    ring.push(1);
    ring.clear();
    expect([...ring]).toEqual([]);
    expect(ring.newest()).toBeUndefined();
  });
});
