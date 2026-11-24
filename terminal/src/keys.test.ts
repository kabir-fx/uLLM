import { describe, expect, it } from "vitest";
import { replayKey, SEEK_STEP_MS, type KeyTarget } from "./keys";

const key = (k: string, over: Partial<Parameters<typeof replayKey>[0]> = {}) => ({
  key: k,
  altKey: false,
  ctrlKey: false,
  metaKey: false,
  shiftKey: false,
  target: null,
  ...over,
});

/** Stands in for an element: `closest` answers for the selector the handler asks about. */
const inside = (matches: boolean): KeyTarget => ({ closest: () => (matches ? {} : null) });

describe("replayKey", () => {
  it("maps Space to play or pause, and the arrows to a 5 s seek", () => {
    expect(replayKey(key(" "))).toEqual({ kind: "toggle" });
    expect(replayKey(key("ArrowLeft"))).toEqual({ kind: "seek", byMs: -SEEK_STEP_MS });
    expect(replayKey(key("ArrowRight"))).toEqual({ kind: "seek", byMs: SEEK_STEP_MS });
    expect(SEEK_STEP_MS).toBe(5000);
  });

  it("ignores other keys and any key with a modifier", () => {
    expect(replayKey(key("a"))).toBeNull();
    for (const modifier of ["altKey", "ctrlKey", "metaKey", "shiftKey"] as const) {
      expect(replayKey(key(" ", { [modifier]: true }))).toBeNull();
    }
  });

  it("leaves a key aimed at a control to the control", () => {
    expect(replayKey(key(" ", { target: inside(true) }))).toBeNull();
    expect(replayKey(key("ArrowLeft", { target: inside(false) }))).toEqual({ kind: "seek", byMs: -SEEK_STEP_MS });
  });
});
