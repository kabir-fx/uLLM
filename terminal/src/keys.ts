export const SEEK_STEP_MS = 5000;

/** Controls that keep their own keys: a form control, a button, a tab (decision 0027). */
const OWN_KEYS = "input, select, textarea, button, [role=tab]";

/** The part of an event target the mapping reads; an `Element` has it. */
export interface KeyTarget {
  closest(selector: string): unknown;
}

export type ReplayAction = { kind: "toggle" } | { kind: "seek"; byMs: number };

/** What a key pressed in replay mode does, or null to leave it to the browser. */
export function replayKey(event: {
  key: string;
  altKey: boolean;
  ctrlKey: boolean;
  metaKey: boolean;
  shiftKey: boolean;
  target: KeyTarget | null;
}): ReplayAction | null {
  if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return null;
  if (event.target?.closest(OWN_KEYS)) return null;
  switch (event.key) {
    case " ":
      return { kind: "toggle" };
    case "ArrowLeft":
      return { kind: "seek", byMs: -SEEK_STEP_MS };
    case "ArrowRight":
      return { kind: "seek", byMs: SEEK_STEP_MS };
    default:
      return null;
  }
}
