// Compile-time checks on the generated telemetry types. `pnpm typecheck` fails if
// the generator stops producing a discriminated union on `type`, which every
// consumer of the stream relies on.
import type { Frame, Step } from "./telemetry";

type EventType = Frame["events"][number]["type"];

export function stepRows(frame: Frame): string[] {
  return frame.events.flatMap((event) => (event.type === "step" ? event.rids : []));
}

export function isStep(event: Frame["events"][number]): event is Step {
  return event.type === "step";
}

// Exhaustiveness: adding an event type without handling it here is a type error.
export function describe(type: EventType): string {
  switch (type) {
    case "hello":
    case "req.arrived":
    case "req.admitted":
    case "req.first_token":
    case "req.finished":
    case "req.aborted":
    case "req.preempted":
    case "step":
    case "kv.snapshot":
    case "router.decision":
      return type;
    default: {
      const unhandled: never = type;
      return unhandled;
    }
  }
}
