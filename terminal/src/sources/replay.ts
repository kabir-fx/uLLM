import type { Frame } from "../contracts/telemetry";
import type { TelemetrySource } from "../source";

export interface ReplayOptions {
  speed?: number;
}

/** Plays a recorded trace, each frame at its own `t` from the first, divided by `speed`. */
export function replaySource(
  label: string,
  load: () => Promise<Frame[]>,
  { speed = 1 }: ReplayOptions = {},
): TelemetrySource {
  return {
    label,
    start(sink) {
      let stopped = false;
      let timer: ReturnType<typeof setTimeout> | undefined;
      sink.status({ kind: "connecting" });
      load().then(
        (frames) => {
          if (stopped) return;
          sink.status({ kind: "open" });
          const t0 = frames[0]?.t ?? 0;
          const wall0 = performance.now();
          let next = 0;
          // Each tick sends every frame that is due, then waits for the next one.
          // Due times come from the start, so timer lateness never accumulates.
          const tick = () => {
            const elapsed = (performance.now() - wall0) * speed;
            try {
              while (next < frames.length && frames[next]!.t - t0 <= elapsed) sink.frame(frames[next++]!);
            } catch (error) {
              sink.status({ kind: "error", message: (error as Error).message });
              return;
            }
            if (next === frames.length) {
              sink.status({ kind: "ended" });
              return;
            }
            timer = setTimeout(tick, (frames[next]!.t - t0 - elapsed) / speed);
          };
          tick();
        },
        (error: unknown) => {
          if (!stopped) sink.status({ kind: "error", message: (error as Error).message });
        },
      );
      return () => {
        stopped = true;
        clearTimeout(timer);
      };
    },
  };
}
