import type { Frame } from "../contracts/telemetry";
import type { Sink, TelemetrySource } from "../source";

export interface ReplayOptions {
  speed?: number;
}

/** A recorded trace, with play, pause, speed and seek (T4, decision 0026). */
export interface ReplaySource extends TelemetrySource {
  readonly playing: boolean;
  readonly speed: number;
  /** Trace length in ms, from the first frame to the last; 0 until the trace loads. */
  readonly duration: number;
  /** Trace ms from the first frame. */
  position(): number;
  /** At the end, starts again from 0. */
  play(): void;
  pause(): void;
  setSpeed(speed: number): void;
  /** Clamped to the trace. */
  seek(ms: number): void;
}

/**
 * Plays a recorded trace, each frame at its own `t` from the first, divided by the speed.
 * The position is `base` plus the wall time since `wall`, times the speed. Every control
 * takes a new base, so the position never jumps and timer lateness never accumulates.
 */
export function replaySource(
  label: string,
  load: () => Promise<Frame[]>,
  { speed = 1 }: ReplayOptions = {},
): ReplaySource {
  let frames: Frame[] = [];
  let t0 = 0;
  let sink: Sink | null = null;
  let loaded = false;
  let next = 0;
  let playing = true;
  let rate = speed;
  let base = 0;
  let wall = 0;
  let timer: ReturnType<typeof setTimeout> | undefined;

  const duration = () => (loaded ? frames.at(-1)!.t - t0 : 0);
  const position = () => (playing ? Math.min(base + (performance.now() - wall) * rate, duration()) : base);
  const rebase = (at: number) => {
    base = at;
    wall = performance.now();
  };

  /** Sends every frame due by `at`. False when the sink threw: the replay is then over. */
  const deliver = (at: number): boolean => {
    try {
      while (next < frames.length && frames[next]!.t - t0 <= at) sink!.frame(frames[next++]!);
      return true;
    } catch (error) {
      clearTimeout(timer);
      playing = false;
      const out = sink!;
      sink = null;
      out.status({ kind: "error", message: (error as Error).message });
      return false;
    }
  };

  const tick = () => {
    clearTimeout(timer);
    if (!sink || !deliver(position())) return;
    if (next === frames.length) {
      rebase(duration());
      playing = false;
      sink.status({ kind: "ended" });
      return;
    }
    if (playing) timer = setTimeout(tick, (frames[next]!.t - t0 - position()) / rate);
  };

  const live = () => sink !== null && loaded;

  return {
    label,
    get playing() {
      return playing;
    },
    get speed() {
      return rate;
    },
    get duration() {
      return duration();
    },
    position,
    start(out) {
      sink = out;
      out.status({ kind: "connecting" });
      load().then(
        (trace) => {
          if (sink !== out) return;
          frames = trace;
          t0 = frames[0]?.t ?? 0;
          loaded = frames.length > 0;
          rebase(0);
          out.status({ kind: playing ? "open" : "paused" });
          if (!loaded) out.status({ kind: "ended" });
          else if (playing) tick();
          else deliver(0);
        },
        (error: unknown) => {
          if (sink === out) out.status({ kind: "error", message: (error as Error).message });
        },
      );
      return () => {
        if (sink === out) sink = null;
        clearTimeout(timer);
      };
    },
    play() {
      if (playing) return;
      if (!live()) {
        playing = true;
        return;
      }
      if (next === frames.length) {
        next = 0;
        rebase(0);
      } else rebase(position());
      playing = true;
      sink!.status({ kind: "open" });
      tick();
    },
    pause() {
      if (!playing) return;
      if (!live()) {
        playing = false;
        return;
      }
      rebase(position());
      playing = false;
      clearTimeout(timer);
      sink!.status({ kind: "paused" });
    },
    setSpeed(speed) {
      if (!(speed > 0 && Number.isFinite(speed))) throw new Error(`speed must be a positive number, not ${speed}`);
      rebase(position());
      rate = speed;
      if (live() && playing) tick();
    },
    seek(ms) {
      if (!live()) return;
      const at = Math.min(Math.max(ms, 0), duration());
      // The store cannot take frames back; it rebuilds from the trace's hello (C2).
      if (next > 0 && frames[next - 1]!.t - t0 > at) next = 0;
      rebase(at);
      if (playing) {
        tick();
        return;
      }
      if (!deliver(at)) return;
      sink!.status(next === frames.length ? { kind: "ended" } : { kind: "paused" });
    },
  };
}
