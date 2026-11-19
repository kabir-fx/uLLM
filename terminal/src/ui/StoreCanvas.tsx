import { useEffect, useRef } from "react";
import type { TelemetryStore } from "../store";
import type { Palette, Pen } from "./draw";

export type Draw = (pen: Pen, store: TelemetryStore, width: number, height: number, palette: Palette) => void;

/**
 * React keeps one canvas element and never draws on it. A loop on `requestAnimationFrame` draws
 * when the store's version or the canvas size changed (T1). `draw` must be stable, a
 * module-level function, or the loop restarts each time the panel re-renders.
 */
export function StoreCanvas({ store, draw }: { store: TelemetryStore; draw: Draw }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current!;
    const pen = canvas.getContext("2d")!;
    const palette = readPalette(canvas);
    let width = 0;
    let height = 0;
    let drawn = -1;
    const resize = new ResizeObserver(() => {
      width = canvas.clientWidth;
      height = canvas.clientHeight;
      canvas.width = Math.round(width * devicePixelRatio);
      canvas.height = Math.round(height * devicePixelRatio);
      drawn = -1;
    });
    resize.observe(canvas);
    let frame = requestAnimationFrame(function tick() {
      if (width > 0 && height > 0 && drawn !== store.version) {
        drawn = store.version;
        pen.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
        draw(pen, store, width, height, palette);
      }
      frame = requestAnimationFrame(tick);
    });
    return () => {
      cancelAnimationFrame(frame);
      resize.disconnect();
    };
  }, [store, draw]);
  return <canvas ref={ref} />;
}

function readPalette(el: Element): Palette {
  const css = getComputedStyle(el);
  const get = (name: string) => css.getPropertyValue(name).trim();
  return {
    text: get("--dim"),
    grid: get("--line"),
    prefill: get("--prefill"),
    decode: get("--decode"),
    rise: get("--rise"),
    fall: get("--fall"),
  };
}
