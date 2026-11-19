import { describe, expect, it } from "vitest";
import { Ring } from "../ring";
import type { StripModel } from "../strip";
import type { Second } from "../timeline";
import { CANDLE_PITCH, CELL_WIDTH, drawStepChart, drawStrip, type Palette, type Pen } from "./draw";

const palette: Palette = { text: "text", grid: "grid", prefill: "prefill", decode: "decode", rise: "rise", fall: "fall" };

interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
  color: string;
}

function recorder(): Pen & { rects: Rect[]; texts: string[] } {
  const pen = {
    fillStyle: "",
    font: "",
    textBaseline: "alphabetic" as CanvasTextBaseline,
    rects: [] as Rect[],
    texts: [] as string[],
    clearRect() {
      pen.rects.length = 0;
      pen.texts.length = 0;
    },
    fillRect(x: number, y: number, w: number, h: number) {
      pen.rects.push({ x, y, w, h, color: String(pen.fillStyle) });
    },
    fillText(text: string) {
      pen.texts.push(text);
    },
  };
  return pen;
}

function second(s: number, ohlc: [number, number, number, number], over: Partial<Second> = {}): Second {
  const [open, high, low, close] = ohlc;
  return { second: s, open, high, low, close, steps: 1, processed: { prefill: 0, decode: 0 }, sampled: { prefill: 0, decode: 0 }, ...over };
}

function seconds(...items: Second[]): Ring<Second> {
  const r = new Ring<Second>(64);
  items.forEach((s) => r.push(s));
  return r;
}

const W = 400;
const H = 200;
/** Left to right. */
const byColor = (pen: { rects: Rect[] }, color: string) => pen.rects.filter((r) => r.color === color).sort((a, b) => a.x - b.x || a.y - b.y);

describe("drawStepChart", () => {
  it("draws the newest second at the right, and leaves a gap for a second with no steps", () => {
    const pen = recorder();
    drawStepChart(pen, seconds(second(5, [10, 20, 5, 15]), second(7, [10, 20, 5, 15])), W, H, palette);
    const wicks = byColor(pen, "rise").filter((r) => r.w === 1);
    expect(wicks).toHaveLength(2);
    const [older, newer] = wicks.map((r) => r.x);
    expect(newer).toBeGreaterThan(older!);
    expect(newer! - older!).toBe(2 * CANDLE_PITCH);
    expect(newer!).toBeGreaterThan(W - CANDLE_PITCH);
  });

  it("colours a candle by whether latency rose or fell in its second", () => {
    const pen = recorder();
    drawStepChart(pen, seconds(second(1, [10, 30, 5, 20]), second(2, [20, 30, 5, 10])), W, H, palette);
    expect(byColor(pen, "rise")).toHaveLength(2);
    expect(byColor(pen, "fall")).toHaveLength(2);
  });

  it("scales the candles so the highest wick in view reaches the top of its pane", () => {
    const pen = recorder();
    drawStepChart(pen, seconds(second(1, [10, 40, 0, 20]), second(2, [10, 80, 0, 20])), W, H, palette);
    const [half, full] = byColor(pen, "rise").filter((r) => r.w === 1);
    const base = full!.y + full!.h;
    expect(half!.y + half!.h).toBeCloseTo(base, 5);
    expect(half!.h / full!.h).toBeCloseTo(0.5, 5);
  });

  it("stacks prefill on decode in two bar graphs, processed and sampled", () => {
    const pen = recorder();
    const s = second(1, [10, 10, 10, 10], { processed: { prefill: 300, decode: 100 }, sampled: { prefill: 1, decode: 3 } });
    drawStepChart(pen, seconds(s), W, H, palette);
    const prefill = byColor(pen, "prefill");
    const decode = byColor(pen, "decode");
    expect(prefill).toHaveLength(2);
    expect(decode).toHaveLength(2);
    const [pProcessed, pSampled] = prefill;
    const [dProcessed, dSampled] = decode;
    expect(pProcessed!.y + pProcessed!.h).toBeCloseTo(dProcessed!.y, 5);
    expect(pProcessed!.h / dProcessed!.h).toBeCloseTo(3, 5);
    expect(dSampled!.h / pSampled!.h).toBeCloseTo(3, 5);
    expect(pSampled!.y).toBeGreaterThan(dProcessed!.y + dProcessed!.h);
  });

  it("labels each pane with its largest value in view", () => {
    const pen = recorder();
    const s = second(1, [10, 227.6, 5, 20], { processed: { prefill: 4000, decode: 123 }, sampled: { prefill: 1, decode: 7 } });
    drawStepChart(pen, seconds(s), W, H, palette);
    expect(pen.texts.join(" | ")).toMatch(/step ms.*228.*\|.*processed tok\/s.*4,123.*\|.*sampled tok\/s.*8/);
  });

  it("draws only the seconds that fit", () => {
    const pen = recorder();
    const many = Array.from({ length: 200 }, (_, i) => second(i, [1, 2, 1, 2]));
    drawStepChart(pen, seconds(...many.slice(-64)), 100, H, palette);
    const wicks = byColor(pen, "rise").filter((r) => r.w === 1);
    expect(wicks.length).toBe(Math.floor(100 / CANDLE_PITCH));
    expect(Math.min(...wicks.map((r) => r.x))).toBeGreaterThanOrEqual(0);
  });

  it("draws no candle and no bar without data", () => {
    const pen = recorder();
    drawStepChart(pen, seconds(), W, H, palette);
    expect(pen.rects.filter((r) => r.color !== "grid")).toEqual([]);
  });
});

describe("drawStrip", () => {
  it("draws one cell per request per step, at its column and lane, coloured by kind", () => {
    const pen = recorder();
    const model: StripModel = {
      lanes: 2,
      cells: [
        { column: 3, lane: 0, kind: "prefill" },
        { column: 4, lane: 1, kind: "decode" },
      ],
    };
    drawStrip(pen, model, W, H, palette);
    const [p] = byColor(pen, "prefill");
    const [d] = byColor(pen, "decode");
    expect(p).toMatchObject({ x: 3 * CELL_WIDTH, y: 0 });
    expect(d!.x).toBe(4 * CELL_WIDTH);
    expect(d!.y).toBeGreaterThan(p!.y);
    expect(d!.y + d!.h).toBeLessThanOrEqual(H);
  });
});
