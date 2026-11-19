import type { Ring } from "../ring";
import type { StripModel } from "../strip";
import type { Second, Split } from "../timeline";
import { int } from "./format";

export type Pen = Pick<CanvasRenderingContext2D, "fillStyle" | "font" | "textBaseline" | "clearRect" | "fillRect" | "fillText">;

export interface Palette {
  text: string;
  grid: string;
  prefill: string;
  decode: string;
  /** A candle whose latency closed above its open. */
  rise: string;
  fall: string;
}

export const CANDLE_PITCH = 7;
const BODY = CANDLE_PITCH - 2;
const LABEL = 14;
export const CELL_WIDTH = 4;
const MAX_LANE = 12;

interface Pane {
  top: number;
  bottom: number;
}

/** Candles of step latency, then processed and sampled tokens per second, one column per second. */
export function drawStepChart(pen: Pen, seconds: Ring<Second>, width: number, height: number, palette: Palette): void {
  pen.clearRect(0, 0, width, height);
  const candleHeight = Math.round(height / 2);
  const barHeight = Math.floor((height - candleHeight) / 2);
  const panes: Pane[] = [
    { top: 0, bottom: candleHeight },
    { top: candleHeight, bottom: candleHeight + barHeight },
    { top: candleHeight + barHeight, bottom: height },
  ];
  pen.fillStyle = palette.grid;
  for (const pane of panes.slice(1)) pen.fillRect(0, pane.top, width, 1);

  const newest = seconds.newest();
  const shown: { left: number; s: Second }[] = [];
  if (newest !== undefined) {
    for (const s of seconds.newestFirst()) {
      const left = width - (newest.second - s.second + 1) * CANDLE_PITCH;
      if (left < 0) break;
      shown.push({ left, s });
    }
  }
  const total = (split: Split) => split.prefill + split.decode;
  const maxMs = Math.max(0, ...shown.map(({ s }) => s.high));
  const maxProcessed = Math.max(0, ...shown.map(({ s }) => total(s.processed)));
  const maxSampled = Math.max(0, ...shown.map(({ s }) => total(s.sampled)));

  for (const { left, s } of shown) {
    const y = scale(panes[0]!, maxMs);
    pen.fillStyle = s.close > s.open ? palette.rise : palette.fall;
    pen.fillRect(left + Math.floor(BODY / 2), y(s.high), 1, y(s.low) - y(s.high));
    const top = y(Math.max(s.open, s.close));
    pen.fillRect(left, top, BODY, Math.max(1, y(Math.min(s.open, s.close)) - top));
    stack(pen, left, s.processed, scale(panes[1]!, maxProcessed), palette);
    stack(pen, left, s.sampled, scale(panes[2]!, maxSampled), palette);
  }

  pen.fillStyle = palette.text;
  pen.font = "11px ui-monospace, Menlo, monospace";
  pen.textBaseline = "top";
  const labels = ["step ms", "processed tok/s", "sampled tok/s"];
  [maxMs, maxProcessed, maxSampled].forEach((max, i) => {
    pen.fillText(shown.length ? `${labels[i]} · max ${int(max)}` : labels[i]!, 4, panes[i]!.top + 2);
  });
}

/** One row per lane, one column per step; the newest step at the right. */
export function drawStrip(pen: Pen, model: StripModel, width: number, height: number, palette: Palette): void {
  pen.clearRect(0, 0, width, height);
  const lane = Math.max(2, Math.min(MAX_LANE, Math.floor(height / Math.max(1, model.lanes))));
  for (const cell of model.cells) {
    pen.fillStyle = cell.kind === "prefill" ? palette.prefill : palette.decode;
    pen.fillRect(cell.column * CELL_WIDTH, cell.lane * lane, CELL_WIDTH - 1, lane - 1);
  }
}

/** Maps a value from 0 to `max` onto the pane, below its label. */
function scale(pane: Pane, max: number): (value: number) => number {
  const span = pane.bottom - pane.top - LABEL;
  return (value) => pane.bottom - (max > 0 ? (value / max) * span : 0);
}

function stack(pen: Pen, left: number, split: Split, y: (value: number) => number, palette: Palette): void {
  const decodeTop = y(split.decode);
  const prefillTop = y(split.decode + split.prefill);
  pen.fillStyle = palette.decode;
  if (split.decode > 0) pen.fillRect(left, decodeTop, BODY, y(0) - decodeTop);
  pen.fillStyle = palette.prefill;
  if (split.prefill > 0) pen.fillRect(left, prefillTop, BODY, decodeTop - prefillTop);
}
