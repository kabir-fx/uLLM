export const ROW_HEIGHT = 20;

/** The rows to render for a scroll position: `start` inclusive, `end` exclusive. */
export function visibleRange(scrollTop: number, viewport: number, count: number, rowHeight = ROW_HEIGHT, overscan = 5): { start: number; end: number } {
  const first = Math.floor(scrollTop / rowHeight);
  const last = Math.ceil((scrollTop + viewport) / rowHeight);
  return { start: Math.max(0, first - overscan), end: Math.min(count, last + overscan) };
}
