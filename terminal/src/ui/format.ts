export const NONE = "—";

export function ms(value: number | null | undefined): string {
  if (value === null || value === undefined) return NONE;
  return value >= 1_000 ? `${(value / 1_000).toFixed(2)} s` : `${value.toFixed(value < 10 ? 1 : 0)} ms`;
}

export function int(value: number | null | undefined): string {
  return value === null || value === undefined ? NONE : Math.round(value).toLocaleString("en-US");
}

export function percent(share: number | null | undefined): string {
  return share === null || share === undefined ? NONE : `${Math.round(share * 100)}%`;
}
