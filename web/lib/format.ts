export function pct(value: number | null | undefined): string {
  if (value == null) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

export function fmtMae(value: number | null | undefined): string {
  if (value == null) return "—";
  return value.toFixed(3);
}

export function fmtMs(value: number | null | undefined): string {
  if (value == null) return "—";
  return `${Math.round(value)}ms`;
}
