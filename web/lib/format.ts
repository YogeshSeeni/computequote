/** Display formatting. Inputs are in base units (bytes, seconds, USD). */

const BIG_UNITS: ReadonlyArray<readonly [number, string]> = [
  [1e12, "T"],
  [1e9, "B"],
  [1e6, "M"],
  [1e3, "K"],
];

/** 3 significant digits with trailing zeros stripped: 6.21, 70.6, 405. */
function sig3(n: number): string {
  return String(Number(n.toPrecision(3)));
}

/** 1.23K / 4.56M / 7.89B / 1.2T. Values below 1000 keep 3 significant digits. */
export function fmtBig(n: number): string {
  if (!Number.isFinite(n)) return "–";
  const abs = Math.abs(n);
  for (const [value, suffix] of BIG_UNITS) {
    if (abs >= value) return `${sig3(n / value)}${suffix}`;
  }
  return sig3(n);
}

/** $12.3M for >= $1000, otherwise cents: $2.49. */
export function fmtUSD(n: number): string {
  if (!Number.isFinite(n)) return "–";
  const sign = n < 0 ? "-" : "";
  const abs = Math.abs(n);
  if (abs >= 1000) return `${sign}$${fmtBig(abs)}`;
  return `${sign}$${abs.toFixed(2)}`;
}

export function toGiB(bytes: number): number {
  return bytes / 2 ** 30;
}

/** 68.4 GiB (two decimals below 10 GiB). */
export function fmtGiB(bytes: number): string {
  const g = toGiB(bytes);
  if (!Number.isFinite(g)) return "–";
  return `${g < 10 ? g.toFixed(2) : g.toFixed(1)} GiB`;
}

/** "45.2 s", "12.5 min", "2h 15m", "3d 4h". */
export function fmtDuration(s: number): string {
  if (!Number.isFinite(s)) return "–";
  if (s < 60) return `${sig3(s)} s`;
  if (s < 3600) return `${sig3(s / 60)} min`;
  if (s < 86_400) {
    const mins = Math.round(s / 60);
    return `${Math.floor(mins / 60)}h ${mins % 60}m`;
  }
  const hours = Math.round(s / 3600);
  return `${Math.floor(hours / 24)}d ${hours % 24}h`;
}

/** Seconds with 3 significant digits: "2.35 s", "0.0123 s". */
export function fmtSeconds(s: number): string {
  if (!Number.isFinite(s)) return "–";
  return `${sig3(s)} s`;
}

/** Fraction in [0, 1] as a percentage: 0.421 -> "42.1%". */
export function fmtPct(fraction: number, digits = 1): string {
  if (!Number.isFinite(fraction)) return "–";
  return `${(fraction * 100).toFixed(digits)}%`;
}

/** Thousands-separated integer. */
export function fmtInt(n: number): string {
  if (!Number.isFinite(n)) return "–";
  return Math.round(n).toLocaleString("en-US");
}

/** Decimal-as-string from the API -> number (NaN if missing). */
export function parseDecimal(s: string | null | undefined): number {
  return s == null ? NaN : Number(s);
}
