import type { MemoryBreakdown } from "@/lib/api";
import { fmtGiB, fmtPct } from "@/lib/format";

const SEGMENTS = [
  { key: "weights_bytes", label: "Weights", cls: "bg-mem-weights" },
  { key: "grads_bytes", label: "Gradients", cls: "bg-mem-grads" },
  { key: "optimizer_bytes", label: "Optimizer", cls: "bg-mem-optimizer" },
  { key: "activation_bytes", label: "Activations", cls: "bg-mem-activations" },
  { key: "logits_bytes", label: "Logits", cls: "bg-mem-logits" },
  { key: "overhead_bytes", label: "Overhead", cls: "bg-mem-overhead" },
] as const;

/** Horizontal stacked bar of per-GPU memory against capacity. Plain divs. */
export function MemoryBar({ memory }: { memory: MemoryBreakdown }) {
  const { total_bytes: total, capacity_bytes: capacity } = memory;
  const over = total > capacity;
  // When the config does not fit, scale the track to the total so every
  // segment stays visible and draw the capacity as a dashed tick.
  const scale = Math.max(total, capacity, 1);
  const pct = (bytes: number) => (bytes / scale) * 100;

  return (
    <div>
      <div
        className="relative h-7 w-full overflow-hidden rounded-md bg-surface-2"
        role="img"
        aria-label={`Memory ${fmtGiB(total)} of ${fmtGiB(capacity)} capacity`}
      >
        <div className="flex h-full">
          {SEGMENTS.map((s) => {
            const bytes = memory[s.key];
            if (bytes <= 0) return null;
            return (
              <div
                key={s.key}
                className={`${s.cls} h-full`}
                style={{ width: `${pct(bytes)}%` }}
                title={`${s.label}: ${fmtGiB(bytes)}`}
              />
            );
          })}
        </div>
        {over && (
          <div
            className="absolute inset-y-0 border-l-2 border-dashed border-ink"
            style={{ left: `${pct(capacity)}%` }}
            title={`Capacity ${fmtGiB(capacity)}`}
          />
        )}
      </div>
      <div className="num mt-1 flex justify-between text-xs text-ink-3">
        <span>0</span>
        <span>{over ? `track ${fmtGiB(scale)} · dashed = capacity` : fmtGiB(capacity)}</span>
      </div>

      <dl className="mt-3 grid grid-cols-1 gap-x-5 gap-y-1.5 min-[480px]:grid-cols-2 sm:grid-cols-3">
        {SEGMENTS.map((s) => (
          <div key={s.key} className="flex items-center gap-2 text-sm">
            <span className={`h-2.5 w-2.5 shrink-0 rounded-sm ${s.cls}`} />
            <dt className="truncate text-ink-2">{s.label}</dt>
            <dd className="num ml-auto whitespace-nowrap">{fmtGiB(memory[s.key])}</dd>
          </div>
        ))}
      </dl>

      <p className="mt-3 flex items-baseline justify-between border-t border-line pt-2 text-sm">
        <span className="text-ink-2">Total per GPU</span>
        <span className={`num ${over ? "text-danger" : ""}`}>
          {fmtGiB(total)} / {fmtGiB(capacity)} ({fmtPct(total / capacity, 0)})
          {over && ` · over by ${fmtGiB(total - capacity)}`}
        </span>
      </p>
    </div>
  );
}
