"use client";

import {
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import {
  describeError,
  estimateTrain,
  fetchPresets,
  isAbortError,
  type EstimateResponse,
  type GPUSpec,
  type PresetsResponse,
} from "@/lib/api";
import {
  derive,
  formFromQuery,
  queryFromForm,
  toRequest,
  type ScenarioForm,
} from "@/lib/scenario";
import { ScenarioFormPanel } from "@/components/ScenarioForm";
import { Results } from "@/components/Results";

// useSearchParams() must sit under a Suspense boundary so the static build
// can prerender the shell and fill the form in on the client.
export default function Page() {
  return (
    <Suspense fallback={<Skeleton />}>
      <Calculator />
    </Suspense>
  );
}

const LAYOUT = "grid gap-6 lg:grid-cols-[360px_minmax(0,1fr)] lg:gap-8";

function Skeleton() {
  return (
    <div className={LAYOUT}>
      <div className="h-[520px] rounded-lg border border-line bg-surface" />
      <div className="h-[260px] rounded-lg border border-dashed border-line" />
    </div>
  );
}

function Calculator() {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  // Form state is initialised from the URL once; after that the URL follows
  // the form (see the mirror effect below), never the other way round.
  const [form, setForm] = useState<ScenarioForm>(() =>
    formFromQuery(searchParams),
  );
  const [presets, setPresets] = useState<PresetsResponse | null>(null);
  const [presetsError, setPresetsError] = useState<string | null>(null);
  const [result, setResult] = useState<EstimateResponse | null>(null);
  const [resultQuery, setResultQuery] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  // Presets (model/GPU lists). Failure is non-fatal: the form keeps working
  // with the slugs it already has.
  useEffect(() => {
    const ac = new AbortController();
    fetchPresets(ac.signal)
      .then(setPresets)
      .catch((e: unknown) => {
        if (!isAbortError(e)) setPresetsError(describeError(e));
      });
    return () => ac.abort();
  }, []);

  // Mirror every field into the query string (debounced, no scroll).
  const query = useMemo(() => queryFromForm(form), [form]);
  useEffect(() => {
    if (query === searchParams.toString()) return;
    const t = window.setTimeout(() => {
      router.replace(`${pathname}?${query}`, { scroll: false });
    }, 150);
    return () => window.clearTimeout(t);
  }, [query, pathname, router, searchParams]);

  const gpuSpec: GPUSpec | undefined = presets?.gpus.find(
    (g) => g.slug === form.gpu,
  );
  const derived = useMemo(() => derive(form, gpuSpec), [form, gpuSpec]);

  const runEstimate = useCallback(async (f: ScenarioForm, gpu?: GPUSpec) => {
    const req = toRequest(f, derive(f, gpu));
    if (!req) {
      setError("Fix the highlighted fields first.");
      return;
    }
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    setLoading(true);
    setError(null);
    try {
      const res = await estimateTrain(req, ac.signal);
      setResult(res);
      setResultQuery(queryFromForm(f));
    } catch (e: unknown) {
      if (isAbortError(e)) return;
      setError(describeError(e));
    } finally {
      if (abortRef.current === ac) setLoading(false);
    }
  }, []);

  // Auto-run once on first load when the URL carries a scenario.
  const autoRun = useRef(searchParams.has("model"));
  useEffect(() => {
    if (!autoRun.current) return;
    autoRun.current = false;
    void runEstimate(form);
  }, [form, runEstimate]);

  const update = (patch: Partial<ScenarioForm>) =>
    setForm((f) => ({ ...f, ...patch }));

  const stale = result !== null && resultQuery !== null && resultQuery !== query;

  const getShareUrl = () => `${window.location.origin}${pathname}?${query}`;

  return (
    <div className={LAYOUT}>
      <ScenarioFormPanel
        form={form}
        derived={derived}
        presets={presets}
        presetsError={presetsError}
        gpuSpec={gpuSpec}
        loading={loading}
        onChange={update}
        onSubmit={() => void runEstimate(form, gpuSpec)}
      />
      <Results
        result={result}
        loading={loading}
        error={error}
        stale={stale}
        getShareUrl={getShareUrl}
      />
    </div>
  );
}
