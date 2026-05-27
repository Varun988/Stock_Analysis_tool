import type { AIProviderStatusResponse } from "@/lib/api";

type AIProviderStatusCardProps = {
  aiStatus: AIProviderStatusResponse;
};

function getStatusTone(status: string) {
  if (status === "AVAILABLE") {
    return "border-emerald-500/40 bg-emerald-950 text-emerald-200";
  }

  if (status.includes("MISSING")) {
    return "border-amber-500/40 bg-amber-950 text-amber-200";
  }

  return "border-slate-600 bg-slate-700 text-slate-200";
}

export function AIProviderStatusCard({ aiStatus }: AIProviderStatusCardProps) {
  const configuredProvider = aiStatus.data.configured_provider;
  const currentProvider = aiStatus.data.providers[configuredProvider];

  return (
    <section className="rounded-2xl border border-purple-500/30 bg-slate-900 p-6">
      <div>
        <p className="text-sm uppercase tracking-wide text-purple-300">
          AI Explanation Provider
        </p>
        <h2 className="mt-2 text-2xl font-semibold">AI Provider Status</h2>
        <p className="mt-2 text-sm text-slate-400">
          Shows which AI provider is currently selected for recommendation
          explanations.
        </p>
      </div>

      <div className="mt-6 grid gap-4 md:grid-cols-3">
        <div className="rounded-xl border border-slate-700 bg-slate-800 p-5">
          <p className="text-sm text-slate-400">Current Provider</p>
          <p className="mt-2 text-2xl font-semibold text-purple-200">
            {configuredProvider}
          </p>
        </div>

        <div className="rounded-xl border border-slate-700 bg-slate-800 p-5">
          <p className="text-sm text-slate-400">Status</p>
          <span
            className={`mt-3 inline-flex rounded-full border px-3 py-1 text-sm ${getStatusTone(
              currentProvider?.status ?? "UNKNOWN",
            )}`}
          >
            {currentProvider?.status ?? "UNKNOWN"}
          </span>
        </div>

        <div className="rounded-xl border border-slate-700 bg-slate-800 p-5">
          <p className="text-sm text-slate-400">Model</p>
          <p className="mt-2 text-lg font-semibold text-slate-100">
            {currentProvider?.model ?? "Not applicable"}
          </p>
        </div>
      </div>

      <div className="mt-5 rounded-xl border border-slate-700 bg-slate-800 p-5">
        <p className="text-sm text-slate-400">Description</p>
        <p className="mt-2 text-sm leading-6 text-slate-300">
          {currentProvider?.description ?? "No provider details available."}
        </p>
      </div>
    </section>
  );
}
