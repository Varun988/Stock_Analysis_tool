import type { ProviderHealthResponse } from "@/lib/api";

type ProviderHealthListProps = {
  providerHealth: ProviderHealthResponse;
};

function getProviderTone(status: string): "success" | "warning" | "neutral" {
  if (["AVAILABLE", "CONFIGURED"].includes(status)) {
    return "success";
  }

  if (
    [
      "API_KEY_MISSING",
      "AVAILABLE_WITH_RATE_LIMIT_RISK",
      "IMPLEMENTED_WITH_RATE_LIMIT_RISK",
    ].includes(status)
  ) {
    return "warning";
  }

  return "neutral";
}

export function ProviderHealthList({
  providerHealth,
}: ProviderHealthListProps) {
  return (
    <section className="rounded-2xl border border-slate-700 bg-slate-900 p-6">
      <div>
        <p className="text-sm uppercase tracking-wide text-sky-300">
          Market data providers
        </p>
        <h2 className="mt-2 text-2xl font-semibold">Provider Health</h2>
        <p className="mt-2 text-sm text-slate-400">
          Shows which market data sources are configured and available.
        </p>
      </div>

      <div className="mt-6 grid gap-4 md:grid-cols-2">
        {Object.entries(providerHealth.data).map(([providerName, provider]) => {
          const tone = getProviderTone(provider.status);

          return (
            <div
              key={providerName}
              className="rounded-xl border border-slate-700 bg-slate-800 p-5"
            >
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="text-lg font-semibold">{providerName}</p>
                  <p className="mt-2 text-sm text-slate-400">
                    {provider.description}
                  </p>
                </div>

                <span
                  className={`rounded-full border px-3 py-1 text-xs ${
                    tone === "success"
                      ? "border-emerald-500/40 bg-emerald-950 text-emerald-200"
                      : tone === "warning"
                        ? "border-amber-500/40 bg-amber-950 text-amber-200"
                        : "border-slate-600 bg-slate-700 text-slate-200"
                  }`}
                >
                  {provider.status}
                </span>
              </div>

              <p className="mt-4 text-sm text-slate-400">
                Configured:{" "}
                <span className="font-medium text-slate-200">
                  {provider.configured ? "Yes" : "No"}
                </span>
              </p>
            </div>
          );
        })}
      </div>
    </section>
  );
}
