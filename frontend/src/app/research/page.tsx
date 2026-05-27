"use client";

import { FormEvent, useEffect, useState } from "react";

type ResearchSource = {
  title: string;
  url?: string | null;
  source_name?: string | null;
  published_at?: string | null;
  snippet?: string | null;
  position?: number | null;
};

type ResearchContext = {
  query: string;
  subject_type: string;
  subject_id?: string | null;
  summary: string;
  key_points: string[];
  sources: ResearchSource[];
  risk_note: string;
  provider: string;
  summarizer: string;
};

type ResearchProviderStatus = {
  configured_provider: string;
  use_gemini_summary: boolean;
  providers: Record<
    string,
    {
      configured: boolean;
      status: string;
      description?: string;
      base_url?: string;
      country?: string;
      language?: string;
      result_count?: number;
    }
  >;
  summarizer: Record<
    string,
    {
      configured: boolean;
      status: string;
      model?: string;
    }
  >;
};

type ApiResponse<T> = {
  success?: boolean;
  message?: string;
  data?: T;
  detail?: string;
};

export default function ResearchPage() {
  const [query, setQuery] = useState("NIFTYBEES ETF India latest news");
  const [subjectType, setSubjectType] = useState("GENERAL");
  const [useLlmSummary, setUseLlmSummary] = useState(true);

  const [researchContext, setResearchContext] =
    useState<ResearchContext | null>(null);
  const [providerStatus, setProviderStatus] =
    useState<ResearchProviderStatus | null>(null);

  const [isLoadingResearch, setIsLoadingResearch] = useState(false);
  const [isLoadingStatus, setIsLoadingStatus] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  async function loadProviderStatus() {
    setIsLoadingStatus(true);
    setErrorMessage(null);

    try {
      const response = await fetch("/api/research/providers/status", {
        cache: "no-store",
      });

      const result =
        (await response.json()) as ApiResponse<ResearchProviderStatus>;

      if (!response.ok) {
        throw new Error(result.detail ?? "Failed to load research provider status.");
      }

      if (!result.data) {
        throw new Error("Research provider status response did not contain data.");
      }

      setProviderStatus(result.data);
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "Unknown provider status error."
      );
    } finally {
      setIsLoadingStatus(false);
    }
  }

  async function loadIndiaMarketResearch() {
    setIsLoadingResearch(true);
    setErrorMessage(null);
    setResearchContext(null);

    try {
      const response = await fetch(
        `/api/research/market/india?use_llm_summary=${useLlmSummary}`,
        {
          cache: "no-store",
        }
      );

      const result = (await response.json()) as ApiResponse<ResearchContext>;

      if (!response.ok) {
        throw new Error(result.detail ?? "Failed to load India market research.");
      }

      if (!result.data) {
        throw new Error("India market research response did not contain data.");
      }

      setResearchContext(result.data);
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "Unknown research error."
      );
    } finally {
      setIsLoadingResearch(false);
    }
  }

  async function handleCustomResearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!query.trim()) {
      setErrorMessage("Please enter a research query.");
      return;
    }

    setIsLoadingResearch(true);
    setErrorMessage(null);
    setResearchContext(null);

    try {
      const response = await fetch("/api/research/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query,
          subject_type: subjectType,
          subject_id: null,
          use_llm_summary: useLlmSummary,
        }),
      });

      const result = (await response.json()) as ApiResponse<ResearchContext>;

      if (!response.ok) {
        throw new Error(result.detail ?? "Failed to load custom research.");
      }

      if (!result.data) {
        throw new Error("Custom research response did not contain data.");
      }

      setResearchContext(result.data);
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "Unknown research error."
      );
    } finally {
      setIsLoadingResearch(false);
    }
  }

  useEffect(() => {
    void loadProviderStatus();
  }, []);

  return (
    <main className="min-h-screen bg-slate-950 px-6 py-10 text-white">
      <section className="mx-auto max-w-7xl space-y-8">
        <section className="rounded-2xl border border-emerald-500/30 bg-slate-900 p-8 shadow-xl">
          <p className="text-sm uppercase tracking-wide text-emerald-300">
            Research Context
          </p>

          <div className="mt-3 flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
            <div>
              <h1 className="text-4xl font-bold tracking-tight">
                Market and Instrument Research
              </h1>

              <p className="mt-4 max-w-3xl text-slate-300">
                Search market or instrument context using the backend research
                engine. Research is informational only and does not override the
                recommendation engine.
              </p>
            </div>

            <div className="rounded-full border border-emerald-500/30 bg-emerald-950/40 px-4 py-2 text-xs font-medium text-emerald-200">
              Context Layer
            </div>
          </div>

          {providerStatus && (
            <div className="mt-6 grid gap-4 md:grid-cols-3">
              <StatusCard
                label="Configured Provider"
                value={providerStatus.configured_provider}
              />
              <StatusCard
                label="Gemini Summary"
                value={providerStatus.use_gemini_summary ? "Enabled" : "Disabled"}
              />
              <StatusCard
                label="Research Status"
                value={
                  providerStatus.providers[providerStatus.configured_provider]
                    ?.status ?? "Unknown"
                }
              />
            </div>
          )}

          {isLoadingStatus && (
            <p className="mt-4 text-sm text-slate-400">
              Loading research provider status...
            </p>
          )}
        </section>

        <section className="grid gap-6 lg:grid-cols-[1fr_360px]">
          <section className="rounded-2xl border border-slate-700 bg-slate-900 p-6 shadow-xl">
            <h2 className="text-2xl font-semibold">Custom Research Query</h2>
            <p className="mt-2 text-sm text-slate-400">
              Enter a query such as “NIFTYBEES ETF India latest news” or “Indian
              mutual fund market latest context”.
            </p>

            <form onSubmit={handleCustomResearch} className="mt-6 space-y-4">
              <div>
                <label className="text-sm font-medium text-slate-300">
                  Research query
                </label>
                <input
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 p-3 text-sm text-slate-100 outline-none focus:border-emerald-400"
                  placeholder="Enter research query"
                />
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <label className="text-sm font-medium text-slate-300">
                    Subject type
                  </label>
                  <select
                    value={subjectType}
                    onChange={(event) => setSubjectType(event.target.value)}
                    className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 p-3 text-sm text-slate-100 outline-none focus:border-emerald-400"
                  >
                    <option value="GENERAL">GENERAL</option>
                    <option value="INSTRUMENT">INSTRUMENT</option>
                    <option value="MARKET">MARKET</option>
                    <option value="SECTOR">SECTOR</option>
                    <option value="FUND">FUND</option>
                    <option value="NEWS">NEWS</option>
                  </select>
                </div>

                <label className="mt-7 flex items-center gap-3 rounded-lg border border-slate-700 bg-slate-950 p-3 text-sm text-slate-200">
                  <input
                    type="checkbox"
                    checked={useLlmSummary}
                    onChange={(event) => setUseLlmSummary(event.target.checked)}
                    className="h-4 w-4"
                  />
                  Use Gemini summary when available
                </label>
              </div>

              <div className="flex flex-wrap gap-3">
                <button
                  type="submit"
                  disabled={isLoadingResearch}
                  className="rounded-lg bg-emerald-500 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
                >
                  {isLoadingResearch ? "Loading..." : "Run Research"}
                </button>

                <button
                  type="button"
                  onClick={loadIndiaMarketResearch}
                  disabled={isLoadingResearch}
                  className="rounded-lg border border-slate-600 px-5 py-3 text-sm font-semibold text-slate-100 transition hover:border-emerald-400 hover:text-emerald-300 disabled:cursor-not-allowed disabled:border-slate-800 disabled:text-slate-600"
                >
                  India Market Context
                </button>
              </div>
            </form>

            {errorMessage && (
              <div className="mt-6 rounded-xl border border-red-500/40 bg-red-950/40 p-4 text-sm text-red-100">
                <p className="font-semibold text-red-200">Something went wrong</p>
                <p className="mt-1">{errorMessage}</p>
              </div>
            )}
          </section>

          <section className="rounded-2xl border border-slate-700 bg-slate-900 p-6 shadow-xl">
            <h2 className="text-2xl font-semibold">Safety Boundary</h2>
            <div className="mt-4 space-y-3 text-sm text-slate-300">
              <p>Research is a context layer only.</p>
              <p>It does not decide where to invest.</p>
              <p>It should not be treated as a buy/sell signal.</p>
              <p>The recommendation engine remains the decision layer.</p>
            </div>
          </section>
        </section>

        {researchContext && <ResearchResultPanel research={researchContext} />}
      </section>
    </main>
  );
}

function StatusCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-slate-700 bg-slate-950/60 p-5">
      <p className="text-sm text-slate-400">{label}</p>
      <p className="mt-2 text-2xl font-bold text-emerald-300">{value}</p>
    </div>
  );
}

function ResearchResultPanel({ research }: { research: ResearchContext }) {
  return (
    <section className="rounded-2xl border border-slate-700 bg-slate-900 p-8 shadow-xl">
      <div className="flex flex-col justify-between gap-4 lg:flex-row lg:items-start">
        <div>
          <p className="text-sm uppercase tracking-wide text-slate-400">
            Research Result
          </p>
          <h2 className="mt-2 text-2xl font-semibold text-white">
            {research.subject_type} Context
          </h2>
          <p className="mt-2 text-sm text-slate-400">
            Query:{" "}
            <span className="font-medium text-emerald-300">
              {research.query}
            </span>
          </p>
        </div>

        <div className="flex flex-wrap gap-2">
          <Badge label={`Provider: ${research.provider}`} />
          <Badge label={`Summarizer: ${research.summarizer}`} />
        </div>
      </div>

      <div className="mt-6 rounded-xl border border-slate-700 bg-slate-950/60 p-5">
        <h3 className="text-lg font-semibold text-white">Summary</h3>
        <p className="mt-3 leading-7 text-slate-300">{research.summary}</p>
      </div>

      <div className="mt-6 rounded-xl border border-slate-700 bg-slate-950/60 p-5">
        <h3 className="text-lg font-semibold text-white">Key Points</h3>
        <ul className="mt-3 list-disc space-y-2 pl-5 text-slate-300">
          {research.key_points.map((point) => (
            <li key={point}>{point}</li>
          ))}
        </ul>
      </div>

      <div className="mt-6 rounded-xl border border-amber-500/30 bg-amber-950/30 p-5">
        <h3 className="text-lg font-semibold text-amber-100">Risk Note</h3>
        <p className="mt-3 leading-7 text-amber-100">{research.risk_note}</p>
      </div>

      <SourcesPanel sources={research.sources} />
    </section>
  );
}

function Badge({ label }: { label: string }) {
  return (
    <span className="rounded-full border border-emerald-500/30 bg-emerald-950/40 px-3 py-1 text-xs font-medium text-emerald-200">
      {label}
    </span>
  );
}

function SourcesPanel({ sources }: { sources: ResearchSource[] }) {
  if (sources.length === 0) {
    return (
      <div className="mt-6 rounded-xl border border-slate-700 bg-slate-950/60 p-5 text-sm text-slate-400">
        No sources were returned.
      </div>
    );
  }

  return (
    <div className="mt-6 overflow-hidden rounded-xl border border-slate-700">
      <div className="border-b border-slate-700 bg-slate-950/80 px-5 py-4">
        <h3 className="text-lg font-semibold text-white">Sources</h3>
        <p className="mt-1 text-sm text-slate-400">
          Review source links directly before relying on article content.
        </p>
      </div>

      <div className="divide-y divide-slate-800 bg-slate-900">
        {sources.map((source, index) => (
          <article key={`${source.title}-${index}`} className="p-5">
            <div className="flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
              <div>
                <p className="text-sm text-slate-500">
                  #{source.position ?? index + 1}{" "}
                  {source.source_name ? `• ${source.source_name}` : ""}
                  {source.published_at ? ` • ${source.published_at}` : ""}
                </p>

                {source.url ? (
                  <a
                    href={source.url}
                    target="_blank"
                    rel="noreferrer"
                    className="mt-1 block text-base font-semibold text-emerald-300 hover:text-emerald-200"
                  >
                    {source.title}
                  </a>
                ) : (
                  <p className="mt-1 text-base font-semibold text-white">
                    {source.title}
                  </p>
                )}
              </div>
            </div>

            {source.snippet && (
              <p className="mt-3 text-sm leading-6 text-slate-300">
                {source.snippet}
              </p>
            )}
          </article>
        ))}
      </div>
    </div>
  );
}