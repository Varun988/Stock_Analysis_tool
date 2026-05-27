"use client";

import { useEffect, useState } from "react";

type AllocationPlanItem = {
  instrument_type: string;
  amount: number;
  reason: string;
};

type ScoreBreakdown = {
  diversification_score: number;
  risk_suitability_score: number;
  preference_match_score: number;
};

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

type Recommendation = {
  recommendation_id: string;
  suggested_action: string;
  suggested_amount: number | null;
  summary: string;
  reason_codes: string[];
  risk_note: string;
  disclaimer: string;
  allocation_plan?: AllocationPlanItem[];
  score_breakdown?: ScoreBreakdown | null;
  research_context?: ResearchContext | null;
};

function formatCurrency(value: number) {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 2,
  }).format(value);
}

function formatLabel(value: string) {
  return value
    .replaceAll("_", " ")
    .toLowerCase()
    .replace(/\w/g, (character) => character.toUpperCase());
}

function getScoreTone(score: number) {
  if (score >= 75) {
    return "text-emerald-300";
  }

  if (score >= 50) {
    return "text-amber-300";
  }

  return "text-red-300";
}

export function RecommendationPanel() {
  const [recommendation, setRecommendation] = useState<Recommendation | null>(
    null,
  );
  const [statusMessage, setStatusMessage] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  async function loadLatestRecommendation() {
    setIsLoading(true);

    try {
      const response = await fetch("/api/recommendations/latest", {
        cache: "no-store",
      });

      const result = await response.json();

      if (!response.ok) {
        if (response.status === 404) {
          setRecommendation(null);
          setStatusMessage("No recommendation generated yet.");
          return;
        }

        throw new Error(result?.detail ?? "Failed to load recommendation");
      }

      setRecommendation(result?.data ?? null);
      setStatusMessage("Latest recommendation loaded.");
    } catch (error) {
      setStatusMessage(
        error instanceof Error
          ? error.message
          : "Something went wrong while loading recommendation.",
      );
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    loadLatestRecommendation();
  }, []);

  async function handleGenerateRecommendation() {
    setIsGenerating(true);
    setStatusMessage("");

    try {
      const response = await fetch("/api/recommendations/generate", {
        method: "POST",
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result?.detail ?? "Recommendation generation failed");
      }

      setRecommendation(result?.data ?? null);
      setStatusMessage("Recommendation generated successfully.");
    } catch (error) {
      setStatusMessage(
        error instanceof Error
          ? error.message
          : "Something went wrong while generating recommendation.",
      );
    } finally {
      setIsGenerating(false);
    }
  }

  return (
    <div className="space-y-8">
      <section className="rounded-2xl border border-slate-700 bg-slate-900 p-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-sm uppercase tracking-wide text-emerald-300">
              Recommendation Engine
            </p>
            <h1 className="mt-2 text-3xl font-bold">
              Investment Recommendation
            </h1>
            <p className="mt-3 max-w-3xl text-slate-400">
              Generate a recommendation using your investor profile, portfolio
              holdings, provider-aware market data, metrics, and risk engine.
            </p>
          </div>

          <div className="flex flex-wrap gap-3">
            <button
              type="button"
              onClick={loadLatestRecommendation}
              disabled={isLoading}
              className="rounded-lg border border-slate-600 px-4 py-3 font-semibold text-slate-100 hover:border-emerald-400 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {isLoading ? "Loading..." : "Refresh Latest"}
            </button>

            <button
              type="button"
              onClick={handleGenerateRecommendation}
              disabled={isGenerating}
              className="rounded-lg bg-emerald-500 px-5 py-3 font-semibold text-slate-950 hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {isGenerating ? "Generating..." : "Generate Recommendation"}
            </button>
          </div>
        </div>

        {statusMessage ? (
          <p className="mt-5 rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-sm text-slate-300">
            {statusMessage}
          </p>
        ) : null}
      </section>

      {recommendation ? (
        <section className="rounded-2xl border border-emerald-500/30 bg-slate-900 p-6">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <p className="text-sm uppercase tracking-wide text-emerald-300">
                Suggested Action
              </p>
              <h2 className="mt-2 text-3xl font-bold">
                {formatLabel(recommendation.suggested_action)}
              </h2>
              <p className="mt-2 text-sm text-slate-400">
                Recommendation ID: {recommendation.recommendation_id}
              </p>
            </div>

            <div className="rounded-xl border border-emerald-500/30 bg-emerald-950/30 p-5 text-emerald-100">
              <p className="text-sm opacity-80">Suggested Amount</p>
              <p className="mt-2 text-2xl font-semibold">
                {recommendation.suggested_amount !== null
                ? formatCurrency(recommendation.suggested_amount)
                : "Not available"}

              </p>
            </div>
          </div>

          <div className="mt-8 grid gap-5 md:grid-cols-2">
            <div className="rounded-xl border border-slate-700 bg-slate-800 p-5">
              <p className="text-sm uppercase tracking-wide text-slate-400">
                Summary
              </p>
              <p className="mt-3 leading-7 text-slate-200">
                {recommendation.summary}
              </p>
            </div>

            <div className="rounded-xl border border-slate-700 bg-slate-800 p-5">
              <p className="text-sm uppercase tracking-wide text-slate-400">
                Risk Note
              </p>
              <p className="mt-3 leading-7 text-slate-200">
                {recommendation.risk_note}
              </p>
            </div>
          </div>

          <div className="mt-6 rounded-xl border border-slate-700 bg-slate-800 p-5">
            <p className="text-sm uppercase tracking-wide text-slate-400">
              Reason Codes
            </p>

            <div className="mt-4 flex flex-wrap gap-2">
              {recommendation.allocation_plan &&
              recommendation.allocation_plan.length > 0 ? (
                <div className="mt-6 rounded-xl border border-emerald-500/30 bg-emerald-950/20 p-5">
                  <p className="text-sm uppercase tracking-wide text-emerald-300">
                    Suggested Monthly Allocation
                  </p>

                  <div className="mt-4 grid gap-3 md:grid-cols-2">
                    {recommendation.allocation_plan.map((item) => (
                      <div
                        key={item.instrument_type}
                        className="rounded-lg border border-emerald-500/20 bg-slate-900 p-4"
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div>
                            <p className="font-semibold">
                              {formatLabel(item.instrument_type)}
                            </p>
                            <p className="mt-2 text-sm leading-6 text-slate-300">
                              {item.reason}
                            </p>
                          </div>

                          <p className="text-lg font-bold text-emerald-300">
                            {formatCurrency(item.amount)}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}

              {recommendation.score_breakdown ? (
                <div className="mt-6 rounded-xl border border-slate-700 bg-slate-800 p-5">
                  <p className="text-sm uppercase tracking-wide text-slate-400">
                    Score Breakdown
                  </p>

                  <div className="mt-4 grid gap-4 md:grid-cols-3">
                    <div className="rounded-lg border border-slate-700 bg-slate-900 p-4">
                      <p className="text-sm text-slate-400">Diversification</p>
                      <p
                        className={`mt-2 text-2xl font-semibold ${getScoreTone(
                          recommendation.score_breakdown.diversification_score,
                        )}`}
                      >
                        {recommendation.score_breakdown.diversification_score}/100
                      </p>
                    </div>

                    <div className="rounded-lg border border-slate-700 bg-slate-900 p-4">
                      <p className="text-sm text-slate-400">Risk Suitability</p>
                      <p
                        className={`mt-2 text-2xl font-semibold ${getScoreTone(
                          recommendation.score_breakdown.risk_suitability_score,
                        )}`}
                      >
                        {recommendation.score_breakdown.risk_suitability_score}/100
                      </p>
                    </div>

                    <div className="rounded-lg border border-slate-700 bg-slate-900 p-4">
                      <p className="text-sm text-slate-400">Preference Match</p>
                      <p
                        className={`mt-2 text-2xl font-semibold ${getScoreTone(
                          recommendation.score_breakdown.preference_match_score,
                        )}`}
                      >
                        {recommendation.score_breakdown.preference_match_score}/100
                      </p>
                    </div>
                  </div>
                </div>
              ) : null}

              {recommendation.reason_codes.map((reasonCode) => (
                <span
                  key={reasonCode}
                  className="rounded-full border border-sky-500/40 bg-sky-950 px-3 py-1 text-xs text-sky-200"
                >
                  {formatLabel(reasonCode)}
                </span>
              ))}
            </div>
          </div>
            {recommendation.research_context ? (
              <ResearchContextPanel research={recommendation.research_context} />
            ) : null}
          <div className="mt-6 rounded-xl border border-amber-500/40 bg-amber-950/30 p-5 text-amber-100">
            <p className="font-semibold">Disclaimer</p>
            <p className="mt-2 text-sm leading-6">
              {recommendation.disclaimer}
            </p>
          </div>
        </section>
      ) : (
        <section className="rounded-2xl border border-slate-700 bg-slate-900 p-6">
          <p className="text-slate-400">
            No recommendation available yet. Create a profile, add instruments,
            add portfolio holdings, then generate a recommendation.
          </p>
        </section>
      )}
    </div>
  );
}
function ResearchContextPanel({ research }: { research: ResearchContext }) {
  return (
    <div className="mt-6 rounded-xl border border-purple-500/30 bg-purple-950/20 p-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-sm uppercase tracking-wide text-purple-300">
            Research Context Used
          </p>
          <h3 className="mt-2 text-xl font-semibold text-white">
            {formatLabel(research.subject_type)} Research
          </h3>
          <p className="mt-2 text-sm text-slate-400">
            Query:{" "}
            <span className="font-medium text-purple-200">
              {research.query}
            </span>
          </p>
        </div>

        <div className="flex flex-wrap gap-2">
          <ResearchBadge label={`Provider: ${research.provider}`} />
          <ResearchBadge label={`Summarizer: ${research.summarizer}`} />
        </div>
      </div>

      <div className="mt-5 rounded-lg border border-slate-700 bg-slate-900 p-4">
        <p className="text-sm uppercase tracking-wide text-slate-400">
          Research Summary
        </p>
        <p className="mt-3 leading-7 text-slate-200">{research.summary}</p>
      </div>

      {research.key_points.length > 0 ? (
        <div className="mt-5 rounded-lg border border-slate-700 bg-slate-900 p-4">
          <p className="text-sm uppercase tracking-wide text-slate-400">
            Key Points
          </p>
          <ul className="mt-3 list-disc space-y-2 pl-5 text-slate-200">
            {research.key_points.map((point) => (
              <li key={point}>{point}</li>
            ))}
          </ul>
        </div>
      ) : null}

      <div className="mt-5 rounded-lg border border-amber-500/30 bg-amber-950/20 p-4">
        <p className="text-sm uppercase tracking-wide text-amber-300">
          Research Risk Note
        </p>
        <p className="mt-3 leading-7 text-amber-100">{research.risk_note}</p>
      </div>

      <ResearchSources sources={research.sources} />
    </div>
  );
}

function ResearchBadge({ label }: { label: string }) {
  return (
    <span className="rounded-full border border-purple-500/40 bg-purple-950 px-3 py-1 text-xs text-purple-200">
      {label}
    </span>
  );
}

function ResearchSources({ sources }: { sources: ResearchSource[] }) {
  if (sources.length === 0) {
    return (
      <div className="mt-5 rounded-lg border border-slate-700 bg-slate-900 p-4 text-sm text-slate-400">
        No research sources were returned.
      </div>
    );
  }

  return (
    <div className="mt-5 overflow-hidden rounded-lg border border-slate-700 bg-slate-900">
      <div className="border-b border-slate-700 px-4 py-3">
        <p className="text-sm uppercase tracking-wide text-slate-400">
          Research Sources
        </p>
        <p className="mt-1 text-xs text-slate-500">
          Review source links directly before relying on article content.
        </p>
      </div>

      <div className="divide-y divide-slate-800">
        {sources.map((source, index) => (
          <article key={`${source.title}-${index}`} className="p-4">
            <p className="text-xs text-slate-500">
              #{source.position ?? index + 1}
              {source.source_name ? ` • ${source.source_name}` : ""}
              {source.published_at ? ` • ${source.published_at}` : ""}
            </p>

            {source.url ? (
              <a
                href={source.url}
                target="_blank"
                rel="noreferrer"
                className="mt-1 block font-semibold text-sky-300 hover:text-sky-200"
              >
                {source.title}
              </a>
            ) : (
              <p className="mt-1 font-semibold text-slate-100">
                {source.title}
              </p>
            )}

            {source.snippet ? (
              <p className="mt-2 text-sm leading-6 text-slate-300">
                {source.snippet}
              </p>
            ) : null}
          </article>
        ))}
      </div>
    </div>
  );
}