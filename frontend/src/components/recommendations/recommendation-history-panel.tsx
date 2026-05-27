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

type RecommendationHistoryItem = {
  recommendation_id: string;
  recommendation_date: string;
  suggested_action: string;
  suggested_amount: number | null;
  summary: string;
  reason_codes: string[];
  risk_note: string;
  disclaimer: string;
  allocation_plan?: AllocationPlanItem[];
  score_breakdown?: ScoreBreakdown | null;
};

function formatCurrency(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return "—";
  }

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

function formatDate(value: string) {
  return new Intl.DateTimeFormat("en-IN", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
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

export function RecommendationHistoryPanel() {
  const [recommendations, setRecommendations] = useState<
    RecommendationHistoryItem[]
  >([]);
  const [statusMessage, setStatusMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  async function loadRecommendationHistory() {
    setIsLoading(true);
    setStatusMessage("");

    try {
      const response = await fetch("/api/recommendations/history", {
        cache: "no-store",
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(
          result?.detail ?? "Failed to load recommendation history",
        );
      }

      setRecommendations(result?.data ?? []);
      setStatusMessage("Recommendation history loaded.");
    } catch (error) {
      setStatusMessage(
        error instanceof Error
          ? error.message
          : "Something went wrong while loading recommendation history.",
      );
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    loadRecommendationHistory();
  }, []);

  return (
    <div className="space-y-8">
      <section className="rounded-2xl border border-slate-700 bg-slate-900 p-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-sm uppercase tracking-wide text-emerald-300">
              Recommendation History
            </p>
            <h1 className="mt-2 text-3xl font-bold">
              Previous Recommendations
            </h1>
            <p className="mt-3 max-w-3xl text-slate-400">
              View previously generated recommendations saved in PostgreSQL.
            </p>
          </div>

          <button
            type="button"
            onClick={loadRecommendationHistory}
            disabled={isLoading}
            className="rounded-lg border border-slate-600 px-4 py-3 font-semibold text-slate-100 hover:border-emerald-400 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {isLoading ? "Loading..." : "Refresh History"}
          </button>
        </div>

        {statusMessage ? (
          <p className="mt-5 rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-sm text-slate-300">
            {statusMessage}
          </p>
        ) : null}
      </section>

      {recommendations.length === 0 ? (
        <section className="rounded-2xl border border-slate-700 bg-slate-900 p-6">
          <p className="text-slate-400">
            No recommendation history found yet. Generate a recommendation first.
          </p>
        </section>
      ) : (
        <section className="space-y-5">
          {recommendations.map((recommendation) => (
            <article
              key={recommendation.recommendation_id}
              className="rounded-2xl border border-slate-700 bg-slate-900 p-6"
            >
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div>
                  <p className="text-sm uppercase tracking-wide text-emerald-300">
                    {formatLabel(recommendation.suggested_action)}
                  </p>

                  <h2 className="mt-2 text-2xl font-semibold">
                    {formatCurrency(recommendation.suggested_amount)}
                  </h2>

                  <p className="mt-2 text-sm text-slate-400">
                    {formatDate(recommendation.recommendation_date)}
                  </p>

                  <p className="mt-1 text-xs text-slate-500">
                    ID: {recommendation.recommendation_id}
                  </p>
                </div>
              </div>

              <div className="mt-6 rounded-xl border border-slate-700 bg-slate-800 p-5">
                <p className="text-sm uppercase tracking-wide text-slate-400">
                  Summary
                </p>
                <p className="mt-3 leading-7 text-slate-200">
                  {recommendation.summary}
                </p>
              </div>

              {recommendation.allocation_plan &&
              recommendation.allocation_plan.length > 0 ? (
                <div className="mt-6 rounded-xl border border-emerald-500/30 bg-emerald-950/20 p-5">
                  <p className="text-sm uppercase tracking-wide text-emerald-300">
                    Allocation Plan
                  </p>

                  <div className="mt-4 grid gap-3 md:grid-cols-2">
                    {recommendation.allocation_plan.map((item) => (
                      <div
                        key={`${recommendation.recommendation_id}-${item.instrument_type}`}
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
                      <p className="text-sm text-slate-400">
                        Diversification
                      </p>
                      <p
                        className={`mt-2 text-2xl font-semibold ${getScoreTone(
                          recommendation.score_breakdown.diversification_score,
                        )}`}
                      >
                        {recommendation.score_breakdown.diversification_score}
                        /100
                      </p>
                    </div>

                    <div className="rounded-lg border border-slate-700 bg-slate-900 p-4">
                      <p className="text-sm text-slate-400">
                        Risk Suitability
                      </p>
                      <p
                        className={`mt-2 text-2xl font-semibold ${getScoreTone(
                          recommendation.score_breakdown
                            .risk_suitability_score,
                        )}`}
                      >
                        {recommendation.score_breakdown.risk_suitability_score}
                        /100
                      </p>
                    </div>

                    <div className="rounded-lg border border-slate-700 bg-slate-900 p-4">
                      <p className="text-sm text-slate-400">
                        Preference Match
                      </p>
                      <p
                        className={`mt-2 text-2xl font-semibold ${getScoreTone(
                          recommendation.score_breakdown.preference_match_score,
                        )}`}
                      >
                        {recommendation.score_breakdown.preference_match_score}
                        /100
                      </p>
                    </div>
                  </div>
                </div>
              ) : null}

              <div className="mt-6 rounded-xl border border-slate-700 bg-slate-800 p-5">
                <p className="text-sm uppercase tracking-wide text-slate-400">
                  Reason Codes
                </p>

                <div className="mt-4 flex flex-wrap gap-2">
                  {recommendation.reason_codes.map((reasonCode) => (
                    <span
                      key={`${recommendation.recommendation_id}-${reasonCode}`}
                      className="rounded-full border border-sky-500/40 bg-sky-950 px-3 py-1 text-xs text-sky-200"
                    >
                      {formatLabel(reasonCode)}
                    </span>
                  ))}
                </div>
              </div>

              <div className="mt-6 rounded-xl border border-slate-700 bg-slate-800 p-5">
                <p className="text-sm uppercase tracking-wide text-slate-400">
                  Risk Note
                </p>
                <p className="mt-3 leading-7 text-slate-200">
                  {recommendation.risk_note}
                </p>
              </div>
            </article>
          ))}
        </section>
      )}
    </div>
  );
}
