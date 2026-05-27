"use client";

import { useEffect, useState } from "react";

type PortfolioSummary = {
  total_invested: number;
  current_value: number;
  gain_loss: number;
  gain_loss_percent: number;
  number_of_holdings: number;
};

type Recommendation = {
  recommendation_id: string;
  suggested_action: string;
  suggested_amount: number | null;
  recommendation_date?: string;
};

type Explanation = {
  explanation_id: string | null;
  recommendation_id: string;
  provider: string | null;
  created_at: string | null;
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

function formatPercent(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return "—";
  }

  return `${value.toFixed(2)}%`;
}

function formatLabel(value: string | null | undefined) {
  if (!value) {
    return "Not available";
  }

  return value
    .replaceAll("_", " ")
    .toLowerCase()
    .replace(/\w/g, (character) => character.toUpperCase());
}

async function fetchOptionalData<T>(url: string): Promise<T | null> {
  const response = await fetch(url, {
    cache: "no-store",
  });

  const result = await response.json();

  if (response.status === 404) {
    return null;
  }

  if (!response.ok) {
    throw new Error(result?.detail ?? `Failed to fetch ${url}`);
  }

  return result?.data ?? null;
}

export function DashboardQuickStats() {
  const [summary, setSummary] = useState<PortfolioSummary | null>(null);
  const [recommendation, setRecommendation] = useState<Recommendation | null>(
    null,
  );
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [statusMessage, setStatusMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  async function loadQuickStats() {
    setIsLoading(true);
    setStatusMessage("");

    try {
      const [portfolioSummary, latestRecommendation, latestExplanation] =
        await Promise.all([
          fetchOptionalData<PortfolioSummary>("/api/portfolio/summary"),
          fetchOptionalData<Recommendation>("/api/recommendations/latest"),
          fetchOptionalData<Explanation>("/api/explanations/latest"),
        ]);

      setSummary(portfolioSummary);
      setRecommendation(latestRecommendation);
      setExplanation(latestExplanation);
      setStatusMessage("Dashboard quick stats loaded.");
    } catch (error) {
      setStatusMessage(
        error instanceof Error
          ? error.message
          : "Something went wrong while loading dashboard quick stats.",
      );
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    loadQuickStats();
  }, []);

  return (
    <section className="rounded-2xl border border-slate-700 bg-slate-900 p-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-sm uppercase tracking-wide text-emerald-300">
            Dashboard Snapshot
          </p>
          <h2 className="mt-2 text-2xl font-semibold">Quick Stats</h2>
          <p className="mt-2 text-sm text-slate-400">
            A quick overview of portfolio value, latest recommendation, and
            latest AI explanation.
          </p>
        </div>

        <button
          type="button"
          onClick={loadQuickStats}
          disabled={isLoading}
          className="rounded-lg border border-slate-600 px-4 py-3 font-semibold text-slate-100 hover:border-emerald-400 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {isLoading ? "Refreshing..." : "Refresh Stats"}
        </button>
      </div>

      {statusMessage ? (
        <p className="mt-5 rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-sm text-slate-300">
          {statusMessage}
        </p>
      ) : null}

      <div className="mt-6 grid gap-4 md:grid-cols-3">
        <div className="rounded-xl border border-slate-700 bg-slate-800 p-5">
          <p className="text-sm text-slate-400">Total Invested</p>
          <p className="mt-2 text-2xl font-semibold">
            {formatCurrency(summary?.total_invested)}
          </p>
        </div>

        <div className="rounded-xl border border-slate-700 bg-slate-800 p-5">
          <p className="text-sm text-slate-400">Current Value</p>
          <p className="mt-2 text-2xl font-semibold">
            {formatCurrency(summary?.current_value)}
          </p>
        </div>

        <div className="rounded-xl border border-slate-700 bg-slate-800 p-5">
          <p className="text-sm text-slate-400">Gain / Loss</p>
          <p
            className={`mt-2 text-2xl font-semibold ${
              summary && summary.gain_loss >= 0
                ? "text-emerald-300"
                : "text-red-300"
            }`}
          >
            {summary
              ? `${formatCurrency(summary.gain_loss)} (${formatPercent(
                  summary.gain_loss_percent,
                )})`
              : "—"}
          </p>
        </div>

        <div className="rounded-xl border border-slate-700 bg-slate-800 p-5">
          <p className="text-sm text-slate-400">Holdings</p>
          <p className="mt-2 text-2xl font-semibold">
            {summary?.number_of_holdings ?? "—"}
          </p>
        </div>

        <div className="rounded-xl border border-slate-700 bg-slate-800 p-5">
          <p className="text-sm text-slate-400">Latest Recommendation</p>
          <p className="mt-2 text-lg font-semibold text-emerald-200">
            {formatLabel(recommendation?.suggested_action)}
          </p>
          <p className="mt-2 text-sm text-slate-400">
            Amount: {formatCurrency(recommendation?.suggested_amount)}
          </p>
        </div>

        <div className="rounded-xl border border-slate-700 bg-slate-800 p-5">
          <p className="text-sm text-slate-400">Latest Explanation Provider</p>
          <p className="mt-2 text-lg font-semibold text-purple-200">
            {explanation?.provider ?? "Not generated"}
          </p>
          <p className="mt-2 text-sm text-slate-400">
            Recommendation ID: {explanation?.recommendation_id ?? "—"}
          </p>
        </div>
      </div>
    </section>
  );
}
