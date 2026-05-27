"use client";

import { useEffect, useState } from "react";

type ExplanationHistoryItem = {
  explanation_id: string | null;
  recommendation_id: string;
  provider: string | null;
  explanation: string;
  beginner_summary: string;
  risk_explanation: string;
  disclaimer: string;
  created_at: string | null;
};

function formatDate(value: string | null) {
  if (!value) {
    return "—";
  }

  return new Intl.DateTimeFormat("en-IN", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

function formatProvider(value: string | null) {
  if (!value) {
    return "UNKNOWN";
  }

  return value.replaceAll("_", " ");
}

function getProviderTone(provider: string | null) {
  if (provider === "GEMINI") {
    return "border-purple-500/40 bg-purple-950 text-purple-200";
  }

  if (provider === "MOCK") {
    return "border-slate-500/40 bg-slate-800 text-slate-200";
  }

  return "border-amber-500/40 bg-amber-950 text-amber-200";
}

export function ExplanationHistoryPanel() {
  const [explanations, setExplanations] = useState<ExplanationHistoryItem[]>(
    [],
  );
  const [statusMessage, setStatusMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  async function loadExplanationHistory() {
    setIsLoading(true);
    setStatusMessage("");

    try {
      const response = await fetch("/api/explanations/history", {
        cache: "no-store",
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result?.detail ?? "Failed to load explanation history");
      }

      setExplanations(result?.data ?? []);
      setStatusMessage("Explanation history loaded.");
    } catch (error) {
      setStatusMessage(
        error instanceof Error
          ? error.message
          : "Something went wrong while loading explanation history.",
      );
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    loadExplanationHistory();
  }, []);

  return (
    <div className="space-y-8">
      <section className="rounded-2xl border border-slate-700 bg-slate-900 p-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-sm uppercase tracking-wide text-purple-300">
              Explanation History
            </p>
            <h1 className="mt-2 text-3xl font-bold">Previous Explanations</h1>
            <p className="mt-3 max-w-3xl text-slate-400">
              View previously generated recommendation explanations saved in
              PostgreSQL.
            </p>
          </div>

          <button
            type="button"
            onClick={loadExplanationHistory}
            disabled={isLoading}
            className="rounded-lg border border-slate-600 px-4 py-3 font-semibold text-slate-100 hover:border-purple-400 disabled:cursor-not-allowed disabled:opacity-60"
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

      {explanations.length === 0 ? (
        <section className="rounded-2xl border border-slate-700 bg-slate-900 p-6">
          <p className="text-slate-400">
            No explanation history found yet. Generate an explanation first.
          </p>
        </section>
      ) : (
        <section className="space-y-5">
          {explanations.map((item) => (
            <article
              key={item.explanation_id ?? item.recommendation_id}
              className="rounded-2xl border border-slate-700 bg-slate-900 p-6"
            >
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div>
                  <p className="text-sm uppercase tracking-wide text-purple-300">
                    Explanation
                  </p>

                  <h2 className="mt-2 text-2xl font-semibold">
                    {formatDate(item.created_at)}
                  </h2>

                  <p className="mt-2 text-xs text-slate-500">
                    Explanation ID: {item.explanation_id ?? "—"}
                  </p>

                  <p className="mt-1 text-xs text-slate-500">
                    Recommendation ID: {item.recommendation_id}
                  </p>
                </div>

                <span
                  className={`rounded-full border px-3 py-1 text-xs ${getProviderTone(
                    item.provider,
                  )}`}
                >
                  {formatProvider(item.provider)}
                </span>
              </div>

              <div className="mt-6 rounded-xl border border-slate-700 bg-slate-800 p-5">
                <p className="text-sm uppercase tracking-wide text-slate-400">
                  Beginner Summary
                </p>
                <p className="mt-3 leading-7 text-slate-200">
                  {item.beginner_summary}
                </p>
              </div>

              <div className="mt-6 rounded-xl border border-slate-700 bg-slate-800 p-5">
                <p className="text-sm uppercase tracking-wide text-slate-400">
                  Explanation
                </p>
                <p className="mt-3 leading-7 text-slate-200">
                  {item.explanation}
                </p>
              </div>

              <div className="mt-6 rounded-xl border border-slate-700 bg-slate-800 p-5">
                <p className="text-sm uppercase tracking-wide text-slate-400">
                  Risk Explanation
                </p>
                <p className="mt-3 leading-7 text-slate-200">
                  {item.risk_explanation}
                </p>
              </div>

              <div className="mt-6 rounded-xl border border-amber-500/40 bg-amber-950/30 p-5 text-amber-100">
                <p className="font-semibold">Disclaimer</p>
                <p className="mt-2 text-sm leading-6">{item.disclaimer}</p>
              </div>
            </article>
          ))}
        </section>
      )}
    </div>
  );
}
