"use client";

import { useState } from "react";

type RecommendationExplanation = {
  recommendation_id: string;
  explanation: string;
  beginner_summary: string;
  risk_explanation: string;
  disclaimer: string;
};

export function ExplanationPanel() {
  const [explanation, setExplanation] =
    useState<RecommendationExplanation | null>(null);
  const [statusMessage, setStatusMessage] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);

  async function handleGenerateExplanation() {
    setIsGenerating(true);
    setStatusMessage("");

    try {
      const response = await fetch("/api/explanations/recommendation", {
        method: "POST",
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result?.detail ?? "Explanation generation failed");
      }

      setExplanation(result?.data ?? null);
      setStatusMessage("Explanation generated successfully.");
    } catch (error) {
      setStatusMessage(
        error instanceof Error
          ? error.message
          : "Something went wrong while generating explanation.",
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
              Explanation Engine
            </p>
            <h1 className="mt-2 text-3xl font-bold">
              Recommendation Explanation
            </h1>
            <p className="mt-3 max-w-3xl text-slate-400">
              Generate a beginner-friendly explanation for the latest
              recommendation. This is currently template-based and will later be
              upgraded with an AI explanation provider.
            </p>
          </div>

          <button
            type="button"
            onClick={handleGenerateExplanation}
            disabled={isGenerating}
            className="rounded-lg bg-emerald-500 px-5 py-3 font-semibold text-slate-950 hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {isGenerating ? "Generating..." : "Generate Explanation"}
          </button>
        </div>

        {statusMessage ? (
          <p className="mt-5 rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-sm text-slate-300">
            {statusMessage}
          </p>
        ) : null}
      </section>

      {explanation ? (
        <section className="space-y-5 rounded-2xl border border-emerald-500/30 bg-slate-900 p-6">
          <div>
            <p className="text-sm uppercase tracking-wide text-emerald-300">
              Recommendation ID
            </p>
            <p className="mt-2 text-slate-300">
              {explanation.recommendation_id}
            </p>
          </div>

          <div className="rounded-xl border border-slate-700 bg-slate-800 p-5">
            <p className="text-sm uppercase tracking-wide text-slate-400">
              Beginner Summary
            </p>
            <p className="mt-3 leading-7 text-slate-200">
              {explanation.beginner_summary}
            </p>
          </div>

          <div className="rounded-xl border border-slate-700 bg-slate-800 p-5">
            <p className="text-sm uppercase tracking-wide text-slate-400">
              Explanation
            </p>
            <p className="mt-3 leading-7 text-slate-200">
              {explanation.explanation}
            </p>
          </div>

          <div className="rounded-xl border border-slate-700 bg-slate-800 p-5">
            <p className="text-sm uppercase tracking-wide text-slate-400">
              Risk Explanation
            </p>
            <p className="mt-3 leading-7 text-slate-200">
              {explanation.risk_explanation}
            </p>
          </div>

          <div className="rounded-xl border border-amber-500/40 bg-amber-950/30 p-5 text-amber-100">
            <p className="font-semibold">Disclaimer</p>
            <p className="mt-2 text-sm leading-6">{explanation.disclaimer}</p>
          </div>
        </section>
      ) : (
        <section className="rounded-2xl border border-slate-700 bg-slate-900 p-6">
          <p className="text-slate-400">
            No explanation generated yet. Generate a recommendation first, then
            create an explanation.
          </p>
        </section>
      )}
    </div>
  );
}