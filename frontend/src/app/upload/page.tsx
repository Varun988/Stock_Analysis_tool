
"use client";

import { ChangeEvent, ReactNode, useState } from "react";

type ExtractedHolding = {
  instrument_id?: string | null;
  instrument_name: string;
  instrument_type: string;
  symbol?: string | null;
  isin?: string | null;
  quantity: number;
  average_cost: number;
  invested_amount: number;
  current_price?: number | null;
  current_value: number;
  gain_loss?: number | null;
  gain_loss_percent?: number | null;
  confidence?: string | null;
  extraction_source?: string | null;

  resolved?: boolean;
  resolved_name?: string | null;
  resolved_instrument_type?: string | null;
  resolved_symbol?: string | null;
  resolved_exchange?: string | null;
  yfinance_symbol?: string | null;
  amfi_scheme_code?: string | null;
  market_data_provider?: string | null;
  benchmark?: string | null;
  exposure_category?: string | null;
  match_confidence?: string | null;
  provider_lookup_required?: boolean;
  match_method?: string | null;
  resolver_warnings?: string[];
};

type InvalidExtractedHolding = {
  row_number: number;
  holding: Partial<ExtractedHolding>;
  errors: string[];
};

type SummaryValidation = {
  summary_found: boolean;
  summary_invested_value?: number | null;
  calculated_invested_value?: number | null;
  invested_value_matches?: boolean;
  summary_current_value?: number | null;
  calculated_current_value?: number | null;
  current_value_matches?: boolean;
  summary_gain_loss?: number | null;
  calculated_gain_loss?: number | null;
  gain_loss_matches?: boolean;
};

type PortfolioExposureAnalysis = {
  total_current_value?: number;
  benchmark_exposure?: Record<string, number>;
  category_exposure?: Record<string, number>;
  instrument_type_exposure?: Record<string, number>;
  market_data_provider_exposure?: Record<string, number>;
  primary_benchmark?: { name: string; percent: number } | null;
  primary_exposure_category?: { name: string; percent: number } | null;
  overlap_warnings?: string[];
  diversification_gaps?: string[];
  candidate_category_hints?: Array<{ category: string; reason: string }>;
  data_quality?: {
    total_holdings?: number;
    resolved_holdings_count?: number;
    unresolved_holdings_count?: number;
    high_confidence_matches_count?: number;
    resolution_coverage_percent?: number;
    unresolved_holdings?: Array<{
      instrument_name?: string | null;
      isin?: string | null;
      reason?: string[];
    }>;
  };
};

type HistoricalHoldingResult = {
  instrument_name?: string | null;
  resolved_name?: string | null;
  isin?: string | null;
  benchmark?: string | null;
  exposure_category?: string | null;
  market_data_provider?: string | null;
  yfinance_symbol?: string | null;
  symbol?: string | null;
  provider?: string | null;
  successful_period?: string | null;
  historical_analysis_available?: boolean;
  data_quality?: string | null;
  data_points?: number;
  first_date?: string | null;
  latest_date?: string | null;
  latest_value?: number | null;
  trailing_returns?: Record<string, number | null>;
  cagr?: Record<string, number | null>;
  volatility_annualized_percent?: number | null;
  max_drawdown_percent?: number | null;
  positive_month_ratio_percent?: number | null;
  scores?: Record<string, number | string | null>;
  message?: string | null;
  provider_errors?: string[];
};

type HistoricalPerformanceAnalysis = {
  analysis_date?: string;
  provider_scope?: string;
  holdings_analyzed_count?: number;
  holdings_skipped_count?: number;
  average_overall_historical_score?: number | null;
  holding_results?: HistoricalHoldingResult[];
  warnings?: string[];
};

type BenchmarkComparisonResult = {
  instrument_name?: string | null;
  yfinance_symbol?: string | null;
  benchmark?: string | null;
  benchmark_comparison_available?: boolean;
  benchmark_name?: string | null;
  benchmark_symbol?: string | null;
  benchmark_provider?: string | null;
  proxy_used?: boolean;
  proxy_note?: string | null;
  period_comparison?: Record<
    string,
    {
      instrument?: number | null;
      benchmark?: number | null;
      difference?: number | null;
      outperformed?: boolean | null;
    }
  >;
  risk_comparison?: Record<string, number | boolean | null>;
  scores?: Record<string, number | string | null>;
  message?: string | null;
};

type BenchmarkComparisonAnalysis = {
  benchmark_comparisons_available_count?: number;
  benchmark_comparisons_skipped_count?: number;
  average_benchmark_score?: number | null;
  comparison_results?: BenchmarkComparisonResult[];
  warnings?: string[];
};

type CandidateQualityCheck = {
  status?: string | null;
  value?: string | number | null;
  note?: string | null;
};

type CandidateReviewInstrument = {
  candidate_category?: string | null;
  isin?: string | null;
  amfi_scheme_code?: string | null;
  instrument_name?: string | null;
  instrument_type?: string | null;
  nse_symbol?: string | null;
  yfinance_symbol?: string | null;
  benchmark?: string | null;
  exposure_category?: string | null;
  verification_status?: string | null;
  final_recommendation_status?: string | null;

  candidate_analysis_score?: number | null;
  candidate_analysis_status?: string | null;
  candidate_analysis_note?: string | null;
  profile_suitability_score?: number | null;
  profile_suitability_reasons?: string[];
  profile_suitability_warnings?: string[];
  final_candidate_review_score?: number | null;
  review_score_note?: string | null;

  candidate_historical_analysis?: HistoricalHoldingResult | null;
  candidate_benchmark_comparison?: BenchmarkComparisonResult | null;

  mf_review_required?: boolean;
  mf_review_status?: string | null;
  mf_pending_checks?: string[];
  mf_quality_checks?: Record<string, CandidateQualityCheck>;
  mf_review_warnings?: string[];

  candidate_rank?: number;
  ranking_basis?: string | null;
};

type Candidate = {
  candidate_id?: string;
  instrument_name?: string;
  instrument_type?: string;
  candidate_category?: string;
  benchmark?: string | null;
  exposure_category?: string | null;
  reason_considered?: string;
  risk_bucket?: string;
  portfolio_gap_score?: number;
  candidate_final_score?: number;
  provider_resolution_score?: number;
  candidate_flags?: string[];
  portfolio_gap_reasons?: string[];
  profile_suitability_score?: number;
  profile_suitability_reasons?: string[];
  resolved_candidate_instruments?: CandidateReviewInstrument[];
  candidate_resolution_method?: string;
  candidate_resolution_warnings?: string[];

  category_review_status?: string | null;
  category_review_note?: string | null;
  reviewable_instruments_count?: number;
  benchmark_pending_instruments_count?: number;
  analyzed_instruments_count?: number;
  placeholder_instruments_count?: number;
  top_final_candidate_review_score?: number | null;

  verified_candidate_instruments_count?: number;
  placeholder_candidate_instruments_count?: number;
  ranked_candidate_instruments?: CandidateReviewInstrument[];
  ranked_candidate_instruments_count?: number;
  top_candidate_instrument?: CandidateReviewInstrument | null;
};

type ExternalCandidateDiscovery = {
  candidate_discovery_scope?: string;
  shortlisted_candidates_count?: number;
  watchlist_candidates_count?: number;
  shortlisted_candidates?: Candidate[];
  watchlist_candidates?: Candidate[];
  next_steps?: string[];
  warnings?: string[];
};

type BackendRecommendation = {
  recommendation_scope?: string;
  recommendation_date?: string;
  suggested_action?: string;
  suggested_amount?: number;
  profile_context_used?: Record<string, unknown>;
  final_recommendation_score?: number | null;
  confidence_level?: string;
  score_breakdown?: Record<string, number | null>;
  allocation_plan?: Array<{
    allocation_type?: string;
    candidate_id?: string;
    candidate_category?: string;
    amount?: number;
    candidate_final_score?: number;
    profile_suitability_score?: number;
    reason?: string;
    status?: string;
  }>;
  reason_codes?: string[];
  risk_note?: string;
  data_quality_note?: string;
  next_steps?: string[];
};

type RecommendationExplanation = {
  explanation_available?: boolean;
  summary?: string;
  why?: string[];
  key_reason_codes?: string[];
  plain_language_allocation?: unknown[];
  cautions?: string[];
  next_steps?: string[];
  ai_error?: string;
};

type ExtractionData = {
  file_name: string;
  extraction_method: string;
  holdings_detected: number;
  valid_holdings_count: number;
  invalid_holdings_count: number;
  valid_holdings: ExtractedHolding[];
  invalid_holdings: InvalidExtractedHolding[];
  summary_validation?: SummaryValidation | null;
  portfolio_exposure_analysis?: PortfolioExposureAnalysis | null;
  historical_performance_analysis?: HistoricalPerformanceAnalysis | null;
  benchmark_comparison_analysis?: BenchmarkComparisonAnalysis | null;
  external_candidate_discovery?: ExternalCandidateDiscovery | null;
  backend_recommendation?: BackendRecommendation | null;
  recommendation_explanation?: RecommendationExplanation | null;
  warnings: string[];
};

type CandidateResolveResponse = {
  success?: boolean;
  resolved_candidates?: Candidate[];
  resolved_candidates_count?: number;
  warnings?: string[];
  next_steps?: string[];
  detail?: string;
};

type ApiResponse<T> = {
  success?: boolean;
  message?: string;
  data?: T;
  detail?: string;
};

type ImportData = {
  source_upload_id: string;
  snapshot_date: string;
  deleted_existing_holdings_for_snapshot: number;
  holdings_received: number;
  holdings_imported: number;
  holdings_failed: number;
  imported_holdings: unknown[];
  failed_holdings: unknown[];
};

const ALLOWED_UPLOAD_EXTENSIONS = [".csv", ".xlsx", ".xls", ".txt", ".pdf", ".xml"];

const MAX_UPLOAD_FILE_SIZE_MB = 10;
const MAX_UPLOAD_FILE_SIZE_BYTES = MAX_UPLOAD_FILE_SIZE_MB * 1024 * 1024;

const PDF_TEXT_EXTRACTION_NOTE =
  "PDF support currently works only for text-based PDFs. Scanned/image PDFs are not supported yet because OCR is not implemented. If extraction fails, try CSV/XLSX/TXT/XML or manually enter holdings.";

function getFileExtension(fileName: string): string {
  const lastDotIndex = fileName.lastIndexOf(".");

  if (lastDotIndex === -1) {
    return "";
  }

  return fileName.slice(lastDotIndex).toLowerCase();
}

function isPdfFile(file: File): boolean {
  return getFileExtension(file.name) === ".pdf";
}

function validateUploadFile(file: File): string | null {
  const fileExtension = getFileExtension(file.name);

  if (!file.name) {
    return "Selected file does not have a valid file name.";
  }

  if (file.size === 0) {
    return "Selected file is empty. Please upload a valid portfolio statement.";
  }

  if (file.size > MAX_UPLOAD_FILE_SIZE_BYTES) {
    return `Selected file is too large. Maximum supported file size is ${MAX_UPLOAD_FILE_SIZE_MB} MB.`;
  }

  if (!ALLOWED_UPLOAD_EXTENSIONS.includes(fileExtension)) {
    return `Unsupported file type "${fileExtension || "unknown"}". Supported formats are CSV, XLSX, XLS, TXT, PDF, and XML.`;
  }

  return null;
}

function formatCurrency(value?: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }

  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 2,
  }).format(value);
}

function formatPercent(value?: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }

  return `${value.toFixed(2)}%`;
}

function formatNumber(value?: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }

  return new Intl.NumberFormat("en-IN", {
    maximumFractionDigits: 2,
  }).format(value);
}

function formatLabel(value?: string | null): string {
  if (!value) {
    return "—";
  }

  return value
    .replaceAll("_", " ")
    .toLowerCase()
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function statusBadgeClass(status?: string | null): string {
  const normalizedStatus = String(status ?? "").toUpperCase();

  if (
    [
      "HIGH",
      "GOOD",
      "TRUE",
      "RESOLVED",
      "HIGH CONFIDENCE",
      "AVAILABLE",
      "PASS",
      "PASSED",
      "FRESH",
    ].includes(normalizedStatus)
  ) {
    return "border-emerald-500/40 bg-emerald-950/50 text-emerald-200";
  }

  if (
    ["MEDIUM", "LIMITED", "PARTIAL"].includes(normalizedStatus) ||
    normalizedStatus.includes("PENDING") ||
    normalizedStatus.includes("REQUIRES") ||
    normalizedStatus.includes("PARTIAL") ||
    normalizedStatus.includes("WARNING")
  ) {
    return "border-amber-500/40 bg-amber-950/50 text-amber-200";
  }

  if (
    ["LOW", "INSUFFICIENT", "FALSE", "UNRESOLVED", "SKIPPED", "FAIL", "FAILED"].includes(
      normalizedStatus
    ) ||
    normalizedStatus.includes("FAILED")
  ) {
    return "border-rose-500/40 bg-rose-950/50 text-rose-200";
  }

  return "border-slate-600 bg-slate-900 text-slate-300";
}

function StatusBadge({ label }: { label?: string | null }) {
  return (
    <span
      className={`inline-flex rounded-full border px-2.5 py-1 text-xs font-medium ${statusBadgeClass(
        label
      )}`}
    >
      {formatLabel(label)}
    </span>
  );
}

function AnalysisCard({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: ReactNode;
}) {
  return (
    <section className="rounded-2xl border border-slate-700 bg-slate-900/80 p-6 shadow-lg">
      <div className="mb-5">
        <h2 className="text-xl font-semibold text-white">{title}</h2>
        {subtitle && <p className="mt-1 text-sm text-slate-400">{subtitle}</p>}
      </div>
      {children}
    </section>
  );
}

function KeyValue({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="rounded-xl border border-slate-700 bg-slate-950/50 p-4">
      <p className="text-xs uppercase tracking-wide text-slate-500">{label}</p>
      <div className="mt-2 text-base font-semibold text-slate-100">{value}</div>
    </div>
  );
}

function PercentBucketList({ buckets }: { buckets?: Record<string, number> }) {
  const entries = Object.entries(buckets ?? {});

  if (entries.length === 0) {
    return <p className="text-sm text-slate-400">No exposure data available.</p>;
  }

  return (
    <div className="space-y-3">
      {entries.map(([name, percent]) => (
        <div key={name}>
          <div className="mb-1 flex items-center justify-between gap-3 text-sm">
            <span className="font-medium text-slate-200">{formatLabel(name)}</span>
            <span className="text-slate-400">{formatPercent(percent)}</span>
          </div>
          <div className="h-2 rounded-full bg-slate-800">
            <div
              className="h-2 rounded-full bg-emerald-400"
              style={{ width: `${Math.min(Math.max(percent, 0), 100)}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function StatementValidationPanel({
  validation,
}: {
  validation?: SummaryValidation | null;
}) {
  if (!validation) {
    return null;
  }

  return (
    <AnalysisCard
      title="Statement Validation"
      subtitle="Compares extracted totals with statement summary totals. Matching totals increase extraction confidence."
    >
      <div className="grid gap-4 md:grid-cols-3">
        <KeyValue
          label="Invested Value"
          value={
            <div className="space-y-1">
              <p>{formatCurrency(validation.calculated_invested_value)}</p>
              <StatusBadge label={validation.invested_value_matches ? "HIGH" : "LOW"} />
            </div>
          }
        />
        <KeyValue
          label="Current Value"
          value={
            <div className="space-y-1">
              <p>{formatCurrency(validation.calculated_current_value)}</p>
              <StatusBadge label={validation.current_value_matches ? "HIGH" : "LOW"} />
            </div>
          }
        />
        <KeyValue
          label="Gain / Loss"
          value={
            <div className="space-y-1">
              <p>{formatCurrency(validation.calculated_gain_loss)}</p>
              <StatusBadge label={validation.gain_loss_matches ? "HIGH" : "LOW"} />
            </div>
          }
        />
      </div>
    </AnalysisCard>
  );
}

function ResolutionStatusPanel({ holdings }: { holdings: ExtractedHolding[] }) {
  const resolvedCount = holdings.filter((holding) => holding.resolved).length;
  const unresolvedCount = holdings.length - resolvedCount;

  return (
    <AnalysisCard
      title="Instrument Resolution"
      subtitle="Shows whether holdings were mapped to market-data identifiers. Historical analysis runs only for confidently resolved holdings."
    >
      <div className="grid gap-4 md:grid-cols-3">
        <KeyValue label="Resolved" value={resolvedCount} />
        <KeyValue label="Unresolved" value={unresolvedCount} />
        <KeyValue label="Total" value={holdings.length} />
      </div>

      <div className="mt-5 overflow-x-auto">
        <table className="min-w-full divide-y divide-slate-800 text-sm">
          <thead className="bg-slate-950/70 text-left text-xs uppercase tracking-wide text-slate-500">
            <tr>
              <th className="px-4 py-3">Holding</th>
              <th className="px-4 py-3">Resolved</th>
              <th className="px-4 py-3">Symbol</th>
              <th className="px-4 py-3">Benchmark</th>
              <th className="px-4 py-3">Confidence</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {holdings.map((holding) => (
              <tr key={`${holding.instrument_name}-${holding.isin ?? "no-isin"}`}>
                <td className="px-4 py-3 text-slate-200">{holding.instrument_name}</td>
                <td className="px-4 py-3">
                  <StatusBadge label={holding.resolved ? "RESOLVED" : "UNRESOLVED"} />
                </td>
                <td className="px-4 py-3 text-slate-300">
                  {holding.yfinance_symbol ?? holding.resolved_symbol ?? "—"}
                </td>
                <td className="px-4 py-3 text-slate-300">
                  {formatLabel(holding.benchmark)}
                </td>
                <td className="px-4 py-3">
                  <StatusBadge label={holding.match_confidence ?? holding.confidence} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </AnalysisCard>
  );
}

function ExposureAnalysisPanel({
  analysis,
}: {
  analysis?: PortfolioExposureAnalysis | null;
}) {
  if (!analysis) {
    return null;
  }

  return (
    <AnalysisCard
      title="Portfolio Exposure Analysis"
      subtitle="Detects benchmark concentration, overlap, and diversification gaps."
    >
      <div className="grid gap-6 lg:grid-cols-2">
        <div>
          <h3 className="mb-3 font-semibold text-slate-200">Benchmark Exposure</h3>
          <PercentBucketList buckets={analysis.benchmark_exposure} />
        </div>
        <div>
          <h3 className="mb-3 font-semibold text-slate-200">Category Exposure</h3>
          <PercentBucketList buckets={analysis.category_exposure} />
        </div>
      </div>

      <div className="mt-6 grid gap-4 md:grid-cols-3">
        <KeyValue label="Total Current Value" value={formatCurrency(analysis.total_current_value)} />
        <KeyValue label="Primary Benchmark" value={formatLabel(analysis.primary_benchmark?.name)} />
        <KeyValue
          label="Resolution Coverage"
          value={formatPercent(analysis.data_quality?.resolution_coverage_percent)}
        />
      </div>

      {(analysis.overlap_warnings?.length ?? 0) > 0 && (
        <div className="mt-6 rounded-xl border border-amber-500/30 bg-amber-950/30 p-4">
          <h3 className="font-semibold text-amber-200">Overlap Warnings</h3>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-amber-100/90">
            {analysis.overlap_warnings?.map((warning) => <li key={warning}>{warning}</li>)}
          </ul>
        </div>
      )}

      {(analysis.diversification_gaps?.length ?? 0) > 0 && (
        <div className="mt-4 rounded-xl border border-sky-500/30 bg-sky-950/30 p-4">
          <h3 className="font-semibold text-sky-200">Diversification Gaps</h3>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-sky-100/90">
            {analysis.diversification_gaps?.map((gap) => <li key={gap}>{gap}</li>)}
          </ul>
        </div>
      )}
    </AnalysisCard>
  );
}

function HistoricalAnalysisPanel({
  analysis,
}: {
  analysis?: HistoricalPerformanceAnalysis | null;
}) {
  if (!analysis) {
    return null;
  }

  const visibleResults = analysis.holding_results ?? [];

  return (
    <AnalysisCard
      title="Historical Performance Analysis"
      subtitle="Runs only for confidently resolved symbols. Unresolved holdings are skipped for safety."
    >
      <div className="grid gap-4 md:grid-cols-4">
        <KeyValue label="Analyzed" value={analysis.holdings_analyzed_count ?? 0} />
        <KeyValue label="Skipped" value={analysis.holdings_skipped_count ?? 0} />
        <KeyValue label="Average Score" value={analysis.average_overall_historical_score ?? "—"} />
        <KeyValue label="Provider Scope" value={analysis.provider_scope ?? "—"} />
      </div>

      <div className="mt-5 space-y-3">
        {visibleResults.map((result) => (
          <div
            key={`${result.instrument_name}-${result.isin ?? "no-isin"}`}
            className="rounded-xl border border-slate-700 bg-slate-950/50 p-4"
          >
            <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
              <div>
                <p className="font-semibold text-slate-100">{result.instrument_name}</p>
                <p className="text-sm text-slate-500">
                  {result.yfinance_symbol ?? result.message ?? "No symbol"}
                </p>
              </div>
              <StatusBadge
                label={result.historical_analysis_available ? result.data_quality : "INSUFFICIENT"}
              />
            </div>

            {result.historical_analysis_available && (
              <div className="mt-4 grid gap-3 md:grid-cols-4">
                <KeyValue label="1Y Return" value={formatPercent(result.trailing_returns?.["1y"])} />
                <KeyValue label="3Y CAGR" value={formatPercent(result.cagr?.["3y"])} />
                <KeyValue label="Volatility" value={formatPercent(result.volatility_annualized_percent)} />
                <KeyValue label="Max Drawdown" value={formatPercent(result.max_drawdown_percent)} />
              </div>
            )}
          </div>
        ))}
      </div>
    </AnalysisCard>
  );
}

function BenchmarkComparisonPanel({
  analysis,
}: {
  analysis?: BenchmarkComparisonAnalysis | null;
}) {
  if (!analysis) {
    return null;
  }

  return (
    <AnalysisCard
      title="Benchmark Comparison"
      subtitle="Compares resolved holdings with configured benchmark/proxy symbols."
    >
      <div className="grid gap-4 md:grid-cols-3">
        <KeyValue label="Available" value={analysis.benchmark_comparisons_available_count ?? 0} />
        <KeyValue label="Skipped" value={analysis.benchmark_comparisons_skipped_count ?? 0} />
        <KeyValue label="Average Benchmark Score" value={analysis.average_benchmark_score ?? "—"} />
      </div>

      {(analysis.warnings?.length ?? 0) > 0 && (
        <ul className="mt-4 list-disc space-y-1 pl-5 text-sm text-slate-400">
          {analysis.warnings?.map((warning) => <li key={warning}>{warning}</li>)}
        </ul>
      )}
    </AnalysisCard>
  );
}

function CandidateDiscoveryPanel({
  discovery,
  resolvedCandidatesByCategory,
  resolvingCandidateCategory,
  onEvaluateCandidate,
}: {
  discovery?: ExternalCandidateDiscovery | null;
  resolvedCandidatesByCategory: Record<string, Candidate>;
  resolvingCandidateCategory: string | null;
  onEvaluateCandidate: (candidate: Candidate) => void;
}) {
  if (!discovery) {
    return null;
  }

  const shortlistedCandidates = discovery.shortlisted_candidates ?? [];

  return (
    <AnalysisCard
      title="External Candidate Discovery"
      subtitle="Candidate categories and review candidates for further analysis. These are not final buy recommendations."
    >
      <div className="grid gap-4 md:grid-cols-3">
        <KeyValue label="Scope" value={discovery.candidate_discovery_scope ?? "—"} />
        <KeyValue label="Shortlisted" value={discovery.shortlisted_candidates_count ?? 0} />
        <KeyValue label="Watchlist" value={discovery.watchlist_candidates_count ?? 0} />
      </div>

      {shortlistedCandidates.length === 0 && (
        <p className="mt-5 text-sm text-slate-400">No shortlisted external candidates available.</p>
      )}

      <div className="mt-5 space-y-4">
        {shortlistedCandidates.map((candidate, index) => {
          const candidateCategory = candidate.candidate_category ?? "";
          const resolvedCandidate =
            resolvedCandidatesByCategory[candidateCategory] ?? candidate;
          const rankedInstruments = resolvedCandidate.ranked_candidate_instruments ?? [];
          const topCandidate =
            resolvedCandidate.top_candidate_instrument ?? rankedInstruments[0];
          const isResolving = resolvingCandidateCategory === candidateCategory;

          return (
            <div
              key={candidate.candidate_id ?? `${candidate.candidate_category}-${index}`}
              className="rounded-xl border border-slate-700 bg-slate-950/50 p-4"
            >
              <div className="flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
                <div>
                  <p className="font-semibold text-slate-100">
                    {resolvedCandidate.instrument_name ?? formatLabel(resolvedCandidate.candidate_category)}
                  </p>
                  <p className="mt-1 text-sm text-slate-400">{resolvedCandidate.reason_considered}</p>

                  {resolvedCandidate.category_review_status && (
                    <div className="mt-3 flex flex-wrap gap-2">
                      <StatusBadge label={resolvedCandidate.category_review_status} />
                      {resolvedCandidate.risk_bucket && <StatusBadge label={resolvedCandidate.risk_bucket} />}
                    </div>
                  )}
                </div>

                <div className="text-left md:text-right">
                  <p className="text-sm text-slate-400">Gap / Review score</p>
                  <p className="font-semibold text-emerald-300">
                    {resolvedCandidate.top_final_candidate_review_score ??
                      resolvedCandidate.portfolio_gap_score ??
                      resolvedCandidate.candidate_final_score ??
                      "—"}
                  </p>
                </div>
              </div>

              <div className="mt-4 flex flex-wrap gap-3">
                <button
                  type="button"
                  onClick={() => onEvaluateCandidate(candidate)}
                  disabled={isResolving || !candidateCategory}
                  className="rounded-lg bg-emerald-500 px-4 py-2 text-xs font-semibold text-slate-950 transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
                >
                  {isResolving ? "Evaluating..." : "Evaluate Candidate"}
                </button>
              </div>

              {resolvedCandidate.category_review_note && (
                <p className="mt-3 rounded-lg border border-sky-500/30 bg-sky-950/20 p-3 text-xs text-sky-100">
                  {resolvedCandidate.category_review_note}
                </p>
              )}

              <div className="mt-4 grid gap-3 md:grid-cols-4">
                <KeyValue
                  label="Verified"
                  value={resolvedCandidate.verified_candidate_instruments_count ?? 0}
                />
                <KeyValue
                  label="Analyzed"
                  value={resolvedCandidate.analyzed_instruments_count ?? 0}
                />
                <KeyValue
                  label="Benchmark Pending"
                  value={resolvedCandidate.benchmark_pending_instruments_count ?? 0}
                />
                <KeyValue
                  label="Placeholders"
                  value={resolvedCandidate.placeholder_instruments_count ?? 0}
                />
              </div>

              {(resolvedCandidate.candidate_flags?.length ?? 0) > 0 && (
                <div className="mt-3 flex flex-wrap gap-2">
                  {resolvedCandidate.candidate_flags?.map((flag) => <StatusBadge key={flag} label={flag} />)}
                </div>
              )}

              {topCandidate && (
                <div className="mt-5 rounded-xl border border-emerald-500/30 bg-emerald-950/10 p-4">
                  <div className="flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
                    <div>
                      <p className="text-xs uppercase tracking-wide text-emerald-300">
                        Top review candidate
                      </p>
                      <h3 className="mt-1 font-semibold text-slate-100">
                        {topCandidate.instrument_name ?? "Unnamed candidate"}
                      </h3>
                      <p className="mt-1 text-xs text-slate-400">
                        {formatLabel(topCandidate.instrument_type)} ·{" "}
                        {topCandidate.yfinance_symbol ??
                          topCandidate.nse_symbol ??
                          topCandidate.amfi_scheme_code ??
                          topCandidate.isin ??
                          "No symbol"}
                      </p>
                    </div>

                    <StatusBadge label={topCandidate.final_recommendation_status} />
                  </div>

                  <div className="mt-4 grid gap-3 md:grid-cols-4">
                    <KeyValue
                      label="Historical Score"
                      value={formatNumber(topCandidate.candidate_analysis_score)}
                    />
                    <KeyValue
                      label="Profile Score"
                      value={formatNumber(topCandidate.profile_suitability_score)}
                    />
                    <KeyValue
                      label="Final Review Score"
                      value={formatNumber(topCandidate.final_candidate_review_score)}
                    />
                    <KeyValue
                      label="Historical Data"
                      value={
                        <StatusBadge
                          label={
                            topCandidate.candidate_historical_analysis
                              ?.historical_analysis_available
                              ? topCandidate.candidate_historical_analysis?.data_quality
                              : "INSUFFICIENT"
                          }
                        />
                      }
                    />
                  </div>

                  {topCandidate.review_score_note && (
                    <p className="mt-3 text-xs text-slate-400">{topCandidate.review_score_note}</p>
                  )}

                  {topCandidate.candidate_benchmark_comparison && (
                    <div className="mt-3 rounded-lg border border-slate-700 bg-slate-950/50 p-3 text-xs text-slate-300">
                      <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                        <span>
                          Benchmark:{" "}
                          {formatLabel(topCandidate.candidate_benchmark_comparison.benchmark)}
                        </span>
                        <StatusBadge
                          label={
                            topCandidate.candidate_benchmark_comparison
                              .benchmark_comparison_available
                              ? "AVAILABLE"
                              : "BENCHMARK_PENDING"
                          }
                        />
                      </div>
                      {topCandidate.candidate_benchmark_comparison.message && (
                        <p className="mt-2 text-slate-500">
                          {topCandidate.candidate_benchmark_comparison.message}
                        </p>
                      )}
                    </div>
                  )}

                  {topCandidate.mf_review_required && (
                    <div className="mt-4 rounded-lg border border-amber-500/30 bg-amber-950/20 p-3">
                      <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                        <h4 className="font-semibold text-amber-200">Mutual fund review checks</h4>
                        <StatusBadge label={topCandidate.mf_review_status} />
                      </div>

                      <div className="mt-3 grid gap-2 md:grid-cols-2">
                        {Object.entries(topCandidate.mf_quality_checks ?? {}).map(([key, check]) => (
                          <div
                            key={key}
                            className="rounded-lg border border-slate-700 bg-slate-950/50 p-3"
                          >
                            <div className="flex items-center justify-between gap-3">
                              <span className="text-sm font-medium text-slate-200">
                                {formatLabel(key)}
                              </span>
                              <StatusBadge label={check.status} />
                            </div>
                            <p className="mt-1 text-xs text-slate-400">
                              Value: {check.value ?? "—"}
                            </p>
                            {check.note && (
                              <p className="mt-1 text-xs text-slate-500">{check.note}</p>
                            )}
                          </div>
                        ))}
                      </div>

                      {(topCandidate.mf_review_warnings?.length ?? 0) > 0 && (
                        <ul className="mt-3 list-disc space-y-1 pl-5 text-xs text-amber-100/90">
                          {topCandidate.mf_review_warnings?.map((warning) => (
                            <li key={warning}>{warning}</li>
                          ))}
                        </ul>
                      )}
                    </div>
                  )}
                </div>
              )}

              {rankedInstruments.length > 1 && (
                <div className="mt-4 overflow-x-auto">
                  <table className="min-w-full divide-y divide-slate-800 text-sm">
                    <thead className="bg-slate-950/70 text-left text-xs uppercase tracking-wide text-slate-500">
                      <tr>
                        <th className="px-4 py-3">Rank</th>
                        <th className="px-4 py-3">Candidate</th>
                        <th className="px-4 py-3">Type</th>
                        <th className="px-4 py-3">Historical</th>
                        <th className="px-4 py-3">Profile</th>
                        <th className="px-4 py-3">Final</th>
                        <th className="px-4 py-3">Status</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800">
                      {rankedInstruments.map((instrument) => (
                        <tr
                          key={`${instrument.instrument_name}-${instrument.candidate_rank ?? "rank"}`}
                        >
                          <td className="px-4 py-3 text-slate-300">
                            {instrument.candidate_rank ?? "—"}
                          </td>
                          <td className="px-4 py-3 text-slate-200">
                            {instrument.instrument_name ?? "—"}
                          </td>
                          <td className="px-4 py-3 text-slate-300">
                            {formatLabel(instrument.instrument_type)}
                          </td>
                          <td className="px-4 py-3 text-slate-300">
                            {formatNumber(instrument.candidate_analysis_score)}
                          </td>
                          <td className="px-4 py-3 text-slate-300">
                            {formatNumber(instrument.profile_suitability_score)}
                          </td>
                          <td className="px-4 py-3 text-emerald-300">
                            {formatNumber(instrument.final_candidate_review_score)}
                          </td>
                          <td className="px-4 py-3">
                            <StatusBadge label={instrument.final_recommendation_status} />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {(resolvedCandidate.resolved_candidate_instruments?.length ?? 0) > 0 && rankedInstruments.length === 0 && (
                <p className="mt-3 text-xs text-emerald-300">
                  {resolvedCandidate.resolved_candidate_instruments?.length} candidate instrument(s) resolved for further checks.
                </p>
              )}
            </div>
          );
        })}
      </div>
    </AnalysisCard>
  );
}

function BackendRecommendationPanel({
  recommendation,
}: {
  recommendation?: BackendRecommendation | null;
}) {
  if (!recommendation) {
    return null;
  }

  return (
    <AnalysisCard
      title="Backend Educational Recommendation"
      subtitle="Generated by backend scoring rules. This is not a direct buy/sell instruction."
    >
      <div className="grid gap-4 md:grid-cols-4">
        <KeyValue label="Suggested Action" value={formatLabel(recommendation.suggested_action)} />
        <KeyValue label="Suggested Amount" value={formatCurrency(recommendation.suggested_amount)} />
        <KeyValue label="Final Score" value={formatNumber(recommendation.final_recommendation_score)} />
        <KeyValue label="Confidence" value={<StatusBadge label={recommendation.confidence_level} />} />
      </div>

      <div className="mt-6 grid gap-4 md:grid-cols-2">
        <div className="rounded-xl border border-slate-700 bg-slate-950/50 p-4">
          <h3 className="font-semibold text-slate-200">Score Breakdown</h3>
          <div className="mt-3 space-y-2 text-sm text-slate-300">
            {Object.entries(recommendation.score_breakdown ?? {}).map(([key, value]) => (
              <div key={key} className="flex justify-between gap-4">
                <span>{formatLabel(key)}</span>
                <span>{formatNumber(value)}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-xl border border-slate-700 bg-slate-950/50 p-4">
          <h3 className="font-semibold text-slate-200">Reason Codes</h3>
          <div className="mt-3 flex flex-wrap gap-2">
            {(recommendation.reason_codes ?? []).map((reason) => (
              <StatusBadge key={reason} label={reason} />
            ))}
          </div>
        </div>
      </div>

      <div className="mt-6 rounded-xl border border-emerald-500/30 bg-emerald-950/20 p-4">
        <h3 className="font-semibold text-emerald-200">Allocation Plan</h3>
        <div className="mt-3 space-y-3">
          {(recommendation.allocation_plan ?? []).map((item, index) => (
            <div
              key={`${item.candidate_category ?? "item"}-${index}`}
              className="rounded-lg border border-slate-700 bg-slate-950/60 p-4"
            >
              <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                <div>
                  <p className="font-semibold text-slate-100">
                    {formatLabel(item.candidate_category ?? item.allocation_type)}
                  </p>
                  <p className="mt-1 text-sm text-slate-400">{item.reason}</p>
                </div>
                <div className="text-left md:text-right">
                  <p className="font-semibold text-emerald-300">{formatCurrency(item.amount)}</p>
                  <StatusBadge label={item.status} />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {recommendation.data_quality_note && (
        <p className="mt-4 text-xs text-slate-500">{recommendation.data_quality_note}</p>
      )}
      {recommendation.risk_note && (
        <p className="mt-2 text-xs text-slate-500">{recommendation.risk_note}</p>
      )}
    </AnalysisCard>
  );
}

function RecommendationExplanationPanel({
  explanation,
}: {
  explanation?: RecommendationExplanation | null;
}) {
  if (!explanation) {
    return null;
  }

  return (
    <AnalysisCard
      title="AI / Fallback Explanation"
      subtitle="Plain-language explanation of the backend recommendation. AI does not override backend logic."
    >
      <div className="mb-4">
        <StatusBadge label={explanation.explanation_available ? "AVAILABLE" : "LIMITED"} />
      </div>

      <p className="text-slate-200">{explanation.summary}</p>

      {(explanation.why?.length ?? 0) > 0 && (
        <div className="mt-5">
          <h3 className="font-semibold text-slate-200">Why</h3>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-300">
            {explanation.why?.map((item) => <li key={item}>{item}</li>)}
          </ul>
        </div>
      )}

      {(explanation.cautions?.length ?? 0) > 0 && (
        <div className="mt-5 rounded-xl border border-rose-500/30 bg-rose-950/20 p-4">
          <h3 className="font-semibold text-rose-200">Cautions</h3>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-rose-100/90">
            {explanation.cautions?.map((item) => <li key={item}>{item}</li>)}
          </ul>
        </div>
      )}

      {explanation.ai_error && (
        <p className="mt-4 text-xs text-amber-300">AI fallback used: {explanation.ai_error}</p>
      )}
    </AnalysisCard>
  );
}

function AnalysisPanels({
  extractionData,
  resolvedCandidatesByCategory,
  resolvingCandidateCategory,
  onEvaluateCandidate,
}: {
  extractionData: ExtractionData;
  resolvedCandidatesByCategory: Record<string, Candidate>;
  resolvingCandidateCategory: string | null;
  onEvaluateCandidate: (candidate: Candidate) => void;
}) {
  return (
    <div className="mt-8 space-y-6">
      <StatementValidationPanel validation={extractionData.summary_validation} />
      <ResolutionStatusPanel holdings={extractionData.valid_holdings} />
      <ExposureAnalysisPanel analysis={extractionData.portfolio_exposure_analysis} />
      <HistoricalAnalysisPanel analysis={extractionData.historical_performance_analysis} />
      <BenchmarkComparisonPanel analysis={extractionData.benchmark_comparison_analysis} />
      <CandidateDiscoveryPanel
        discovery={extractionData.external_candidate_discovery}
        resolvedCandidatesByCategory={resolvedCandidatesByCategory}
        resolvingCandidateCategory={resolvingCandidateCategory}
        onEvaluateCandidate={onEvaluateCandidate}
      />
      <BackendRecommendationPanel recommendation={extractionData.backend_recommendation} />
      <RecommendationExplanationPanel explanation={extractionData.recommendation_explanation} />
      <ReviewBeforeActionCard />
    </div>
  );
}

function ReviewBeforeActionCard() {
  return (
    <div className="rounded-2xl border border-amber-500/30 bg-amber-950/30 p-5 text-sm text-amber-100">
      <h3 className="font-semibold text-amber-200">Review before import or action</h3>
      <ul className="mt-3 list-disc space-y-1 pl-5">
        <li>Check that extracted holdings match the actual statement.</li>
        <li>Check unresolved holdings and confidence levels before relying on analysis.</li>
        <li>Do not treat category-level allocation as a direct buy instruction.</li>
        <li>
          Instrument-level checks, expense ratio, liquidity, tax impact, and personal suitability
          should be reviewed before investing.
        </li>
      </ul>
    </div>
  );
}

export default function UploadPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [extractionData, setExtractionData] = useState<ExtractionData | null>(
    null
  );
  const [importData, setImportData] = useState<ImportData | null>(null);
  const [isExtracting, setIsExtracting] = useState(false);
  const [isImporting, setIsImporting] = useState(false);
  const [resolvedCandidatesByCategory, setResolvedCandidatesByCategory] = useState<
    Record<string, Candidate>
  >({});
  const [resolvingCandidateCategory, setResolvingCandidateCategory] = useState<
    string | null
  >(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [uploadNoticeMessage, setUploadNoticeMessage] = useState<string | null>(
    null
  );

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;

    setSelectedFile(file);
    setExtractionData(null);
    setImportData(null);
    setResolvedCandidatesByCategory({});
    setResolvingCandidateCategory(null);
    setErrorMessage(null);

    if (!file) {
      setUploadNoticeMessage(null);
      return;
    }

    const validationError = validateUploadFile(file);

    if (validationError) {
      setErrorMessage(validationError);
      setUploadNoticeMessage(null);
      return;
    }

    if (isPdfFile(file)) {
      setUploadNoticeMessage(PDF_TEXT_EXTRACTION_NOTE);
      return;
    }

    setUploadNoticeMessage(null);
  }

  async function handleExtract() {
    if (!selectedFile) {
      setErrorMessage("Please select a portfolio statement file first.");
      return;
    }

    const validationError = validateUploadFile(selectedFile);

    if (validationError) {
      setErrorMessage(validationError);
      return;
    }

    setIsExtracting(true);
    setErrorMessage(null);
    setExtractionData(null);
    setImportData(null);
    setResolvedCandidatesByCategory({});
    setResolvingCandidateCategory(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch("/api/portfolio/uploads/file/extract", {
        method: "POST",
        body: formData,
      });

      const result = (await response.json()) as ApiResponse<ExtractionData>;

      if (!response.ok) {
        throw new Error(result.detail ?? "Failed to extract portfolio file.");
      }

      if (!result.data) {
        throw new Error("Extraction response did not contain data.");
      }

      setExtractionData(result.data);
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "Unknown extraction error."
      );
    } finally {
      setIsExtracting(false);
    }
  }

  async function handleEvaluateCandidate(candidate: Candidate) {
    const candidateCategory = candidate.candidate_category;

    if (!candidateCategory) {
      setErrorMessage("Candidate category is missing.");
      return;
    }

    setResolvingCandidateCategory(candidateCategory);
    setErrorMessage(null);

    try {
      const response = await fetch("/api/candidates/resolve", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          candidate_id: candidate.candidate_id,
          candidate_category: candidateCategory,
          risk_appetite: "MODERATE",
          time_horizon_years: 5,
          monthly_investment_amount: 2000,
          include_analysis: true,
        }),
      });

      const result = (await response.json()) as CandidateResolveResponse;

      if (!response.ok || !result.success) {
        throw new Error(result.detail ?? "Failed to evaluate candidate.");
      }

      const resolvedCandidate = result.resolved_candidates?.[0];

      if (!resolvedCandidate) {
        throw new Error("Candidate resolve response did not include a resolved candidate.");
      }

      setResolvedCandidatesByCategory((current) => ({
        ...current,
        [candidateCategory]: resolvedCandidate,
      }));
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "Unknown candidate evaluation error."
      );
    } finally {
      setResolvingCandidateCategory(null);
    }
  }

  async function handleImportReviewed() {
    if (!extractionData || extractionData.valid_holdings.length === 0) {
      setErrorMessage("No valid holdings available to import.");
      return;
    }

    setIsImporting(true);
    setErrorMessage(null);
    setImportData(null);

    try {
      const response = await fetch("/api/portfolio/uploads/import-reviewed", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          holdings: extractionData.valid_holdings,
        }),
      });

      const result = (await response.json()) as ApiResponse<ImportData>;

      if (!response.ok) {
        throw new Error(result.detail ?? "Failed to import reviewed holdings.");
      }

      if (!result.data) {
        throw new Error("Import response did not contain data.");
      }

      setImportData(result.data);
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "Unknown import error."
      );
    } finally {
      setIsImporting(false);
    }
  }

  return (
    <main className="min-h-screen bg-slate-950 px-6 py-10 text-white">
      <section className="mx-auto max-w-7xl space-y-8">
        <section className="rounded-2xl border border-emerald-500/30 bg-slate-900 p-8 shadow-xl">
          <p className="text-sm uppercase tracking-wide text-emerald-300">
            Portfolio Import
          </p>

          <div className="mt-3 flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
            <div>
              <h1 className="text-4xl font-bold tracking-tight">
                Upload Portfolio Statement
              </h1>

              <p className="mt-4 max-w-3xl text-slate-300">
                Upload a CSV, XLSX, XLS, TXT, PDF, or XML portfolio statement.
                The backend will extract holdings, validate rows, analyze
                exposure, and generate an educational recommendation preview.
              </p>
            </div>

            <div className="rounded-full border border-emerald-500/30 bg-emerald-950/40 px-4 py-2 text-xs font-medium text-emerald-200">
              Educational Analysis MVP
            </div>
          </div>

          <div className="mt-8 rounded-2xl border border-slate-700 bg-slate-950/60 p-5">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-center">
              <input
                type="file"
                accept=".csv,.xlsx,.xls,.txt,.pdf,.xml"
                onChange={handleFileChange}
                className="block w-full rounded-lg border border-slate-700 bg-slate-900 p-3 text-sm text-slate-200 file:mr-4 file:rounded-md file:border-0 file:bg-emerald-500 file:px-4 file:py-2 file:text-sm file:font-semibold file:text-slate-950 hover:file:bg-emerald-400"
              />

              <button
                type="button"
                onClick={handleExtract}
                disabled={!selectedFile || isExtracting}
                className="inline-flex min-w-40 justify-center rounded-lg bg-emerald-500 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
              >
                {isExtracting ? "Extracting..." : "Extract Holdings"}
              </button>
            </div>

            {selectedFile && (
              <p className="mt-4 text-sm text-slate-300">
                Selected file:{" "}
                <span className="font-semibold text-emerald-300">
                  {selectedFile.name}
                </span>{" "}
                <span className="text-slate-500">
                  ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
                </span>
              </p>
            )}

            <p className="mt-3 text-xs text-slate-500">
              Tip: CSV/XLSX/XLS is parsed deterministically. TXT, XML, and
              text-readable PDFs can use Gemini extraction. Review rows and
              analysis before import or investment action.
            </p>
          </div>

          {uploadNoticeMessage && (
            <div className="mt-6 rounded-xl border border-amber-500/40 bg-amber-950/30 p-4 text-sm text-amber-100">
              <p className="font-semibold text-amber-200">Upload note</p>
              <p className="mt-1">{uploadNoticeMessage}</p>
            </div>
          )}

          {errorMessage && (
            <div className="mt-6 rounded-xl border border-red-500/40 bg-red-950/40 p-4 text-sm text-red-100">
              <p className="font-semibold text-red-200">Something went wrong</p>
              <p className="mt-1">{errorMessage}</p>
            </div>
          )}
        </section>

        {extractionData && (
          <section className="rounded-2xl border border-slate-700 bg-slate-900 p-8 shadow-xl">
            <div className="flex flex-col justify-between gap-5 lg:flex-row lg:items-start">
              <div>
                <p className="text-sm uppercase tracking-wide text-slate-400">
                  Extraction Result
                </p>

                <h2 className="mt-2 text-2xl font-semibold text-white">
                  Holdings extracted from statement
                </h2>

                <p className="mt-2 text-sm text-slate-400">
                  Method:{" "}
                  <span className="font-medium text-emerald-300">
                    {extractionData.extraction_method}
                  </span>{" "}
                  | File:{" "}
                  <span className="font-medium text-slate-200">
                    {extractionData.file_name}
                  </span>
                </p>
              </div>

              <button
                type="button"
                onClick={handleImportReviewed}
                disabled={isImporting || extractionData.valid_holdings.length === 0}
                className="inline-flex justify-center rounded-lg bg-emerald-500 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
              >
                {isImporting ? "Importing..." : "Import Reviewed Holdings"}
              </button>
            </div>

            <div className="mt-6 grid gap-4 md:grid-cols-3">
              <SummaryCard
                label="Holdings detected"
                value={extractionData.holdings_detected}
                tone="neutral"
              />
              <SummaryCard
                label="Valid holdings"
                value={extractionData.valid_holdings_count}
                tone="success"
              />
              <SummaryCard
                label="Invalid holdings"
                value={extractionData.invalid_holdings_count}
                tone={extractionData.invalid_holdings_count > 0 ? "warning" : "neutral"}
              />
            </div>

            {extractionData.warnings.length > 0 && (
              <div className="mt-6 rounded-xl border border-amber-500/40 bg-amber-950/30 p-4 text-sm text-amber-100">
                <p className="font-semibold text-amber-200">Warnings</p>
                <ul className="mt-2 list-disc space-y-1 pl-5">
                  {extractionData.warnings.map((warning) => (
                    <li key={warning}>{warning}</li>
                  ))}
                </ul>
              </div>
            )}

            <HoldingsTable holdings={extractionData.valid_holdings} />

            {extractionData.invalid_holdings.length > 0 && (
              <InvalidHoldingsTable invalidHoldings={extractionData.invalid_holdings} />
            )}

            <AnalysisPanels
              extractionData={extractionData}
              resolvedCandidatesByCategory={resolvedCandidatesByCategory}
              resolvingCandidateCategory={resolvingCandidateCategory}
              onEvaluateCandidate={handleEvaluateCandidate}
            />

            <div className="mt-8 flex flex-wrap gap-3">
              <button
                type="button"
                onClick={handleImportReviewed}
                disabled={isImporting || extractionData.valid_holdings.length === 0}
                className="inline-flex justify-center rounded-lg bg-emerald-500 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
              >
                {isImporting ? "Importing..." : "Import Reviewed Holdings"}
              </button>
            </div>
          </section>
        )}

        {importData && (
          <section className="rounded-2xl border border-emerald-500/40 bg-emerald-950/30 p-8 shadow-xl">
            <p className="text-sm uppercase tracking-wide text-emerald-300">
              Import Completed
            </p>

            <h2 className="mt-2 text-2xl font-semibold text-white">
              Reviewed holdings imported successfully
            </h2>

            <div className="mt-6 grid gap-4 md:grid-cols-3">
              <SummaryCard
                label="Holdings received"
                value={importData.holdings_received}
                tone="neutral"
              />
              <SummaryCard
                label="Holdings imported"
                value={importData.holdings_imported}
                tone="success"
              />
              <SummaryCard
                label="Replaced old snapshot rows"
                value={importData.deleted_existing_holdings_for_snapshot}
                tone="warning"
              />
            </div>

            <div className="mt-6 rounded-xl border border-slate-700 bg-slate-950/60 p-4 text-sm text-slate-300">
              <p>
                Snapshot date:{" "}
                <span className="font-semibold text-emerald-300">
                  {importData.snapshot_date}
                </span>
              </p>
              <p className="mt-1">
                Source upload ID:{" "}
                <span className="font-semibold text-slate-100">
                  {importData.source_upload_id}
                </span>
              </p>
            </div>

            <div className="mt-6 flex flex-wrap gap-3">
              <a
                href="/portfolio"
                className="rounded-lg border border-slate-600 px-5 py-3 text-sm font-semibold text-slate-100 transition hover:border-emerald-400 hover:text-emerald-300"
              >
                View Portfolio
              </a>

              <a
                href="/recommendations"
                className="rounded-lg bg-emerald-500 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-emerald-400"
              >
                Generate Recommendation
              </a>

              <a
                href="/explanations"
                className="rounded-lg border border-slate-600 px-5 py-3 text-sm font-semibold text-slate-100 transition hover:border-emerald-400 hover:text-emerald-300"
              >
                Explain Recommendation
              </a>
            </div>
          </section>
        )}
      </section>
    </main>
  );
}

function SummaryCard({
  label,
  value,
  tone,
}: {
  label: string;
  value: string | number;
  tone: "neutral" | "success" | "warning";
}) {
  const toneClassName = {
    neutral: "border-slate-700 bg-slate-950/60 text-slate-100",
    success: "border-emerald-500/30 bg-emerald-950/30 text-emerald-200",
    warning: "border-amber-500/30 bg-amber-950/30 text-amber-200",
  }[tone];

  return (
    <div className={`rounded-xl border p-5 ${toneClassName}`}>
      <p className="text-sm text-slate-400">{label}</p>
      <p className="mt-2 text-3xl font-bold">{value}</p>
    </div>
  );
}

function HoldingsTable({ holdings }: { holdings: ExtractedHolding[] }) {
  if (holdings.length === 0) {
    return (
      <div className="mt-6 rounded-xl border border-slate-700 bg-slate-950/60 p-5 text-sm text-slate-400">
        No valid holdings found.
      </div>
    );
  }

  return (
    <div className="mt-8 overflow-hidden rounded-xl border border-slate-700">
      <div className="border-b border-slate-700 bg-slate-950/80 px-5 py-4">
        <h3 className="text-lg font-semibold text-white">Valid Holdings</h3>
        <p className="mt-1 text-sm text-slate-400">
          These rows passed backend validation and are ready for reviewed import.
        </p>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full border-collapse text-left text-sm">
          <thead className="bg-slate-800 text-slate-300">
            <tr>
              <th className="p-3">Instrument</th>
              <th className="p-3">Type</th>
              <th className="p-3">ISIN</th>
              <th className="p-3">Resolved Symbol</th>
              <th className="p-3 text-right">Qty</th>
              <th className="p-3 text-right">Avg Cost</th>
              <th className="p-3 text-right">Invested</th>
              <th className="p-3 text-right">Current</th>
              <th className="p-3 text-right">P&L</th>
              <th className="p-3">Confidence</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800 bg-slate-900 text-slate-200">
            {holdings.map((holding, index) => (
              <tr
                key={`${holding.instrument_name}-${holding.isin ?? index}`}
                className="hover:bg-slate-800/70"
              >
                <td className="p-3 font-medium text-white">{holding.instrument_name}</td>
                <td className="p-3">{holding.instrument_type}</td>
                <td className="p-3">{holding.isin || "-"}</td>
                <td className="p-3">{holding.yfinance_symbol ?? holding.resolved_symbol ?? holding.symbol ?? "-"}</td>
                <td className="p-3 text-right">{formatNumber(holding.quantity)}</td>
                <td className="p-3 text-right">{formatCurrency(holding.average_cost)}</td>
                <td className="p-3 text-right">{formatCurrency(holding.invested_amount)}</td>
                <td className="p-3 text-right">{formatCurrency(holding.current_value)}</td>
                <td className="p-3 text-right">{formatCurrency(holding.gain_loss)}</td>
                <td className="p-3">
                  <StatusBadge label={holding.match_confidence ?? holding.confidence ?? "REVIEWED"} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function InvalidHoldingsTable({
  invalidHoldings,
}: {
  invalidHoldings: InvalidExtractedHolding[];
}) {
  return (
    <div className="mt-8 overflow-hidden rounded-xl border border-red-500/40">
      <div className="border-b border-red-500/30 bg-red-950/40 px-5 py-4">
        <h3 className="text-lg font-semibold text-red-100">Invalid Holdings</h3>
        <p className="mt-1 text-sm text-red-200">
          These rows need correction before they can be imported.
        </p>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full border-collapse text-left text-sm">
          <thead className="bg-red-950/60 text-red-100">
            <tr>
              <th className="p-3">Row</th>
              <th className="p-3">Instrument</th>
              <th className="p-3">Errors</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-red-900/50 bg-slate-900 text-red-100">
            {invalidHoldings.map((item) => (
              <tr key={item.row_number}>
                <td className="p-3">{item.row_number}</td>
                <td className="p-3">{item.holding.instrument_name ?? "Unknown"}</td>
                <td className="p-3">
                  <ul className="list-disc space-y-1 pl-5">
                    {item.errors.map((error) => (
                      <li key={error}>{error}</li>
                    ))}
                  </ul>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
