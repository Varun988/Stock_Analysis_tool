"use client";

import { ChangeEvent, useState } from "react";

type ExtractedHolding = {
  instrument_id?: string | null;
  instrument_name: string;
  instrument_type: string;
  symbol?: string | null;
  isin?: string | null;
  quantity: number;
  average_cost: number;
  invested_amount: number;
  current_value: number;
  confidence?: string | null;
};

type InvalidExtractedHolding = {
  row_number: number;
  holding: Partial<ExtractedHolding>;
  errors: string[];
};

type ExtractionData = {
  file_name: string;
  extraction_method: string;
  holdings_detected: number;
  valid_holdings_count: number;
  invalid_holdings_count: number;
  valid_holdings: ExtractedHolding[];
  invalid_holdings: InvalidExtractedHolding[];
  warnings: string[];
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

export default function UploadPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [extractionData, setExtractionData] = useState<ExtractionData | null>(
    null
  );
  const [importData, setImportData] = useState<ImportData | null>(null);
  const [isExtracting, setIsExtracting] = useState(false);
  const [isImporting, setIsImporting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;

    setSelectedFile(file);
    setExtractionData(null);
    setImportData(null);
    setErrorMessage(null);
  }

  async function handleExtract() {
    if (!selectedFile) {
      setErrorMessage("Please select a portfolio statement file first.");
      return;
    }

    setIsExtracting(true);
    setErrorMessage(null);
    setExtractionData(null);
    setImportData(null);

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
                Upload a CSV, XLSX, or TXT portfolio statement. The backend will
                extract holdings, validate the rows, and let you import reviewed
                holdings into the latest portfolio snapshot.
              </p>
            </div>

            <div className="rounded-full border border-emerald-500/30 bg-emerald-950/40 px-4 py-2 text-xs font-medium text-emerald-200">
              Statement Upload MVP
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
                </span>
              </p>
            )}

            <p className="mt-3 text-xs text-slate-500">
              Tip: CSV/XLSX is parsed deterministically. TXT/unstructured
              statements can use Gemini extraction. Review rows before import.
            </p>
          </div>

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
                disabled={
                  isImporting || extractionData.valid_holdings.length === 0
                }
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
                tone={
                  extractionData.invalid_holdings_count > 0
                    ? "warning"
                    : "neutral"
                }
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
              <InvalidHoldingsTable
                invalidHoldings={extractionData.invalid_holdings}
              />
            )}
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
              <th className="p-3">Symbol</th>
              <th className="p-3">ISIN</th>
              <th className="p-3 text-right">Qty</th>
              <th className="p-3 text-right">Avg Cost</th>
              <th className="p-3 text-right">Invested</th>
              <th className="p-3 text-right">Current</th>
              <th className="p-3">Confidence</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800 bg-slate-900 text-slate-200">
            {holdings.map((holding, index) => (
              <tr
                key={`${holding.instrument_name}-${index}`}
                className="hover:bg-slate-800/70"
              >
                <td className="p-3 font-medium text-white">
                  {holding.instrument_name}
                </td>
                <td className="p-3">{holding.instrument_type}</td>
                <td className="p-3">{holding.symbol || "-"}</td>
                <td className="p-3">{holding.isin || "-"}</td>
                <td className="p-3 text-right">{holding.quantity}</td>
                <td className="p-3 text-right">₹{holding.average_cost}</td>
                <td className="p-3 text-right">₹{holding.invested_amount}</td>
                <td className="p-3 text-right">₹{holding.current_value}</td>
                <td className="p-3">
                  <span className="rounded-full border border-emerald-500/30 bg-emerald-950/40 px-2 py-1 text-xs font-medium text-emerald-200">
                    {holding.confidence ?? "REVIEWED"}
                  </span>
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
        <h3 className="text-lg font-semibold text-red-100">
          Invalid Holdings
        </h3>
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
                <td className="p-3">
                  {item.holding.instrument_name ?? "Unknown"}
                </td>
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