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
    <main className="mx-auto max-w-6xl space-y-6 p-6">
      <section className="rounded-lg border bg-white p-6 shadow-sm">
        <h1 className="text-2xl font-semibold text-gray-900">
          Upload Portfolio Statement
        </h1>
        <p className="mt-2 text-sm text-gray-600">
          Upload a CSV, XLSX, or TXT portfolio statement. The backend will
          extract holdings, validate them, and let you import reviewed rows into
          the latest portfolio snapshot.
        </p>

        <div className="mt-6 flex flex-col gap-4 sm:flex-row sm:items-center">
          <input
            type="file"
            accept=".csv,.xlsx,.xls,.txt,.pdf,.xml"
            onChange={handleFileChange}
            className="block w-full rounded-md border border-gray-300 p-2 text-sm"
          />

          <button
            type="button"
            onClick={handleExtract}
            disabled={!selectedFile || isExtracting}
            className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-gray-400"
          >
            {isExtracting ? "Extracting..." : "Extract Holdings"}
          </button>
        </div>

        {selectedFile && (
          <p className="mt-3 text-sm text-gray-700">
            Selected file:{" "}
            <span className="font-medium">{selectedFile.name}</span>
          </p>
        )}

        {errorMessage && (
          <div className="mt-4 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">
            {errorMessage}
          </div>
        )}
      </section>

      {extractionData && (
        <section className="rounded-lg border bg-white p-6 shadow-sm">
          <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
            <div>
              <h2 className="text-xl font-semibold text-gray-900">
                Extraction Result
              </h2>
              <p className="mt-1 text-sm text-gray-600">
                Method:{" "}
                <span className="font-medium">
                  {extractionData.extraction_method}
                </span>{" "}
                | File:{" "}
                <span className="font-medium">{extractionData.file_name}</span>
              </p>
            </div>

            <button
              type="button"
              onClick={handleImportReviewed}
              disabled={
                isImporting || extractionData.valid_holdings.length === 0
              }
              className="rounded-md bg-green-600 px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-gray-400"
            >
              {isImporting ? "Importing..." : "Import Reviewed Holdings"}
            </button>
          </div>

          <div className="mt-6 grid gap-4 sm:grid-cols-3">
            <SummaryCard
              label="Holdings detected"
              value={extractionData.holdings_detected}
            />
            <SummaryCard
              label="Valid holdings"
              value={extractionData.valid_holdings_count}
            />
            <SummaryCard
              label="Invalid holdings"
              value={extractionData.invalid_holdings_count}
            />
          </div>

          {extractionData.warnings.length > 0 && (
            <div className="mt-4 rounded-md border border-yellow-200 bg-yellow-50 p-3 text-sm text-yellow-800">
              <p className="font-medium">Warnings</p>
              <ul className="mt-2 list-disc pl-5">
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
        <section className="rounded-lg border border-green-200 bg-green-50 p-6 shadow-sm">
          <h2 className="text-xl font-semibold text-green-900">
            Import Completed
          </h2>
          <div className="mt-4 grid gap-4 sm:grid-cols-3">
            <SummaryCard
              label="Holdings received"
              value={importData.holdings_received}
            />
            <SummaryCard
              label="Holdings imported"
              value={importData.holdings_imported}
            />
            <SummaryCard
              label="Deleted old snapshot rows"
              value={importData.deleted_existing_holdings_for_snapshot}
            />
          </div>

          <p className="mt-4 text-sm text-green-800">
            Snapshot date:{" "}
            <span className="font-medium">{importData.snapshot_date}</span>
          </p>
          <p className="mt-1 text-sm text-green-800">
            Source upload ID:{" "}
            <span className="font-medium">{importData.source_upload_id}</span>
          </p>

          <div className="mt-4 flex flex-wrap gap-3">
            <a
              href="/portfolio"
              className="rounded-md bg-white px-4 py-2 text-sm font-medium text-green-700 ring-1 ring-green-300"
            >
              View Portfolio
            </a>
            <a
              href="/recommendations"
              className="rounded-md bg-green-700 px-4 py-2 text-sm font-medium text-white"
            >
              Generate Recommendation
            </a>
          </div>
        </section>
      )}
    </main>
  );
}

function SummaryCard({
  label,
  value,
}: {
  label: string;
  value: string | number;
}) {
  return (
    <div className="rounded-lg border bg-gray-50 p-4">
      <p className="text-sm text-gray-600">{label}</p>
      <p className="mt-1 text-2xl font-semibold text-gray-900">{value}</p>
    </div>
  );
}

function HoldingsTable({ holdings }: { holdings: ExtractedHolding[] }) {
  if (holdings.length === 0) {
    return (
      <div className="mt-6 rounded-md border border-gray-200 bg-gray-50 p-4 text-sm text-gray-600">
        No valid holdings found.
      </div>
    );
  }

  return (
    <div className="mt-6 overflow-x-auto">
      <h3 className="mb-3 text-lg font-semibold text-gray-900">
        Valid Holdings
      </h3>
      <table className="min-w-full border-collapse text-left text-sm">
        <thead>
          <tr className="border-b bg-gray-50">
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
        <tbody>
          {holdings.map((holding, index) => (
            <tr
              key={`${holding.instrument_name}-${index}`}
              className="border-b"
            >
              <td className="p-3 font-medium text-gray-900">
                {holding.instrument_name}
              </td>
              <td className="p-3">{holding.instrument_type}</td>
              <td className="p-3">{holding.symbol ?? "-"}</td>
              <td className="p-3">{holding.isin ?? "-"}</td>
              <td className="p-3 text-right">{holding.quantity}</td>
              <td className="p-3 text-right">₹{holding.average_cost}</td>
              <td className="p-3 text-right">₹{holding.invested_amount}</td>
              <td className="p-3 text-right">₹{holding.current_value}</td>
              <td className="p-3">{holding.confidence ?? "-"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function InvalidHoldingsTable({
  invalidHoldings,
}: {
  invalidHoldings: InvalidExtractedHolding[];
}) {
  return (
    <div className="mt-8 overflow-x-auto">
      <h3 className="mb-3 text-lg font-semibold text-red-900">
        Invalid Holdings
      </h3>
      <table className="min-w-full border-collapse text-left text-sm">
        <thead>
          <tr className="border-b bg-red-50">
            <th className="p-3">Row</th>
            <th className="p-3">Instrument</th>
            <th className="p-3">Errors</th>
          </tr>
        </thead>
        <tbody>
          {invalidHoldings.map((item) => (
            <tr key={item.row_number} className="border-b">
              <td className="p-3">{item.row_number}</td>
              <td className="p-3">
                {item.holding.instrument_name ?? "Unknown"}
              </td>
              <td className="p-3 text-red-700">
                <ul className="list-disc pl-5">
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
  );
}