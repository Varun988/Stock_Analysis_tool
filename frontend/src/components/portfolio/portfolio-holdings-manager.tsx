"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

type InstrumentType = "ETF" | "MUTUAL_FUND" | "STOCK";

type Instrument = {
  instrument_id: string;
  name: string;
  instrument_type: InstrumentType;
  market: "INDIA";
  symbol: string | null;
  isin: string | null;
  amfi_scheme_code: string | null;
  category: string | null;
};

type HoldingCreate = {
  instrument_id: string | null;
  instrument_name: string;
  instrument_type: InstrumentType;
  quantity: number;
  average_cost: number;
  invested_amount: number;
  current_value: number;
};

type HoldingResponse = HoldingCreate & {
  holding_id: string;
  gain_loss: number;
  gain_loss_percent: number;
};

type PortfolioSummary = {
  total_invested: number;
  current_value: number;
  gain_loss: number;
  gain_loss_percent: number;
  number_of_holdings: number;
  allocation_by_instrument: Record<string, number>;
  allocation_by_instrument_type: Record<string, number>;
  largest_holding_name: string | null;
  largest_holding_percent: number;
  concentration_warning: string | null;
};

type AllocationChartProps = {
  title: string;
  description: string;
  allocation: Record<string, number>;
};

const defaultHolding: HoldingCreate = {
  instrument_id: null,
  instrument_name: "",
  instrument_type: "STOCK",
  quantity: 1,
  average_cost: 0,
  invested_amount: 0,
  current_value: 0,
};

function formatCurrency(value: number) {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 2,
  }).format(value);
}

function formatPercent(value: number) {
  return `${value.toFixed(2)}%`;
}

function formatAllocationLabel(value: string) {
  return value
    .replaceAll("_", " ")
    .toLowerCase()
    .replace(/\w/g, (character) => character.toUpperCase());
}

function getAllocationEntries(allocation: Record<string, number>) {
  return Object.entries(allocation)
    .sort(([, firstValue], [, secondValue]) => secondValue - firstValue)
    .map(([label, value]) => ({
      label,
      value,
    }));
}

function getAllocationBarColor(index: number) {
  const colors = [
    "bg-emerald-400",
    "bg-sky-400",
    "bg-purple-400",
    "bg-amber-400",
    "bg-rose-400",
    "bg-cyan-400",
  ];

  return colors[index % colors.length];
}

function AllocationChart({
  title,
  description,
  allocation,
}: AllocationChartProps) {
  const entries = getAllocationEntries(allocation);

  return (
    <section className="rounded-2xl border border-slate-700 bg-slate-900 p-6">
      <div>
        <p className="text-sm uppercase tracking-wide text-emerald-300">
          Allocation
        </p>
        <h2 className="mt-2 text-2xl font-semibold">{title}</h2>
        <p className="mt-2 text-sm text-slate-400">{description}</p>
      </div>

      <div className="mt-6 space-y-4">
        {entries.length === 0 ? (
          <p className="text-slate-400">No allocation data available yet.</p>
        ) : (
          entries.map((entry, index) => (
            <div key={entry.label} className="space-y-2">
              <div className="flex items-center justify-between gap-4 text-sm">
                <span className="font-medium text-slate-200">
                  {formatAllocationLabel(entry.label)}
                </span>
                <span className="text-slate-400">
                  {formatPercent(entry.value)}
                </span>
              </div>

              <div className="h-3 overflow-hidden rounded-full bg-slate-800">
                <div
                  className={`h-full rounded-full ${getAllocationBarColor(
                    index,
                  )}`}
                  style={{
                    width: `${Math.min(Math.max(entry.value, 0), 100)}%`,
                  }}
                />
              </div>
            </div>
          ))
        )}
      </div>
    </section>
  );
}

export function PortfolioHoldingsManager() {
  const [instruments, setInstruments] = useState<Instrument[]>([]);
  const [holdings, setHoldings] = useState<HoldingResponse[]>([]);
  const [summary, setSummary] = useState<PortfolioSummary | null>(null);
  const [holding, setHolding] = useState<HoldingCreate>(defaultHolding);
  const [selectedInstrumentId, setSelectedInstrumentId] = useState("");
  const [statusMessage, setStatusMessage] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const selectedInstrument = useMemo(
    () =>
      instruments.find(
        (instrument) => instrument.instrument_id === selectedInstrumentId,
      ),
    [instruments, selectedInstrumentId],
  );

  async function loadInstruments() {
    const response = await fetch("/api/instruments", {
      cache: "no-store",
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result?.detail ?? "Failed to load instruments");
    }

    setInstruments(result?.data ?? []);
  }

  async function loadHoldings() {
    const response = await fetch("/api/portfolio/holdings", {
      cache: "no-store",
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result?.detail ?? "Failed to load holdings");
    }

    setHoldings(result?.data ?? []);
  }

  async function loadSummary() {
    const response = await fetch("/api/portfolio/summary", {
      cache: "no-store",
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result?.detail ?? "Failed to load portfolio summary");
    }

    setSummary(result?.data ?? null);
  }

  async function loadPageData() {
    setIsLoading(true);

    try {
      await Promise.all([loadInstruments(), loadHoldings(), loadSummary()]);
    } catch (error) {
      setStatusMessage(
        error instanceof Error
          ? error.message
          : "Something went wrong while loading portfolio data.",
      );
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    loadPageData();
  }, []);

  useEffect(() => {
    if (!selectedInstrument) {
      return;
    }

    setHolding((currentHolding) => ({
      ...currentHolding,
      instrument_id: selectedInstrument.instrument_id,
      instrument_name: selectedInstrument.symbol || selectedInstrument.name,
      instrument_type: selectedInstrument.instrument_type,
    }));
  }, [selectedInstrument]);

  function updateField<K extends keyof HoldingCreate>(
    key: K,
    value: HoldingCreate[K],
  ) {
    setHolding((currentHolding) => ({
      ...currentHolding,
      [key]: value,
    }));
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsSaving(true);
    setStatusMessage("");

    try {
      const payload: HoldingCreate = {
        instrument_id: holding.instrument_id,
        instrument_name: holding.instrument_name.trim(),
        instrument_type: holding.instrument_type,
        quantity: Number(holding.quantity),
        average_cost: Number(holding.average_cost),
        invested_amount: Number(holding.invested_amount),
        current_value: Number(holding.current_value),
      };

      const response = await fetch("/api/portfolio/holdings", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result?.detail ?? "Portfolio holding save failed");
      }

      setStatusMessage("Portfolio holding created successfully.");
      setHolding(defaultHolding);
      setSelectedInstrumentId("");
      await Promise.all([loadHoldings(), loadSummary()]);
    } catch (error) {
      setStatusMessage(
        error instanceof Error
          ? error.message
          : "Something went wrong while saving holding.",
      );
    } finally {
      setIsSaving(false);
    }
  }

  return (
    <div className="space-y-8">
      <section className="rounded-2xl border border-slate-700 bg-slate-900 p-6">
        <div>
          <p className="text-sm uppercase tracking-wide text-emerald-300">
            Portfolio
          </p>
          <h1 className="mt-2 text-3xl font-bold">Portfolio Holdings</h1>
          <p className="mt-3 text-slate-400">
            Add holdings linked to saved instruments. Portfolio summary and
            recommendations depend on this data.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="mt-8 grid gap-5 md:grid-cols-2">
          <label className="space-y-2 md:col-span-2">
            <span className="text-sm text-slate-300">Linked Instrument</span>
            <select
              value={selectedInstrumentId}
              onChange={(event) => setSelectedInstrumentId(event.target.value)}
              className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-emerald-400"
            >
              <option value="">Select an instrument</option>
              {instruments.map((instrument) => (
                <option
                  key={instrument.instrument_id}
                  value={instrument.instrument_id}
                >
                  {instrument.name} ({instrument.instrument_type})
                </option>
              ))}
            </select>
          </label>

          <label className="space-y-2">
            <span className="text-sm text-slate-300">Instrument Name</span>
            <input
              type="text"
              required
              value={holding.instrument_name}
              onChange={(event) =>
                updateField("instrument_name", event.target.value)
              }
              placeholder="RELIANCE / NIFTYBEES"
              className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-emerald-400"
            />
          </label>

          <label className="space-y-2">
            <span className="text-sm text-slate-300">Instrument Type</span>
            <select
              value={holding.instrument_type}
              onChange={(event) =>
                updateField(
                  "instrument_type",
                  event.target.value as InstrumentType,
                )
              }
              className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-emerald-400"
            >
              <option value="STOCK">Stock</option>
              <option value="ETF">ETF</option>
              <option value="MUTUAL_FUND">Mutual Fund</option>
            </select>
          </label>

          <label className="space-y-2">
            <span className="text-sm text-slate-300">Quantity</span>
            <input
              type="number"
              min={0.0001}
              step="any"
              value={holding.quantity}
              onChange={(event) =>
                updateField("quantity", Number(event.target.value))
              }
              className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-emerald-400"
            />
          </label>

          <label className="space-y-2">
            <span className="text-sm text-slate-300">Average Cost</span>
            <input
              type="number"
              min={0.01}
              step="any"
              value={holding.average_cost}
              onChange={(event) =>
                updateField("average_cost", Number(event.target.value))
              }
              className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-emerald-400"
            />
          </label>

          <label className="space-y-2">
            <span className="text-sm text-slate-300">Invested Amount</span>
            <input
              type="number"
              min={0.01}
              step="any"
              value={holding.invested_amount}
              onChange={(event) =>
                updateField("invested_amount", Number(event.target.value))
              }
              className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-emerald-400"
            />
          </label>

          <label className="space-y-2">
            <span className="text-sm text-slate-300">Current Value</span>
            <input
              type="number"
              min={0}
              step="any"
              value={holding.current_value}
              onChange={(event) =>
                updateField("current_value", Number(event.target.value))
              }
              className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-emerald-400"
            />
          </label>

          <div className="flex items-center gap-4 md:col-span-2">
            <button
              type="submit"
              disabled={isSaving}
              className="rounded-lg bg-emerald-500 px-5 py-3 font-semibold text-slate-950 hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {isSaving ? "Saving..." : "Add Holding"}
            </button>

            {statusMessage ? (
              <p className="text-sm text-slate-300">{statusMessage}</p>
            ) : null}
          </div>
        </form>
      </section>

      <section className="grid gap-4 md:grid-cols-3">
        <div className="rounded-2xl border border-slate-700 bg-slate-900 p-5">
          <p className="text-sm text-slate-400">Total Invested</p>
          <p className="mt-2 text-2xl font-semibold">
            {summary ? formatCurrency(summary.total_invested) : "—"}
          </p>
        </div>

        <div className="rounded-2xl border border-slate-700 bg-slate-900 p-5">
          <p className="text-sm text-slate-400">Current Value</p>
          <p className="mt-2 text-2xl font-semibold">
            {summary ? formatCurrency(summary.current_value) : "—"}
          </p>
        </div>

        <div className="rounded-2xl border border-slate-700 bg-slate-900 p-5">
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
      </section>

      {summary ? (
        <section className="grid gap-5 lg:grid-cols-2">
          <AllocationChart
            title="Allocation by Instrument Type"
            description="Shows how your current portfolio value is split across stocks, ETFs, and mutual funds."
            allocation={summary.allocation_by_instrument_type}
          />

          <AllocationChart
            title="Allocation by Instrument"
            description="Shows how much each individual holding contributes to your current portfolio value."
            allocation={summary.allocation_by_instrument}
          />
        </section>
      ) : null}

      {summary?.concentration_warning ? (
        <section className="rounded-2xl border border-amber-500/40 bg-amber-950/30 p-5 text-amber-100">
          <p className="font-semibold">Concentration Warning</p>
          <p className="mt-2 text-sm">{summary.concentration_warning}</p>
        </section>
      ) : null}

      <section className="rounded-2xl border border-slate-700 bg-slate-900 p-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-sm uppercase tracking-wide text-sky-300">
              Holdings
            </p>
            <h2 className="mt-2 text-2xl font-semibold">Saved Holdings</h2>
          </div>

          <button
            type="button"
            onClick={loadPageData}
            className="rounded-lg border border-slate-600 px-4 py-2 text-sm text-slate-200 hover:border-emerald-400"
          >
            Refresh
          </button>
        </div>

        <div className="mt-6 space-y-3">
          {isLoading ? (
            <p className="text-slate-400">Loading portfolio...</p>
          ) : holdings.length === 0 ? (
            <p className="text-slate-400">No holdings added yet.</p>
          ) : (
            holdings.map((item) => (
              <div
                key={item.holding_id}
                className="rounded-xl border border-slate-700 bg-slate-800 p-5"
              >
                <div className="flex flex-wrap items-start justify-between gap-4">
                  <div>
                    <p className="text-lg font-semibold">
                      {item.instrument_name}
                    </p>
                    <p className="mt-1 text-sm text-slate-400">
                      Holding ID: {item.holding_id}
                    </p>
                  </div>

                  <span className="rounded-full border border-emerald-500/40 bg-emerald-950 px-3 py-1 text-xs text-emerald-200">
                    {item.instrument_type}
                  </span>
                </div>

                <div className="mt-4 grid gap-3 text-sm text-slate-300 md:grid-cols-3">
                  <p>Quantity: {item.quantity}</p>
                  <p>Avg Cost: {formatCurrency(item.average_cost)}</p>
                  <p>Invested: {formatCurrency(item.invested_amount)}</p>
                  <p>Current: {formatCurrency(item.current_value)}</p>
                  <p>Gain/Loss: {formatCurrency(item.gain_loss)}</p>
                  <p>Gain/Loss %: {formatPercent(item.gain_loss_percent)}</p>
                </div>
              </div>
            ))
          )}
        </div>
      </section>
    </div>
  );
}
