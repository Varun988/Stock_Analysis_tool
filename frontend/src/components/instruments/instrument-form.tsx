"use client";

import { FormEvent, useEffect, useState } from "react";

type InstrumentType = "ETF" | "MUTUAL_FUND" | "STOCK";

type InstrumentCreate = {
  name: string;
  instrument_type: InstrumentType;
  market: "INDIA";
  symbol: string | null;
  isin: string | null;
  amfi_scheme_code: string | null;
  category: string | null;
};

type InstrumentResponse = InstrumentCreate & {
  instrument_id: string;
};

const defaultInstrument: InstrumentCreate = {
  name: "",
  instrument_type: "STOCK",
  market: "INDIA",
  symbol: "",
  isin: "",
  amfi_scheme_code: "",
  category: "",
};

function normalizeOptionalText(value: string): string | null {
  const trimmedValue = value.trim();
  return trimmedValue.length > 0 ? trimmedValue : null;
}

export function InstrumentForm() {
  const [instrument, setInstrument] =
    useState<InstrumentCreate>(defaultInstrument);
  const [instruments, setInstruments] = useState<InstrumentResponse[]>([]);
  const [statusMessage, setStatusMessage] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  async function loadInstruments() {
    setIsLoading(true);

    try {
      const response = await fetch("/api/instruments", {
        cache: "no-store",
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result?.detail ?? "Failed to load instruments");
      }

      setInstruments(result?.data ?? []);
    } catch (error) {
      setStatusMessage(
        error instanceof Error
          ? error.message
          : "Something went wrong while loading instruments.",
      );
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    loadInstruments();
  }, []);

  function updateField<K extends keyof InstrumentCreate>(
    key: K,
    value: InstrumentCreate[K],
  ) {
    setInstrument((currentInstrument) => ({
      ...currentInstrument,
      [key]: value,
    }));
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsSaving(true);
    setStatusMessage("");

    const payload: InstrumentCreate = {
      name: instrument.name.trim(),
      instrument_type: instrument.instrument_type,
      market: "INDIA",
      symbol: normalizeOptionalText(instrument.symbol ?? ""),
      isin: normalizeOptionalText(instrument.isin ?? ""),
      amfi_scheme_code: normalizeOptionalText(
        instrument.amfi_scheme_code ?? "",
      ),
      category: normalizeOptionalText(instrument.category ?? ""),
    };

    try {
      const response = await fetch("/api/instruments", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result?.detail ?? "Instrument save failed");
      }

      setStatusMessage("Instrument created successfully.");
      setInstrument(defaultInstrument);
      await loadInstruments();
    } catch (error) {
      setStatusMessage(
        error instanceof Error
          ? error.message
          : "Something went wrong while saving instrument.",
      );
    } finally {
      setIsSaving(false);
    }
  }

  return (
    <div className="space-y-8">
      <form
        onSubmit={handleSubmit}
        className="rounded-2xl border border-slate-700 bg-slate-900 p-6"
      >
        <div>
          <p className="text-sm uppercase tracking-wide text-emerald-300">
            Instruments
          </p>
          <h1 className="mt-2 text-3xl font-bold">Instrument Management</h1>
          <p className="mt-3 text-slate-400">
            Add Indian stocks, ETFs, and mutual funds. Instruments are used by
            portfolio holdings, market data providers, metrics, risk, and
            recommendations.
          </p>
        </div>

        <div className="mt-8 grid gap-5 md:grid-cols-2">
          <label className="space-y-2 md:col-span-2">
            <span className="text-sm text-slate-300">Instrument Name</span>
            <input
              type="text"
              required
              value={instrument.name}
              onChange={(event) => updateField("name", event.target.value)}
              placeholder="Reliance Industries"
              className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-emerald-400"
            />
          </label>

          <label className="space-y-2">
            <span className="text-sm text-slate-300">Instrument Type</span>
            <select
              value={instrument.instrument_type}
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
            <span className="text-sm text-slate-300">Market</span>
            <input
              type="text"
              value="INDIA"
              disabled
              className="w-full rounded-lg border border-slate-700 bg-slate-800 px-4 py-3 text-slate-400"
            />
          </label>

          <label className="space-y-2">
            <span className="text-sm text-slate-300">Symbol</span>
            <input
              type="text"
              value={instrument.symbol ?? ""}
              onChange={(event) => updateField("symbol", event.target.value)}
              placeholder="RELIANCE / NIFTYBEES"
              className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-emerald-400"
            />
          </label>

          <label className="space-y-2">
            <span className="text-sm text-slate-300">ISIN</span>
            <input
              type="text"
              value={instrument.isin ?? ""}
              onChange={(event) => updateField("isin", event.target.value)}
              placeholder="INE002A01018"
              className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-emerald-400"
            />
          </label>

          <label className="space-y-2">
            <span className="text-sm text-slate-300">AMFI Scheme Code</span>
            <input
              type="text"
              value={instrument.amfi_scheme_code ?? ""}
              onChange={(event) =>
                updateField("amfi_scheme_code", event.target.value)
              }
              placeholder="119551 for MFAPI mutual fund"
              className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-emerald-400"
            />
          </label>

          <label className="space-y-2">
            <span className="text-sm text-slate-300">Category</span>
            <input
              type="text"
              value={instrument.category ?? ""}
              onChange={(event) => updateField("category", event.target.value)}
              placeholder="Large Cap Stock / Nifty 50 ETF / Debt Fund"
              className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-emerald-400"
            />
          </label>
        </div>

        <div className="mt-8 flex items-center gap-4">
          <button
            type="submit"
            disabled={isSaving}
            className="rounded-lg bg-emerald-500 px-5 py-3 font-semibold text-slate-950 hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {isSaving ? "Saving..." : "Create Instrument"}
          </button>

          {statusMessage ? (
            <p className="text-sm text-slate-300">{statusMessage}</p>
          ) : null}
        </div>
      </form>

      <section className="rounded-2xl border border-slate-700 bg-slate-900 p-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-sm uppercase tracking-wide text-sky-300">
              Saved Instruments
            </p>
            <h2 className="mt-2 text-2xl font-semibold">
              Instrument Universe
            </h2>
            <p className="mt-2 text-sm text-slate-400">
              These instruments can be linked to portfolio holdings.
            </p>
          </div>

          <button
            type="button"
            onClick={loadInstruments}
            className="rounded-lg border border-slate-600 px-4 py-2 text-sm text-slate-200 hover:border-emerald-400"
          >
            Refresh
          </button>
        </div>

        <div className="mt-6 space-y-3">
          {isLoading ? (
            <p className="text-slate-400">Loading instruments...</p>
          ) : instruments.length === 0 ? (
            <p className="text-slate-400">No instruments created yet.</p>
          ) : (
            instruments.map((item) => (
              <div
                key={item.instrument_id}
                className="rounded-xl border border-slate-700 bg-slate-800 p-5"
              >
                <div className="flex flex-wrap items-start justify-between gap-4">
                  <div>
                    <p className="text-lg font-semibold">{item.name}</p>
                    <p className="mt-1 text-sm text-slate-400">
                      ID: {item.instrument_id}
                    </p>
                  </div>

                  <span className="rounded-full border border-emerald-500/40 bg-emerald-950 px-3 py-1 text-xs text-emerald-200">
                    {item.instrument_type}
                  </span>
                </div>

                <div className="mt-4 grid gap-3 text-sm text-slate-300 md:grid-cols-2">
                  <p>Symbol: {item.symbol ?? "—"}</p>
                  <p>ISIN: {item.isin ?? "—"}</p>
                  <p>AMFI Scheme Code: {item.amfi_scheme_code ?? "—"}</p>
                  <p>Category: {item.category ?? "—"}</p>
                </div>
              </div>
            ))
          )}
        </div>
      </section>
    </div>
  );
}
