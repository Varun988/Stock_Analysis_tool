"use client";

import { FormEvent, useEffect, useState } from "react";

type InvestorProfile = {
  monthly_investment_amount: number;
  risk_appetite: "low" | "moderate" | "high";
  investment_goal: string;
  time_horizon_years: number;
  experience_level: "beginner" | "intermediate" | "advanced";
  preferred_instruments: Array<"ETF" | "MUTUAL_FUND" | "STOCK">;
  preferred_market: "INDIA";
};

const defaultProfile: InvestorProfile = {
  monthly_investment_amount: 2000,
  risk_appetite: "moderate",
  investment_goal: "long_term_wealth_creation",
  time_horizon_years: 10,
  experience_level: "beginner",
  preferred_instruments: ["ETF", "MUTUAL_FUND"],
  preferred_market: "INDIA",
};

export function ProfileForm() {
  const [profile, setProfile] = useState<InvestorProfile>(defaultProfile);
  const [statusMessage, setStatusMessage] = useState<string>("");
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    async function loadProfile() {
      try {
        const response = await fetch("/api/profile", {
          cache: "no-store",
        });

        if (!response.ok) {
          return;
        }

        const result = await response.json();

        if (result?.data) {
          setProfile({
            monthly_investment_amount: result.data.monthly_investment_amount,
            risk_appetite: result.data.risk_appetite,
            investment_goal: result.data.investment_goal,
            time_horizon_years: result.data.time_horizon_years,
            experience_level: result.data.experience_level,
            preferred_instruments: result.data.preferred_instruments,
            preferred_market: result.data.preferred_market,
          });

          setStatusMessage("Existing profile loaded.");
        }
      } catch {
        setStatusMessage("No existing profile found yet.");
      }
    }

    loadProfile();
  }, []);

  function updateField<K extends keyof InvestorProfile>(
    key: K,
    value: InvestorProfile[K],
  ) {
    setProfile((currentProfile) => ({
      ...currentProfile,
      [key]: value,
    }));
  }

  function toggleInstrument(instrument: "ETF" | "MUTUAL_FUND" | "STOCK") {
    setProfile((currentProfile) => {
      const exists = currentProfile.preferred_instruments.includes(instrument);

      return {
        ...currentProfile,
        preferred_instruments: exists
          ? currentProfile.preferred_instruments.filter(
              (item) => item !== instrument,
            )
          : [...currentProfile.preferred_instruments, instrument],
      };
    });
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsSaving(true);
    setStatusMessage("");

    try {
      const response = await fetch("/api/profile", {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(profile),
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result?.detail ?? "Profile save failed");
      }

      setStatusMessage("Investor profile saved successfully.");
    } catch (error) {
      setStatusMessage(
        error instanceof Error
          ? error.message
          : "Something went wrong while saving profile.",
      );
    } finally {
      setIsSaving(false);
    }
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="rounded-2xl border border-slate-700 bg-slate-900 p-6"
    >
      <div>
        <p className="text-sm uppercase tracking-wide text-emerald-300">
          Investor Profile
        </p>
        <h1 className="mt-2 text-3xl font-bold">Profile Setup</h1>
        <p className="mt-3 text-slate-400">
          This profile helps the recommendation engine understand your monthly
          investment amount, risk appetite, time horizon, and preferred
          instruments.
        </p>
      </div>

      <div className="mt-8 grid gap-5 md:grid-cols-2">
        <label className="space-y-2">
          <span className="text-sm text-slate-300">
            Monthly Investment Amount
          </span>
          <input
            type="number"
            min={1}
            value={profile.monthly_investment_amount}
            onChange={(event) =>
              updateField(
                "monthly_investment_amount",
                Number(event.target.value),
              )
            }
            className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-emerald-400"
          />
        </label>

        <label className="space-y-2">
          <span className="text-sm text-slate-300">Time Horizon Years</span>
          <input
            type="number"
            min={1}
            value={profile.time_horizon_years}
            onChange={(event) =>
              updateField("time_horizon_years", Number(event.target.value))
            }
            className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-emerald-400"
          />
        </label>

        <label className="space-y-2">
          <span className="text-sm text-slate-300">Risk Appetite</span>
          <select
            value={profile.risk_appetite}
            onChange={(event) =>
              updateField(
                "risk_appetite",
                event.target.value as InvestorProfile["risk_appetite"],
              )
            }
            className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-emerald-400"
          >
            <option value="low">Low</option>
            <option value="moderate">Moderate</option>
            <option value="high">High</option>
          </select>
        </label>

        <label className="space-y-2">
          <span className="text-sm text-slate-300">Experience Level</span>
          <select
            value={profile.experience_level}
            onChange={(event) =>
              updateField(
                "experience_level",
                event.target.value as InvestorProfile["experience_level"],
              )
            }
            className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-emerald-400"
          >
            <option value="beginner">Beginner</option>
            <option value="intermediate">Intermediate</option>
            <option value="advanced">Advanced</option>
          </select>
        </label>

        <label className="space-y-2 md:col-span-2">
          <span className="text-sm text-slate-300">Investment Goal</span>
          <input
            type="text"
            value={profile.investment_goal}
            onChange={(event) =>
              updateField("investment_goal", event.target.value)
            }
            className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-emerald-400"
          />
        </label>
      </div>

      <div className="mt-6">
        <p className="text-sm text-slate-300">Preferred Instruments</p>

        <div className="mt-3 flex flex-wrap gap-3">
          {(["ETF", "MUTUAL_FUND", "STOCK"] as const).map((instrument) => (
            <button
              key={instrument}
              type="button"
              onClick={() => toggleInstrument(instrument)}
              className={`rounded-full border px-4 py-2 text-sm ${
                profile.preferred_instruments.includes(instrument)
                  ? "border-emerald-400 bg-emerald-950 text-emerald-200"
                  : "border-slate-700 bg-slate-950 text-slate-300"
              }`}
            >
              {instrument}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-8 flex items-center gap-4">
        <button
          type="submit"
          disabled={isSaving}
          className="rounded-lg bg-emerald-500 px-5 py-3 font-semibold text-slate-950 hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {isSaving ? "Saving..." : "Save Profile"}
        </button>

        {statusMessage ? (
          <p className="text-sm text-slate-300">{statusMessage}</p>
        ) : null}
      </div>
    </form>
  );
}
