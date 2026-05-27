
import { AIProviderStatusCard } from "@/components/dashboard/ai-provider-status-card";
import { ProviderHealthList } from "@/components/dashboard/provider-health-list";
import { StatusCard } from "@/components/dashboard/status-card";
import { getAIProviderStatus, getBackendHealth, getProviderHealth } from "@/lib/api";
import { DashboardQuickStats } from "@/components/dashboard/dashboard-quick-stats";
import Link from "next/link";
export default async function Home() {
  try {
    const [health, providerHealth, aiStatus] = await Promise.all([
      getBackendHealth(),
      getProviderHealth(),
      getAIProviderStatus(),
    ]);

    return (
      <main className="min-h-screen bg-slate-950 px-6 py-10 text-white">
        <section className="mx-auto max-w-6xl space-y-8">
          <div className="rounded-2xl border border-emerald-500/30 bg-slate-900 p-8 shadow-xl">
            <p className="text-sm uppercase tracking-wide text-emerald-300">
              Full-stack dashboard
            </p>

            <h1 className="mt-3 text-4xl font-bold">Stock Analysis Tool</h1>

            <p className="mt-4 max-w-3xl text-slate-300">
              Educational investment recommendation and portfolio analysis tool
              for Indian stocks, ETFs, and mutual funds.
            </p>

            <div className="mt-6 flex flex-wrap gap-3">
              <Link
                href="/profile"
                className="inline-flex rounded-lg bg-emerald-500 px-5 py-3 font-semibold text-slate-950 hover:bg-emerald-400"
              >
                Set up investor profile
              </Link>

              <Link
                href="/portfolio"
                className="inline-flex rounded-lg border border-slate-600 px-5 py-3 font-semibold text-slate-100 hover:border-emerald-400"
              >
                Manage portfolio
              </Link>
              <Link
                href="/upload"
                className="inline-flex rounded-lg border border-emerald-500/60 px-5 py-3 font-semibold text-emerald-200 hover:border-emerald-300 hover:text-emerald-100"
              >
                Upload statement
              </Link>


              <Link
                href="/recommendations"
                className="inline-flex rounded-lg border border-slate-600 px-5 py-3 font-semibold text-slate-100 hover:border-emerald-400"
              >
                Generate recommendation
              </Link>
              <Link
                href="/explanations"
                className="inline-flex rounded-lg border border-slate-600 px-5 py-3 font-semibold text-slate-100 hover:border-emerald-400"
              >
                Explain recommendation
              </Link>

              <Link
                href="/instruments"
                className="inline-flex rounded-lg border border-slate-600 px-5 py-3 font-semibold text-slate-100 hover:border-emerald-400"
              >
                Manage instruments
              </Link>
            </div>

            <div className="mt-8 grid gap-4 md:grid-cols-3">
              <StatusCard
                title="Backend Status"
                value={health.status}
                description="FastAPI backend health check"
                tone="success"
              />

              <StatusCard
                title="Service"
                value={health.service}
                description="Connected backend service"
              />

              <StatusCard
                title="Version"
                value={health.version}
                description="Backend API version"
              />
            </div>
          </div>

          <ProviderHealthList providerHealth={providerHealth} />
          <AIProviderStatusCard aiStatus={aiStatus} />
          <DashboardQuickStats />
          <section className="grid gap-4 md:grid-cols-2">
            <div className="rounded-2xl border border-slate-700 bg-slate-900 p-6">
              <p className="text-sm uppercase tracking-wide text-slate-400">
                Portfolio
              </p>
              <h2 className="mt-2 text-2xl font-semibold">
                Portfolio Summary
              </h2>
              <p className="mt-3 text-slate-400">
                Portfolio summary UI will be connected in a later step.
              </p>
            </div>

            <div className="rounded-2xl border border-slate-700 bg-slate-900 p-6">
              <p className="text-sm uppercase tracking-wide text-slate-400">
                Recommendation
              </p>
              <h2 className="mt-2 text-2xl font-semibold">
                Latest Recommendation
              </h2>
              <p className="mt-3 text-slate-400">
                Recommendation and explanation UI will be connected in a later
                step.
              </p>
            </div>
          </section>
        </section>
      </main>
    );
  } catch (error) {
    return (
      <main className="min-h-screen bg-slate-950 px-6 py-10 text-white">
        <section className="mx-auto max-w-4xl rounded-2xl border border-red-500/40 bg-red-950/40 p-8">
          <p className="text-sm uppercase tracking-wide text-red-300">
            Backend connection failed
          </p>
          <h1 className="mt-3 text-3xl font-bold">Stock Analysis Tool</h1>
          <p className="mt-4 text-red-100">
            The frontend is running, but it could not connect to the backend.
          </p>
          <p className="mt-4 rounded-lg bg-black/30 p-4 text-sm text-red-100">
            Make sure FastAPI is running on port 8000 and that the provider
            health endpoint is available.
          </p>
        </section>
      </main>
    );
  }
}