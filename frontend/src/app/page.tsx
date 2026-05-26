import { getBackendHealth } from "@/lib/api";

export default async function Home() {
  let health;

  try {
    health = await getBackendHealth();
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
            Make sure FastAPI is running on port 8000.
          </p>
        </section>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-slate-950 px-6 py-10 text-white">
      <section className="mx-auto max-w-4xl">
        <div className="rounded-2xl border border-emerald-500/30 bg-slate-900 p-8 shadow-xl">
          <p className="text-sm uppercase tracking-wide text-emerald-300">
            Frontend foundation
          </p>

          <h1 className="mt-3 text-4xl font-bold">Stock Analysis Tool</h1>

          <p className="mt-4 max-w-2xl text-slate-300">
            Educational investment recommendation and portfolio analysis tool
            for Indian stocks, ETFs, and mutual funds.
          </p>

          <div className="mt-8 grid gap-4 md:grid-cols-3">
            <div className="rounded-xl border border-slate-700 bg-slate-800 p-5">
              <p className="text-sm text-slate-400">Backend Status</p>
              <p className="mt-2 text-2xl font-semibold text-emerald-300">
                {health.status}
              </p>
            </div>

            <div className="rounded-xl border border-slate-700 bg-slate-800 p-5">
              <p className="text-sm text-slate-400">Service</p>
              <p className="mt-2 text-lg font-semibold">{health.service}</p>
            </div>

            <div className="rounded-xl border border-slate-700 bg-slate-800 p-5">
              <p className="text-sm text-slate-400">Version</p>
              <p className="mt-2 text-lg font-semibold">{health.version}</p>
            </div>
          </div>

          <div className="mt-8 rounded-xl border border-slate-700 bg-slate-800 p-5">
            <h2 className="text-xl font-semibold">Next frontend modules</h2>
            <ul className="mt-4 list-inside list-disc space-y-2 text-slate-300">
              <li>Investor profile form</li>
              <li>Instrument management screen</li>
              <li>Portfolio holdings dashboard</li>
              <li>Recommendation and explanation view</li>
            </ul>
          </div>
        </div>
      </section>
    </main>
  );
}