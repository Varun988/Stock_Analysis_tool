import { PortfolioHoldingsManager } from "@/components/portfolio/portfolio-holdings-manager";

export default function PortfolioPage() {
  return (
    <main className="min-h-screen bg-slate-950 px-6 py-10 text-white">
      <section className="mx-auto max-w-6xl">
        <PortfolioHoldingsManager />
      </section>
    </main>
  );
}