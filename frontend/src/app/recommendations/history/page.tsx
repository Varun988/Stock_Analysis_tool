import { RecommendationHistoryPanel } from "@/components/recommendations/recommendation-history-panel";

export default function RecommendationHistoryPage() {
  return (
    <main className="min-h-screen bg-slate-950 px-6 py-10 text-white">
      <section className="mx-auto max-w-6xl">
        <RecommendationHistoryPanel />
      </section>
    </main>
  );
}