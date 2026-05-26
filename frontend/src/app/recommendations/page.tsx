import { RecommendationPanel } from "@/components/recommendations/recommendation-panel";

export default function RecommendationsPage() {
  return (
    <main className="min-h-screen bg-slate-950 px-6 py-10 text-white">
      <section className="mx-auto max-w-6xl">
        <RecommendationPanel />
      </section>
    </main>
  );
}