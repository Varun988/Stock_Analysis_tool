import { ExplanationHistoryPanel } from "@/components/explanations/explanation-history-panel";

export default function ExplanationHistoryPage() {
  return (
    <main className="min-h-screen bg-slate-950 px-6 py-10 text-white">
      <section className="mx-auto max-w-6xl">
        <ExplanationHistoryPanel />
      </section>
    </main>
  );
}