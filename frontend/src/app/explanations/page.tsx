import { ExplanationPanel } from "@/components/explanations/explanation-panel";

export default function ExplanationsPage() {
  return (
    <main className="min-h-screen bg-slate-950 px-6 py-10 text-white">
      <section className="mx-auto max-w-6xl">
        <ExplanationPanel />
      </section>
    </main>
  );
}