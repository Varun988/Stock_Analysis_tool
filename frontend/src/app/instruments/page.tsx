import { InstrumentForm } from "@/components/instruments/instrument-form";

export default function InstrumentsPage() {
  return (
    <main className="min-h-screen bg-slate-950 px-6 py-10 text-white">
      <section className="mx-auto max-w-5xl">
        <InstrumentForm />
      </section>
    </main>
  );
}