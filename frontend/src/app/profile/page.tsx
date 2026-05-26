import { ProfileForm } from "@/components/profile/profile-form";

export default function ProfilePage() {
  return (
    <main className="min-h-screen bg-slate-950 px-6 py-10 text-white">
      <section className="mx-auto max-w-4xl">
        <ProfileForm />
      </section>
    </main>
  );
}
