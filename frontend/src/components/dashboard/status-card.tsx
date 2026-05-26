type StatusCardProps = {
  title: string;
  value: string;
  description?: string;
  tone?: "success" | "warning" | "neutral";
};

const toneClasses = {
  success: "border-emerald-500/30 bg-emerald-950/30 text-emerald-200",
  warning: "border-amber-500/30 bg-amber-950/30 text-amber-200",
  neutral: "border-slate-700 bg-slate-800 text-slate-200",
};

export function StatusCard({
  title,
  value,
  description,
  tone = "neutral",
}: StatusCardProps) {
  return (
    <div className={`rounded-xl border p-5 ${toneClasses[tone]}`}>
      <p className="text-sm opacity-80">{title}</p>
      <p className="mt-2 text-2xl font-semibold">{value}</p>
      {description ? (
        <p className="mt-3 text-sm leading-6 opacity-80">{description}</p>
      ) : null}
    </div>
  );
}