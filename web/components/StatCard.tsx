interface StatCardProps {
  value: string;
  label: string;
  detail: string;
}

export default function StatCard({ value, label, detail }: StatCardProps) {
  return (
    <div className="bg-bg-card border border-border rounded-[var(--radius)] p-5">
      <div className="text-3xl font-bold text-accent mb-1">{value}</div>
      <div className="text-sm font-medium text-text">{label}</div>
      <div className="text-xs text-text-muted mt-1">{detail}</div>
    </div>
  );
}
