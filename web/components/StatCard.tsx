interface StatCardProps {
  value: string;
  label: string;
  detail: string;
  source?: string;
}

export default function StatCard({ value, label, detail, source }: StatCardProps) {
  return (
    <div className="bg-bg-card border border-border rounded-[var(--radius)] p-5">
      <div className="text-3xl font-bold text-accent mb-1">{value}</div>
      <div className="text-sm font-medium text-text">{label}</div>
      <div className="text-xs text-text-muted mt-1">{detail}</div>
      {source && (
        <div className="mt-3 font-mono text-[10px] leading-snug text-text-dim break-all" title={source}>
          {source}
        </div>
      )}
    </div>
  );
}
