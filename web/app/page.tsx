import Link from "next/link";
import { loadDashboardData } from "@/lib/data";
import { pct } from "@/lib/format";
import StatCard from "@/components/StatCard";

export default function HomePage() {
  const data = loadDashboardData();

  const jevRun = data.eval_runs.find((r) => r.method === "jev_api");
  const jevBanking = jevRun?.results.by_source.banking77?.accuracy;

  const uniqueModels = new Set(data.eval_runs.map((r) => r.model.name));
  const allBenchmarks = new Set(
    data.eval_runs.flatMap((r) => Object.keys(r.results.by_source))
  );

  // Top runs by accuracy (prefer full-coverage)
  const topRuns = [...data.eval_runs]
    .filter((r) => r.results.aggregate.accuracy != null)
    .sort((a, b) => {
      const diff =
        (b.results.aggregate.accuracy ?? 0) - (a.results.aggregate.accuracy ?? 0);
      if (Math.abs(diff) < 0.001) return b.n_benchmarks - a.n_benchmarks;
      return diff;
    })
    .slice(0, 7);

  // Best training result on Banking77 specifically
  const bestTrained = data.training_runs
    .filter((t) => t.benchmark === "banking77" && t.best_accuracy != null)
    .sort((a, b) => (b.best_accuracy ?? 0) - (a.best_accuracy ?? 0))[0];

  // Jev API benchmark-specific accuracies for key findings
  const jevArc = jevRun?.results.by_source.arc?.accuracy;
  const jevHellaswag = jevRun?.results.by_source.hellaswag?.accuracy;

  return (
    <div className="max-w-7xl mx-auto px-4 py-12">
      {/* Hero */}
      <div className="mb-12">
        <h1 className="text-4xl font-bold tracking-tight mb-3">krino</h1>
        <p className="text-lg text-text-dim max-w-2xl">
          Typed decision models that output calibrated probabilities instead of
          free text. Open replication of TypeSafe&apos;s Jev.
        </p>
        <div className="mt-4 flex gap-3">
          <Link
            href="/benchmarks"
            className="px-4 py-2 rounded-[var(--radius)] bg-accent text-[var(--accent-on)] font-medium text-sm hover:opacity-90 transition-opacity"
          >
            View Benchmarks
          </Link>
          <a
            href="https://github.com/Oaklight/krino"
            target="_blank"
            rel="noopener noreferrer"
            className="px-4 py-2 rounded-[var(--radius)] border border-border text-sm hover:bg-bg-hover transition-colors"
          >
            GitHub
          </a>
        </div>
      </div>

      {/* Key numbers */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12">
        <StatCard
          value={bestTrained ? pct(bestTrained.best_accuracy) : "—"}
          label="Best Banking77"
          detail={bestTrained ? `${bestTrained.model}, trained heads` : "—"}
        />
        <StatCard
          value={jevBanking != null ? pct(jevBanking) : "—"}
          label="Jev Reference"
          detail="Banking77 (API benchmark)"
        />
        <StatCard
          value={String(uniqueModels.size)}
          label="Models Swept"
          detail="Causal + encoder + reranker"
        />
        <StatCard
          value={String(allBenchmarks.size)}
          label="Benchmarks"
          detail="Choice + noul + score types"
        />
      </div>

      {/* Top results mini-table */}
      <div className="mb-12">
        <h2 className="text-xl font-semibold mb-4">Top Results</h2>
        <div className="overflow-x-auto border border-border rounded-[var(--radius)]">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-bg-card">
                <th className="th-material text-left px-3 py-2">Model</th>
                <th className="th-material text-left px-3 py-2">Params</th>
                <th className="th-material text-left px-3 py-2">Method</th>
                <th className="th-material text-right px-3 py-2">Benchmarks</th>
                <th className="th-material text-right px-3 py-2">Avg Accuracy</th>
              </tr>
            </thead>
            <tbody>
              {topRuns.map((run) => (
                <tr
                  key={run.run_id}
                  className={`border-b border-border hover:bg-bg-hover transition-colors ${
                    run.method === "jev_api" ? "row-jev" : ""
                  }`}
                >
                  <td className="px-3 py-2 font-medium">{run.model.name}</td>
                  <td className="px-3 py-2 text-text-dim">{run.model.params}</td>
                  <td className="px-3 py-2 text-text-dim">{run.method}</td>
                  <td className="px-3 py-2 text-right text-text-dim">
                    {run.n_benchmarks}
                  </td>
                  <td className="px-3 py-2 text-right font-mono font-medium">
                    {pct(run.results.aggregate.accuracy)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-3">
          <Link
            href="/benchmarks"
            className="text-sm text-accent hover:text-accent-hover transition-colors"
          >
            View all benchmarks →
          </Link>
        </div>
      </div>

      {/* Key findings */}
      <div className="mb-12">
        <h2 className="text-xl font-semibold mb-4">Key Findings</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-bg-card border border-border rounded-[var(--radius)] p-5">
            <h3 className="font-medium mb-2">Training &gt; Scaling</h3>
            <p className="text-sm text-text-dim">
              A 150M-parameter Ettin reranker with trained heads reaches{" "}
              {bestTrained ? pct(bestTrained.best_accuracy) : "95%+"} on Banking77
              — outperforming 7B causal models at zero-shot.
            </p>
          </div>
          <div className="bg-bg-card border border-border rounded-[var(--radius)] p-5">
            <h3 className="font-medium mb-2">Reranker Pretraining Transfers</h3>
            <p className="text-sm text-text-dim">
              Cross-encoder rerankers show strong zero-shot baselines, suggesting
              passage-relevance pretraining aligns well with probability estimation.
            </p>
          </div>
          <div className="bg-bg-card border border-border rounded-[var(--radius)] p-5">
            <h3 className="font-medium mb-2">Jev Leads on Reasoning</h3>
            <p className="text-sm text-text-dim">
              Jev API dominates ARC ({pct(jevArc)}), HellaSwag ({pct(jevHellaswag)}),
              and other reasoning benchmarks — a gap no open model has closed yet.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
