import { loadDashboardData } from "@/lib/data";
import BenchmarkTable from "@/components/BenchmarkTable";
import JevBenchTable from "@/components/JevBenchTable";
import { pct } from "@/lib/format";
import ErrorBoundary from "@/components/ErrorBoundary";

export default function BenchmarksPage() {
  const data = loadDashboardData();

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      {/* JevBench — trained heads and zero-shot readouts */}
      {data.jevbench_runs.length > 0 && (
        <div className="mb-12">
          <h1 className="text-2xl font-bold mb-2">JevBench Results</h1>
          <p className="text-sm text-text-dim mb-6">
            JevBench (231 curated items across 3 difficulty tiers), sorted by overall
            accuracy. Every row names the readout that produced it: trained heads,
            option-text continuation, per-option judgement and native reranker scores are
            separate instruments and are not directly comparable. H2O-Lightning-4B is a
            third-party trained model scored through its own serving stack; its native
            decision rows are not a zero-shot readout cell. Hover a row for the
            recorded readout and model id. Trained-head rows also carry the project
            generation: v1 heads predate the v4 line, so a v1 head on a Qwen3.5-4B
            backbone is not a v4 result.
          </p>
          <p className="text-xs text-text-muted mb-4">
            All local numbers are JevBench-public-231 scores under our own
            harness, not official JevBench scores. Official reference (v1.7.3
            snapshot): Jev 1.13.0 Intelligence 65.60 / composite 71.45.
          </p>
          <ErrorBoundary>
            <JevBenchTable runs={data.jevbench_runs} />
          </ErrorBoundary>
        </div>
      )}

      {/* Readout instruments on their own item sets */}
      {data.probe_runs.length > 0 && (
        <div className="mb-12">
          <h2 className="text-2xl font-bold mb-2">Readout instruments</h2>
          <p className="text-sm text-text-dim mb-6">
            Probes that answer a different question on a different item set from
            JevBench 231. The dataset and item count are in every row, and the
            source file is under each detail; these are not JevBench accuracies
            and are not comparable to the table above.
          </p>
          <div className="overflow-x-auto border border-border rounded-[var(--radius)]">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border bg-bg-card">
                  <th className="th-material px-3 py-2 text-left">Instrument</th>
                  <th className="th-material px-3 py-2 text-left">Dataset</th>
                  <th className="th-material px-3 py-2 text-right">n</th>
                  <th className="th-material px-3 py-2 text-right">Result</th>
                  <th className="th-material px-3 py-2 text-left">Detail</th>
                </tr>
              </thead>
              <tbody>
                {data.probe_runs.map((row) => (
                  <tr
                    key={row.run_id}
                    className="border-b border-border hover:bg-bg-hover transition-colors align-top"
                  >
                    <td className="px-3 py-2 font-medium whitespace-nowrap">
                      {row.instrument}
                    </td>
                    <td className="px-3 py-2 text-text-dim whitespace-nowrap">
                      {row.dataset}
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-text-dim">
                      {row.n}
                    </td>
                    <td className="px-3 py-2 text-right font-mono font-medium whitespace-nowrap">
                      {pct(row.value)}{" "}
                      <span className="text-text-dim">
                        ({row.correct}/{row.total})
                      </span>
                    </td>
                    <td className="px-3 py-2 text-text-dim">
                      {row.detail}
                      <span className="block font-mono text-[10px] text-text-muted break-all">
                        {row.source_file}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* 19-benchmark sweep */}
      <h2 className="text-2xl font-bold mb-2">19-Benchmark Sweep</h2>
      <p className="text-sm text-text-dim mb-6">
        Zero-shot and trained evaluation results across {Object.keys(data.benchmarks).length}{" "}
        benchmarks. Click a row to expand per-benchmark details.
      </p>
      <ErrorBoundary>
        <BenchmarkTable
          runs={data.eval_runs}
          benchmarkTypes={data.benchmarks}
          jevComparison={data.jev_comparison}
        />
      </ErrorBoundary>
    </div>
  );
}
