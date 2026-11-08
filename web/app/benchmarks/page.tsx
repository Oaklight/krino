import { loadDashboardData } from "@/lib/data";
import BenchmarkTable from "@/components/BenchmarkTable";
import JevBenchTable from "@/components/JevBenchTable";
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
            separate instruments and are not directly comparable. Hover a row for the
            recorded readout and model id.
          </p>
          <ErrorBoundary>
            <JevBenchTable runs={data.jevbench_runs} />
          </ErrorBoundary>
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
