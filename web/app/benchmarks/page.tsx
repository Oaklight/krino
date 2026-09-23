import { loadDashboardData } from "@/lib/data";
import BenchmarkTable from "@/components/BenchmarkTable";
import ErrorBoundary from "@/components/ErrorBoundary";

export default function BenchmarksPage() {
  const data = loadDashboardData();

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold mb-2">Benchmark Results</h1>
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
