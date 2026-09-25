export interface BenchmarkResult {
  n: number;
  type: "choice" | "noul" | "score" | "mixed";
  accuracy?: number;
  ece?: number;
  mae?: number;
  latency_ms?: number;
  subtypes?: Record<string, { n: number; accuracy?: number; ece?: number; mae?: number }>;
}

export interface EvalRun {
  run_id: string;
  model: {
    name: string;
    params: string;
    type: "causal" | "encoder" | "reranker" | "moe" | "api";
  };
  method: string;
  phase: string | null;
  n_benchmarks: number;
  results: {
    aggregate: {
      accuracy: number | null;
      n_benchmarks_with_accuracy: number;
    };
    by_source: Record<string, BenchmarkResult>;
  };
}

export interface TrainingEpoch {
  epoch: number;
  train_loss: number;
  eval_loss: number | null;
  eval_accuracy: number | null;
  lr: number | null;
  elapsed_s: number;
  best?: boolean;
}

export interface TrainingRun {
  run_id: string;
  model: string;
  params: string;
  benchmark: string;
  info: string;
  best_accuracy: number | null;
  best_loss: number | null;
  n_epochs: number;
  history: TrainingEpoch[];
}

export interface JevComparison {
  jev: number;
  best_model: string | null;
  best_accuracy: number | null;
}

export interface DashboardData {
  generated_at: string | null;
  eval_runs: EvalRun[];
  training_runs: TrainingRun[];
  jevbench_runs: JevBenchRun[];
  jev_comparison: Record<string, JevComparison>;
  benchmarks: Record<string, string>;
}

export interface JevBenchRun {
  run_id: string;
  model: {
    name: string;
    params: string;
    type: "causal" | "encoder" | "reranker" | "moe" | "api";
  };
  method: string;
  benchmark: string;
  accuracy: number;
  correct: number;
  total: number;
  by_type: Record<string, { accuracy: number; correct: number; total: number }>;
  by_tier: Record<string, { accuracy: number; correct: number; total: number }>;
  latency_ms: number | null;
}
