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
  source_file: string;
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
  source_file: string;
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

export interface LandingCheckpoint {
  model_id: string;
  name: string;
  pp: number;
}

export interface Landing {
  judgement_vs_continuation: {
    measured: number;
    wins: number;
    losses: { model_id: string; name: string; gap_pp: number }[];
    above_10_count: number;
    above_10_min_pp: number | null;
    above_10_max_pp: number | null;
    largest_tier: string;
    source_files: string[];
  };
  panel2_vs_judgement: {
    type_delta: Record<string, { items: number; total: number; pp: number }>;
    measured: number;
    wins: number;
    min_pp: number | null;
    max_pp: number | null;
    pairs: {
      model_id: string;
      name: string;
      judgement_pp: number;
      panel2_pp: number;
      gap_pp: number;
    }[];
    source_files: string[];
  };
  post_trained_4b: {
    base_pp: number | null;
    min_pp: number | null;
    max_pp: number | null;
    checkpoints: LandingCheckpoint[];
    source_files: string[];
  };
  resolution: {
    jevbench_pp: number;
    jevbench_pair: string[];
    dev_min_pp: number;
    dev_max_pp: number;
    dev_card_min_pp: number;
    dev_card_max_pp: number;
    dev_pair: string[];
    dev_pair_diff_pp: number;
    dev_pair_p: number;
    jevbench_pair_item_diff: number | null;
    top_four: LandingCheckpoint[];
    top_four_spread_pp: number;
    source_files: string[];
  };
  instrumented_rows: number;
  banking77: {
    trained_pp: number | null;
    trained_model: string | null;
    trained_source_file: string | null;
    jev_pp: number | null;
    jev_source_file: string | null;
  };
  source_files: string[];
}

export interface DashboardData {
  generated_at: string | null;
  eval_runs: EvalRun[];
  training_runs: TrainingRun[];
  jevbench_runs: JevBenchRun[];
  jev_comparison: Record<string, JevComparison>;
  landing: Landing;
  benchmarks: Record<string, string>;
}

export interface JevBenchRun {
  run_id: string;
  model: {
    name: string;
    params: string;
    type: "causal" | "encoder" | "reranker" | "moe" | "api" | "unknown";
  };
  model_id: string | null;
  head_version: string | null;
  backbone_model_id: string | null;
  code_commit: string | null;
  method: string;
  readout: string;
  readout_detail: string | null;
  source_file: string;
  benchmark: string;
  n: number;
  accuracy: number;
  correct: number;
  total: number;
  by_type: Record<string, { accuracy: number; correct: number; total: number }>;
  by_tier: Record<string, { accuracy: number; correct: number; total: number }>;
  latency_ms: number | null;
}
