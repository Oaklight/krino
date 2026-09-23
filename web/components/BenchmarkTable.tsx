"use client";

import React, { useMemo, useState } from "react";
import type { EvalRun, JevComparison } from "@/lib/types";
import { pct, fmtMae } from "@/lib/format";

type SortKey = "model" | "params" | "method" | "n_benchmarks" | "accuracy" | string;
type SortDir = "asc" | "desc";

const MODEL_TYPE_COLORS: Record<string, string> = {
  causal: "bg-blue text-white",
  encoder: "bg-green text-white",
  reranker: "bg-[var(--accent)] text-[var(--accent-on)]",
  moe: "bg-[var(--purple,#7c3aed)] text-white",
  api: "bg-[var(--orange,#d97706)] text-white",
};

const DEFAULT_VISIBLE_BENCHMARKS = ["banking77", "sst2", "agnews", "arc", "race"];

interface Props {
  runs: EvalRun[];
  benchmarkTypes: Record<string, string>;
  jevComparison: Record<string, JevComparison>;
}

export default function BenchmarkTable({ runs, benchmarkTypes, jevComparison }: Props) {
  const [sortKey, setSortKey] = useState<SortKey>("accuracy");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [typeFilter, setTypeFilter] = useState<Set<string>>(new Set());
  const [searchQuery, setSearchQuery] = useState("");
  const [fullCoverageOnly, setFullCoverageOnly] = useState(false);
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());
  const [visibleBenchmarks, setVisibleBenchmarks] = useState<Set<string>>(
    new Set(DEFAULT_VISIBLE_BENCHMARKS)
  );

  const allBenchmarks = useMemo(() => Object.keys(benchmarkTypes).sort(), [benchmarkTypes]);

  const maxBenchmarks = useMemo(
    () => Math.max(...runs.map((r) => r.n_benchmarks)),
    [runs]
  );

  const filtered = useMemo(() => {
    return runs.filter((run) => {
      if (typeFilter.size > 0 && !typeFilter.has(run.model.type)) return false;
      if (fullCoverageOnly && run.n_benchmarks < maxBenchmarks) return false;
      if (searchQuery) {
        const q = searchQuery.toLowerCase();
        if (
          !run.model.name.toLowerCase().includes(q) &&
          !run.method.toLowerCase().includes(q)
        )
          return false;
      }
      return true;
    });
  }, [runs, typeFilter, fullCoverageOnly, searchQuery, maxBenchmarks]);

  const sorted = useMemo(() => {
    return [...filtered].sort((a, b) => {
      let cmp = 0;
      switch (sortKey) {
        case "model":
          cmp = a.model.name.localeCompare(b.model.name);
          break;
        case "params":
          cmp = parseParams(a.model.params) - parseParams(b.model.params);
          break;
        case "method":
          cmp = a.method.localeCompare(b.method);
          break;
        case "n_benchmarks":
          cmp = a.n_benchmarks - b.n_benchmarks;
          break;
        case "accuracy":
          cmp = (a.results.aggregate.accuracy ?? -1) - (b.results.aggregate.accuracy ?? -1);
          break;
        default: {
          const bmType = benchmarkTypes[sortKey];
          const aVal = getBenchmarkValue(a, sortKey, bmType);
          const bVal = getBenchmarkValue(b, sortKey, bmType);
          cmp = aVal - bVal;
          break;
        }
      }
      return sortDir === "asc" ? cmp : -cmp;
    });
  }, [filtered, sortKey, sortDir, benchmarkTypes]);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      const isScoreCol = key in benchmarkTypes && benchmarkTypes[key] === "score";
      setSortDir(key === "model" || key === "method" || isScoreCol ? "asc" : "desc");
    }
  };

  const toggleType = (t: string) => {
    setTypeFilter((prev) => {
      const next = new Set(prev);
      if (next.has(t)) next.delete(t);
      else next.add(t);
      return next;
    });
  };

  const toggleBenchmark = (bm: string) => {
    setVisibleBenchmarks((prev) => {
      const next = new Set(prev);
      if (next.has(bm)) next.delete(bm);
      else next.add(bm);
      return next;
    });
  };

  const toggleExpand = (runId: string) => {
    setExpandedRows((prev) => {
      const next = new Set(prev);
      if (next.has(runId)) next.delete(runId);
      else next.add(runId);
      return next;
    });
  };

  const sortArrow = (key: SortKey) => {
    if (sortKey !== key) return "";
    return sortDir === "asc" ? " ▲" : " ▼";
  };

  return (
    <div>
      {/* Filters */}
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <input
          type="text"
          placeholder="Search models..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="px-3 py-1.5 text-sm border border-border rounded-[var(--radius)] bg-bg-card text-text placeholder:text-text-muted focus:outline-none focus:ring-1 focus:ring-accent"
        />
        {["causal", "encoder", "reranker", "moe", "api"].map((t) => (
          <button
            key={t}
            onClick={() => toggleType(t)}
            className={`px-3 py-1 text-xs rounded-full border transition-colors ${
              typeFilter.size === 0 || typeFilter.has(t)
                ? MODEL_TYPE_COLORS[t]
                : "border-border text-text-muted bg-transparent"
            }`}
          >
            {t}
          </button>
        ))}
        <label className="flex items-center gap-1.5 text-xs text-text-dim cursor-pointer ml-2">
          <input
            type="checkbox"
            checked={fullCoverageOnly}
            onChange={(e) => setFullCoverageOnly(e.target.checked)}
            className="accent-accent"
          />
          {maxBenchmarks}-benchmark runs only
        </label>
      </div>

      {/* Benchmark column selector */}
      <div className="mb-3 flex flex-wrap gap-1.5">
        <span className="text-xs text-text-muted py-1">Columns:</span>
        {allBenchmarks.map((bm) => (
          <button
            key={bm}
            onClick={() => toggleBenchmark(bm)}
            className={`px-2 py-0.5 text-xs rounded-[var(--radius)] border transition-colors ${
              visibleBenchmarks.has(bm)
                ? "bg-accent-subtle border-accent text-text"
                : "border-border text-text-muted hover:text-text"
            }`}
          >
            {bm}
          </button>
        ))}
      </div>

      {/* Table */}
      <div className="overflow-x-auto border border-border rounded-[var(--radius)]">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border bg-bg-card">
              <Th onClick={() => toggleSort("model")}>Model{sortArrow("model")}</Th>
              <Th onClick={() => toggleSort("params")}>Params{sortArrow("params")}</Th>
              <Th onClick={() => toggleSort("method")}>Method{sortArrow("method")}</Th>
              <Th onClick={() => toggleSort("n_benchmarks")} align="right">
                BMs{sortArrow("n_benchmarks")}
              </Th>
              <Th onClick={() => toggleSort("accuracy")} align="right">
                Avg Acc{sortArrow("accuracy")}
              </Th>
              {allBenchmarks
                .filter((bm) => visibleBenchmarks.has(bm))
                .map((bm) => (
                  <Th key={bm} onClick={() => toggleSort(bm)} align="right">
                    {bm}
                    {sortArrow(bm)}
                  </Th>
                ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((run) => {
              const isJev = run.method === "jev_api";
              const isExpanded = expandedRows.has(run.run_id);
              return (
                <React.Fragment key={run.run_id}>
                  <tr
                    className={`border-b border-border hover:bg-bg-hover cursor-pointer transition-colors ${
                      isJev ? "row-jev" : ""
                    }`}
                    tabIndex={0}
                    role="button"
                    aria-expanded={isExpanded}
                    onClick={() => toggleExpand(run.run_id)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        toggleExpand(run.run_id);
                      }
                    }}
                  >
                    <td className="px-3 py-2 font-medium whitespace-nowrap">
                      <span
                        className={`inline-block px-1.5 py-0.5 text-[10px] rounded-full mr-2 ${
                          MODEL_TYPE_COLORS[run.model.type] ?? ""
                        }`}
                      >
                        {run.model.type}
                      </span>
                      {run.model.name}
                      {run.phase && (
                        <span className="ml-1 text-[10px] text-text-muted">
                          (Phase {run.phase})
                        </span>
                      )}
                    </td>
                    <td className="px-3 py-2 text-text-dim">{run.model.params}</td>
                    <td className="px-3 py-2 text-text-dim">{run.method}</td>
                    <td className="px-3 py-2 text-right text-text-dim">{run.n_benchmarks}</td>
                    <td className="px-3 py-2 text-right font-mono font-medium">
                      {pct(run.results.aggregate.accuracy)}
                    </td>
                    {allBenchmarks
                      .filter((bm) => visibleBenchmarks.has(bm))
                      .map((bm) => {
                        const bmData = run.results.by_source[bm];
                        const bmType = benchmarkTypes[bm];
                        return (
                          <td key={bm} className="px-3 py-2 text-right font-mono text-text-dim">
                            {bmData
                              ? bmType === "score"
                                ? fmtMae(bmData.mae)
                                : bmData.type === "mixed"
                                  ? pct(getMixedAccuracy(bmData))
                                  : pct(bmData.accuracy)
                              : "—"}
                          </td>
                        );
                      })}
                  </tr>
                  {isExpanded && (
                    <tr className="bg-bg-card">
                      <td
                        colSpan={5 + visibleBenchmarks.size}
                        className="px-6 py-3"
                      >
                        <ExpandedDetails
                          run={run}
                          benchmarkTypes={benchmarkTypes}
                          jevComparison={jevComparison}
                        />
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="mt-2 text-xs text-text-muted">
        Showing {sorted.length} of {runs.length} runs
      </div>
    </div>
  );
}

function Th({
  children,
  onClick,
  align = "left",
}: {
  children: React.ReactNode;
  onClick?: () => void;
  align?: "left" | "right";
}) {
  return (
    <th
      className={`th-material px-3 py-2 cursor-pointer hover:text-text select-none whitespace-nowrap ${
        align === "right" ? "text-right" : "text-left"
      }`}
      onClick={onClick}
    >
      {children}
    </th>
  );
}

function ExpandedDetails({
  run,
  benchmarkTypes,
  jevComparison,
}: {
  run: EvalRun;
  benchmarkTypes: Record<string, string>;
  jevComparison: Record<string, JevComparison>;
}) {
  const benchmarks = Object.entries(run.results.by_source).sort(([a], [b]) =>
    a.localeCompare(b)
  );

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <table className="text-xs w-full">
        <thead>
          <tr className="text-text-muted">
            <th className="text-left py-1 font-normal">Benchmark</th>
            <th className="text-left py-1 font-normal">Type</th>
            <th className="text-right py-1 font-normal">Metric</th>
            <th className="text-right py-1 font-normal">ECE</th>
            <th className="text-right py-1 font-normal">vs Jev</th>
          </tr>
        </thead>
        <tbody>
          {benchmarks.map(([name, data]) => {
            const bmType = benchmarkTypes[name] ?? data.type;
            const isScore = bmType === "score";
            const accuracy =
              data.type === "mixed" ? getMixedAccuracy(data) : data.accuracy;
            const jev = jevComparison[name];
            const delta =
              jev && accuracy != null ? accuracy - jev.jev : null;

            return (
              <tr key={name} className="border-t border-border">
                <td className="py-1 font-medium">{name}</td>
                <td className="py-1 text-text-muted">{bmType}</td>
                <td className="py-1 text-right font-mono">
                  {isScore ? fmtMae(data.mae) : pct(accuracy)}
                </td>
                <td className="py-1 text-right font-mono text-text-muted">
                  {data.ece != null ? data.ece.toFixed(3) : "—"}
                </td>
                <td
                  className={`py-1 text-right font-mono ${
                    delta != null
                      ? delta >= 0
                        ? "text-green"
                        : "text-red"
                      : "text-text-muted"
                  }`}
                >
                  {delta != null
                    ? `${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(1)}%`
                    : "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function parseParams(params: string): number {
  const match = params.match(/([\d.]+)\s*([BMK])/i);
  if (!match) return 0;
  const num = parseFloat(match[1]);
  const unit = match[2].toUpperCase();
  if (unit === "B") return num * 1e9;
  if (unit === "M") return num * 1e6;
  if (unit === "K") return num * 1e3;
  return num;
}

function getBenchmarkValue(run: EvalRun, bm: string, bmType: string): number {
  const data = run.results.by_source[bm];
  if (!data) return bmType === "score" ? Infinity : -1;
  if (bmType === "score") return data.mae ?? Infinity;
  if (data.type === "mixed") return getMixedAccuracy(data) ?? -1;
  return data.accuracy ?? -1;
}

function getMixedAccuracy(
  data: { subtypes?: Record<string, { accuracy?: number }> }
): number | null {
  if (!data.subtypes) return null;
  const accs: number[] = [];
  for (const st of ["noul", "choice"]) {
    const sub = data.subtypes[st];
    if (sub?.accuracy != null) accs.push(sub.accuracy);
  }
  return accs.length > 0 ? accs.reduce((a, b) => a + b, 0) / accs.length : null;
}
