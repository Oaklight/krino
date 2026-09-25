"use client";

import { useMemo, useState } from "react";
import type { JevBenchRun } from "@/lib/types";
import { pct } from "@/lib/format";

type SortKey = "model" | "params" | "accuracy" | "choice" | "noul" | "score" | "easy" | "hard" | "original" | "latency";
type SortDir = "asc" | "desc";

const TYPE_COLORS: Record<string, string> = {
  causal: "bg-blue text-white",
  encoder: "bg-green text-white",
  reranker: "bg-[var(--accent)] text-[var(--accent-on)]",
};

interface Props {
  runs: JevBenchRun[];
}

export default function JevBenchTable({ runs }: Props) {
  const [sortKey, setSortKey] = useState<SortKey>("accuracy");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const sorted = useMemo(() => {
    return [...runs].sort((a, b) => {
      let cmp = 0;
      switch (sortKey) {
        case "model": cmp = a.model.name.localeCompare(b.model.name); break;
        case "params": cmp = parseParams(a.model.params) - parseParams(b.model.params); break;
        case "accuracy": cmp = a.accuracy - b.accuracy; break;
        case "choice": cmp = (a.by_type.choice?.accuracy ?? -1) - (b.by_type.choice?.accuracy ?? -1); break;
        case "noul": cmp = (a.by_type.noul?.accuracy ?? -1) - (b.by_type.noul?.accuracy ?? -1); break;
        case "score": cmp = (a.by_type.score?.accuracy ?? -1) - (b.by_type.score?.accuracy ?? -1); break;
        case "easy": cmp = (a.by_tier.easy?.accuracy ?? -1) - (b.by_tier.easy?.accuracy ?? -1); break;
        case "hard": cmp = (a.by_tier.hard?.accuracy ?? -1) - (b.by_tier.hard?.accuracy ?? -1); break;
        case "original": cmp = (a.by_tier.original?.accuracy ?? -1) - (b.by_tier.original?.accuracy ?? -1); break;
        case "latency": cmp = (a.latency_ms ?? 9999) - (b.latency_ms ?? 9999); break;
      }
      return sortDir === "asc" ? cmp : -cmp;
    });
  }, [runs, sortKey, sortDir]);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir(key === "model" ? "asc" : "desc");
    }
  };

  const arrow = (key: SortKey) => sortKey === key ? (sortDir === "asc" ? " ▲" : " ▼") : "";

  return (
    <div className="overflow-x-auto border border-border rounded-[var(--radius)]">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border bg-bg-card">
            <Th onClick={() => toggleSort("model")}>Model{arrow("model")}</Th>
            <Th onClick={() => toggleSort("params")} align="right">Params{arrow("params")}</Th>
            <Th onClick={() => toggleSort("accuracy")} align="right">Overall{arrow("accuracy")}</Th>
            <Th onClick={() => toggleSort("choice")} align="right">Choice{arrow("choice")}</Th>
            <Th onClick={() => toggleSort("noul")} align="right">Noul{arrow("noul")}</Th>
            <Th onClick={() => toggleSort("score")} align="right">Score{arrow("score")}</Th>
            <Th onClick={() => toggleSort("easy")} align="right">Easy{arrow("easy")}</Th>
            <Th onClick={() => toggleSort("hard")} align="right">Hard{arrow("hard")}</Th>
            <Th onClick={() => toggleSort("original")} align="right">Original{arrow("original")}</Th>
            <Th onClick={() => toggleSort("latency")} align="right">Latency{arrow("latency")}</Th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((run) => (
            <tr key={run.run_id} className="border-b border-border hover:bg-bg-hover transition-colors">
              <td className="px-3 py-2 font-medium whitespace-nowrap">
                <span className={`inline-block px-1.5 py-0.5 text-[10px] rounded-full mr-2 ${TYPE_COLORS[run.model.type] ?? ""}`}>
                  {run.model.type}
                </span>
                {run.model.name}
              </td>
              <td className="px-3 py-2 text-right text-text-dim">{run.model.params}</td>
              <td className="px-3 py-2 text-right font-mono font-medium">{pct(run.accuracy)}</td>
              <td className="px-3 py-2 text-right font-mono text-text-dim">{pct(run.by_type.choice?.accuracy)}</td>
              <td className="px-3 py-2 text-right font-mono text-text-dim">{pct(run.by_type.noul?.accuracy)}</td>
              <td className="px-3 py-2 text-right font-mono text-text-dim">{pct(run.by_type.score?.accuracy)}</td>
              <td className="px-3 py-2 text-right font-mono text-text-dim">{pct(run.by_tier.easy?.accuracy)}</td>
              <td className="px-3 py-2 text-right font-mono text-text-dim">{pct(run.by_tier.hard?.accuracy)}</td>
              <td className="px-3 py-2 text-right font-mono text-text-dim">{pct(run.by_tier.original?.accuracy)}</td>
              <td className="px-3 py-2 text-right font-mono text-text-dim">
                {run.latency_ms != null ? `${Math.round(run.latency_ms)}ms` : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Th({ children, onClick, align = "left" }: { children: React.ReactNode; onClick?: () => void; align?: "left" | "right" }) {
  return (
    <th className={`th-material px-3 py-2 cursor-pointer hover:text-text select-none whitespace-nowrap ${align === "right" ? "text-right" : "text-left"}`}
      onClick={onClick}>{children}</th>
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
