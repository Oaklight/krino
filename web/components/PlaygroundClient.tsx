"use client";

import { useState, useCallback, useEffect } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { PRESETS, MODELS, type Preset, type QuestionPreset } from "@/lib/presets";

const BACKEND_STORAGE_KEY = "krino-backend-url";

interface InferenceResult {
  type: string;
  noul?: number;
  choice?: string;
  probabilities?: Record<string, number>;
  score?: number;
  legend?: Record<string, string>;
  confidence?: number;
  latency_ms?: number;
  model?: string;
  error?: string;
}

async function callGradioApi(
  backendUrl: string,
  args: unknown[]
): Promise<InferenceResult> {
  const base = backendUrl.replace(/\/+$/, "");

  const submitRes = await fetch(`${base}/gradio_api/call/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data: args }),
  });
  if (!submitRes.ok) {
    throw new Error(`Server returned ${submitRes.status}: ${await submitRes.text()}`);
  }
  const { event_id } = await submitRes.json();

  const resultRes = await fetch(`${base}/gradio_api/call/predict/${event_id}`);
  if (!resultRes.ok) {
    throw new Error(`Result fetch failed: ${resultRes.status}`);
  }
  const text = await resultRes.text();
  const dataLine = text.split("\n").find((l) => l.startsWith("data: "));
  if (!dataLine) throw new Error("No data in response");
  const parsed = JSON.parse(dataLine.slice(6));
  const raw = Array.isArray(parsed) ? parsed[0] : parsed;
  if (typeof raw === "string") return JSON.parse(raw);
  return raw as InferenceResult;
}

function ResultCard({ result, latency }: { result: InferenceResult; latency: number | null }) {
  const [showJson, setShowJson] = useState(false);

  const chartData = result.probabilities
    ? Object.entries(result.probabilities)
        .map(([name, value]) => ({ name, value: Math.round(value * 1000) / 10 }))
        .sort((a, b) => b.value - a.value)
    : [];

  return (
    <div className="bg-bg-card border border-border rounded-[var(--radius)] p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="th-material">{result.type} result</span>
        <div className="flex items-center gap-3 text-xs text-text-muted">
          {latency != null && <span>{latency}ms</span>}
          {result.latency_ms != null && <span>{result.latency_ms}ms (gpu)</span>}
        </div>
      </div>

      {result.type === "noul" && result.noul != null && (
        <div className="text-center py-3">
          <div className="text-3xl font-bold text-accent">
            {(result.noul * 100).toFixed(1)}%
          </div>
          <div className="text-xs text-text-muted mt-1">probability of yes</div>
          <div className="mt-2 h-2 bg-bg-hover rounded-full overflow-hidden">
            <div
              className="h-full bg-accent rounded-full transition-all"
              style={{ width: `${result.noul * 100}%` }}
            />
          </div>
        </div>
      )}

      {result.type === "choice" && result.choice && (
        <div className="text-sm mb-2">
          Selected: <span className="font-medium text-accent">{result.choice}</span>
          {result.confidence != null && (
            <span className="text-text-muted ml-2">
              ({(result.confidence * 100).toFixed(0)}%)
            </span>
          )}
        </div>
      )}

      {result.type === "score" && result.score != null && (
        <div className="text-center py-2">
          <div className="text-3xl font-bold text-accent">{result.score.toFixed(2)}</div>
          <div className="text-xs text-text-muted mt-1">predicted score</div>
        </div>
      )}

      {chartData.length > 0 && (
        <ResponsiveContainer width="100%" height={Math.max(80, chartData.length * 28)}>
          <BarChart data={chartData} layout="vertical" margin={{ left: 0, right: 16 }}>
            <XAxis type="number" domain={[0, 100]} tickFormatter={(v) => `${v}%`}
              tick={{ fill: "var(--text-muted)", fontSize: 9 }} />
            <YAxis type="category" dataKey="name" width={100}
              tick={{ fill: "var(--text-dim)", fontSize: 11 }} />
            <Tooltip formatter={(v) => `${v}%`}
              contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border-color)", borderRadius: "var(--radius)" }} />
            <Bar dataKey="value" radius={[0, 3, 3, 0]}>
              {chartData.map((_, i) => (
                <Cell key={i} fill={i === 0 ? "var(--accent)" : "var(--text-muted)"} fillOpacity={i === 0 ? 1 : 0.4} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}

      <button
        onClick={() => setShowJson((s) => !s)}
        className="text-xs text-text-muted hover:text-text transition-colors mt-2"
      >
        {showJson ? "Hide" : "Show"} JSON
      </button>
      {showJson && (
        <pre className="mt-1 p-2 text-xs bg-bg border border-border rounded-[var(--radius)] overflow-x-auto font-mono">
          {JSON.stringify(result, null, 2)}
        </pre>
      )}
    </div>
  );
}

export default function PlaygroundClient() {
  const [backendUrl, setBackendUrl] = useState("");
  const [model, setModel] = useState(MODELS[0].id);
  const [state, setState] = useState("");
  const [activePreset, setActivePreset] = useState<Preset | null>(null);
  // Single-question manual mode
  const [questionType, setQuestionType] = useState<"noul" | "choice" | "score">("choice");
  const [instructions, setInstructions] = useState("");
  const [options, setOptions] = useState("");
  // Results per question type
  const [results, setResults] = useState<Record<string, { result: InferenceResult; latency: number }>>({});
  const [loading, setLoading] = useState<Set<string>>(new Set());
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const stored = localStorage.getItem(BACKEND_STORAGE_KEY);
    if (stored) setBackendUrl(stored);
  }, []);

  const saveBackendUrl = useCallback((url: string) => {
    setBackendUrl(url);
    if (url.trim()) localStorage.setItem(BACKEND_STORAGE_KEY, url.trim());
    else localStorage.removeItem(BACKEND_STORAGE_KEY);
  }, []);

  const applyPreset = useCallback((preset: Preset) => {
    setState(preset.state);
    setActivePreset(preset);
    setResults({});
    setError(null);
    if (preset.questions.length > 0) {
      const q = preset.questions[0];
      setQuestionType(q.type);
      setInstructions(q.instructions);
      setOptions(q.options);
    }
  }, []);

  const runOne = useCallback(async (qType: string, instr: string, opts: string) => {
    if (!backendUrl.trim()) {
      setError("Enter a backend URL. Run the Colab/Kaggle notebook to get one.");
      return;
    }
    setLoading((prev) => new Set(prev).add(qType));
    setError(null);
    const t0 = performance.now();

    try {
      const data = await callGradioApi(backendUrl.trim(), [model, state, qType, instr, opts]);
      const clientLatency = Math.round(performance.now() - t0);
      if (data.error) {
        setError(data.error);
      } else {
        setResults((prev) => ({ ...prev, [qType]: { result: data, latency: clientLatency } }));
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Inference failed");
    } finally {
      setLoading((prev) => { const n = new Set(prev); n.delete(qType); return n; });
    }
  }, [backendUrl, model, state]);

  const runAll = useCallback(async () => {
    if (!activePreset) return;
    for (const q of activePreset.questions) {
      runOne(q.type, q.instructions, q.options);
    }
  }, [activePreset, runOne]);

  const questions = activePreset?.questions ?? [];
  const hasPreset = activePreset != null;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Left panel */}
      <div className="space-y-4">
        {/* Backend URL */}
        <div>
          <label className="th-material block mb-2">Backend URL</label>
          <input
            type="url"
            value={backendUrl}
            onChange={(e) => saveBackendUrl(e.target.value)}
            className="w-full px-3 py-2 text-sm border border-border rounded-[var(--radius)] bg-bg-card text-text placeholder:text-text-muted font-mono"
            placeholder="https://your-tunnel-url.trycloudflare.com"
          />
          <p className="text-xs text-text-muted mt-1">
            Run the{" "}
            <a href="https://colab.research.google.com/github/Oaklight/krino/blob/main/notebooks/krino_inference_server.ipynb"
              target="_blank" rel="noopener noreferrer" className="text-accent hover:text-accent-hover">
              Colab notebook
            </a>{" "}to get a free GPU backend URL.
          </p>
        </div>

        {/* Presets */}
        <div>
          <label className="th-material block mb-2">Presets</label>
          <div className="flex flex-wrap gap-2">
            {PRESETS.map((p) => (
              <button
                key={p.name}
                onClick={() => applyPreset(p)}
                className={`px-3 py-1 text-xs border rounded-[var(--radius)] transition-colors ${
                  activePreset?.name === p.name
                    ? "bg-accent-subtle border-accent text-text"
                    : "border-border hover:bg-bg-hover"
                }`}
              >
                {p.name}
              </button>
            ))}
          </div>
        </div>

        {/* Model */}
        <div>
          <label className="th-material block mb-2">Model</label>
          <select value={model} onChange={(e) => setModel(e.target.value)}
            className="w-full px-3 py-2 text-sm border border-border rounded-[var(--radius)] bg-bg-card text-text">
            {MODELS.map((m) => (
              <option key={m.id} value={m.id}>{m.label} ({m.params})</option>
            ))}
          </select>
        </div>

        {/* State */}
        <div>
          <label className="th-material block mb-2">State / Input Text</label>
          <textarea value={state} onChange={(e) => { setState(e.target.value); setActivePreset(null); }}
            rows={4}
            className="w-full px-3 py-2 text-sm border border-border rounded-[var(--radius)] bg-bg-card text-text placeholder:text-text-muted resize-y font-mono"
            placeholder="Enter the text to analyze..." />
        </div>

        {/* Preset questions or manual mode */}
        {hasPreset ? (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="th-material">Questions ({questions.length})</label>
              <button onClick={runAll} disabled={loading.size > 0 || !state.trim()}
                className="px-3 py-1 text-xs rounded-[var(--radius)] bg-accent text-[var(--accent-on)] hover:opacity-90 disabled:opacity-40">
                {loading.size > 0 ? `Running ${loading.size}...` : "Run All"}
              </button>
            </div>
            {questions.map((q, i) => (
              <div key={i} className="border border-border rounded-[var(--radius)] p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className={`px-2 py-0.5 text-[10px] rounded-full font-medium ${
                    q.type === "noul" ? "bg-blue text-white" :
                    q.type === "choice" ? "bg-accent text-[var(--accent-on)]" :
                    "bg-[var(--purple)] text-white"
                  }`}>{q.type}</span>
                  <button
                    onClick={() => runOne(q.type, q.instructions, q.options)}
                    disabled={loading.has(q.type) || !state.trim()}
                    className="px-2 py-0.5 text-xs border border-border rounded-[var(--radius)] hover:bg-bg-hover disabled:opacity-40 transition-colors"
                  >
                    {loading.has(q.type) ? "..." : "Run"}
                  </button>
                </div>
                <p className="text-sm text-text-dim">{q.instructions}</p>
                {q.options && (
                  <p className="text-xs text-text-muted mt-1 font-mono whitespace-pre-line">{q.options}</p>
                )}
              </div>
            ))}
          </div>
        ) : (
          <>
            {/* Manual question type */}
            <div>
              <label className="th-material block mb-2">Question Type</label>
              <div className="flex gap-2">
                {(["noul", "choice", "score"] as const).map((t) => (
                  <button key={t} onClick={() => setQuestionType(t)}
                    className={`px-4 py-1.5 text-sm rounded-[var(--radius)] border transition-colors ${
                      questionType === t ? "bg-accent text-[var(--accent-on)] border-accent" : "border-border text-text-dim hover:text-text"
                    }`}>{t}</button>
                ))}
              </div>
            </div>
            <div>
              <label className="th-material block mb-2">Instructions</label>
              <input type="text" value={instructions} onChange={(e) => setInstructions(e.target.value)}
                className="w-full px-3 py-2 text-sm border border-border rounded-[var(--radius)] bg-bg-card text-text placeholder:text-text-muted"
                placeholder="What question should the model answer?" />
            </div>
            {questionType !== "noul" && (
              <div>
                <label className="th-material block mb-2">
                  {questionType === "choice" ? "Options" : "Scale Legend"} (key: description)
                </label>
                <textarea value={options} onChange={(e) => setOptions(e.target.value)} rows={4}
                  className="w-full px-3 py-2 text-sm border border-border rounded-[var(--radius)] bg-bg-card text-text placeholder:text-text-muted resize-y font-mono"
                  placeholder={"option1: Description\noption2: Description"} />
              </div>
            )}
            <button onClick={() => runOne(questionType, instructions, options)}
              disabled={loading.size > 0 || !state.trim() || !instructions.trim()}
              className="w-full px-4 py-2.5 rounded-[var(--radius)] bg-accent text-[var(--accent-on)] font-medium text-sm hover:opacity-90 transition-opacity disabled:opacity-40 disabled:cursor-not-allowed">
              {loading.size > 0 ? "Running..." : "Run Inference"}
            </button>
          </>
        )}
      </div>

      {/* Right panel — results */}
      <div className="space-y-4">
        {error && (
          <div className="border border-red rounded-[var(--radius)] bg-[var(--red-subtle,rgba(255,54,33,0.06))] p-4">
            <p className="text-sm font-medium text-red">Error</p>
            <p className="text-sm text-text-dim mt-1">{error}</p>
          </div>
        )}

        {Object.entries(results).map(([qType, { result, latency }]) => (
          <ResultCard key={qType} result={result} latency={latency} />
        ))}

        {Object.keys(results).length === 0 && !error && loading.size === 0 && (
          <div className="border border-border rounded-[var(--radius)] p-8 text-center text-text-muted">
            <p className="text-sm">Select a preset or fill in the fields and click Run</p>
          </div>
        )}

        {loading.size > 0 && (
          <div className="border border-border rounded-[var(--radius)] p-6 text-center text-text-muted animate-pulse">
            <p className="text-sm">Running inference... (first call loads the model, ~15-30s)</p>
          </div>
        )}
      </div>
    </div>
  );
}
