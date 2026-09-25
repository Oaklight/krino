"use client";

import { useState, useCallback, useEffect } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { PRESETS, MODELS, type Preset } from "@/lib/presets";

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
  const url = `${backendUrl.replace(/\/+$/, "")}/api/predict`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data: args }),
  });
  if (!res.ok) {
    throw new Error(`Server returned ${res.status}: ${await res.text()}`);
  }
  const json = await res.json();
  const raw = json.data?.[0];
  if (typeof raw === "string") return JSON.parse(raw);
  return raw as InferenceResult;
}

export default function PlaygroundClient() {
  const [backendUrl, setBackendUrl] = useState("");
  const [model, setModel] = useState(MODELS[0].id);
  const [state, setState] = useState("");
  const [questionType, setQuestionType] = useState<"noul" | "choice" | "score">("choice");
  const [instructions, setInstructions] = useState("");
  const [options, setOptions] = useState("");
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [latency, setLatency] = useState<number | null>(null);
  const [showJson, setShowJson] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const stored = localStorage.getItem(BACKEND_STORAGE_KEY);
    if (stored) setBackendUrl(stored);
  }, []);

  const saveBackendUrl = useCallback((url: string) => {
    setBackendUrl(url);
    if (url.trim()) {
      localStorage.setItem(BACKEND_STORAGE_KEY, url.trim());
    } else {
      localStorage.removeItem(BACKEND_STORAGE_KEY);
    }
  }, []);

  const applyPreset = useCallback((preset: Preset) => {
    setState(preset.state);
    setQuestionType(preset.questionType);
    setInstructions(preset.instructions);
    setOptions(preset.options);
    setResult(null);
    setError(null);
  }, []);

  const runInference = useCallback(async () => {
    if (!backendUrl.trim()) {
      setError("Enter a backend URL. Run the Colab/Kaggle notebook to get one.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    const t0 = performance.now();

    try {
      const data = await callGradioApi(backendUrl.trim(), [
        model, state, questionType, instructions, options,
      ]);
      const clientLatency = Math.round(performance.now() - t0);
      setLatency(clientLatency);

      if (data.error) {
        setError(data.error);
      } else {
        setResult(data);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Inference failed");
    } finally {
      setLoading(false);
    }
  }, [backendUrl, model, state, questionType, instructions, options]);

  const chartData = result?.probabilities
    ? Object.entries(result.probabilities)
        .map(([name, value]) => ({ name, value: Math.round(value * 1000) / 10 }))
        .sort((a, b) => b.value - a.value)
    : [];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Left panel — inputs */}
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
            <a
              href="https://colab.research.google.com/github/Oaklight/krino/blob/main/notebooks/krino_inference_server.ipynb"
              target="_blank"
              rel="noopener noreferrer"
              className="text-accent hover:text-accent-hover"
            >
              Colab notebook
            </a>{" "}
            to get a free GPU backend URL.
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
                className="px-3 py-1 text-xs border border-border rounded-[var(--radius)] hover:bg-bg-hover transition-colors"
              >
                {p.name}
              </button>
            ))}
          </div>
        </div>

        {/* Model selector */}
        <div>
          <label className="th-material block mb-2">Model</label>
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="w-full px-3 py-2 text-sm border border-border rounded-[var(--radius)] bg-bg-card text-text"
          >
            {MODELS.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label} ({m.params})
              </option>
            ))}
          </select>
        </div>

        {/* State input */}
        <div>
          <label className="th-material block mb-2">State / Input Text</label>
          <textarea
            value={state}
            onChange={(e) => setState(e.target.value)}
            rows={4}
            className="w-full px-3 py-2 text-sm border border-border rounded-[var(--radius)] bg-bg-card text-text placeholder:text-text-muted resize-y font-mono"
            placeholder="Enter the text to analyze..."
          />
        </div>

        {/* Question type */}
        <div>
          <label className="th-material block mb-2">Question Type</label>
          <div className="flex gap-2">
            {(["noul", "choice", "score"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setQuestionType(t)}
                className={`px-4 py-1.5 text-sm rounded-[var(--radius)] border transition-colors ${
                  questionType === t
                    ? "bg-accent text-[var(--accent-on)] border-accent"
                    : "border-border text-text-dim hover:text-text"
                }`}
              >
                {t}
              </button>
            ))}
          </div>
        </div>

        {/* Instructions */}
        <div>
          <label className="th-material block mb-2">Instructions</label>
          <input
            type="text"
            value={instructions}
            onChange={(e) => setInstructions(e.target.value)}
            className="w-full px-3 py-2 text-sm border border-border rounded-[var(--radius)] bg-bg-card text-text placeholder:text-text-muted"
            placeholder="What question should the model answer?"
          />
        </div>

        {/* Options (for choice/score) */}
        {questionType !== "noul" && (
          <div>
            <label className="th-material block mb-2">
              {questionType === "choice" ? "Options" : "Scale Legend"} (one per line: key: description)
            </label>
            <textarea
              value={options}
              onChange={(e) => setOptions(e.target.value)}
              rows={4}
              className="w-full px-3 py-2 text-sm border border-border rounded-[var(--radius)] bg-bg-card text-text placeholder:text-text-muted resize-y font-mono"
              placeholder={"option1: Description\noption2: Description"}
            />
          </div>
        )}

        {/* Run button */}
        <button
          onClick={runInference}
          disabled={loading || !state.trim() || !instructions.trim()}
          className="w-full px-4 py-2.5 rounded-[var(--radius)] bg-accent text-[var(--accent-on)] font-medium text-sm hover:opacity-90 transition-opacity disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {loading ? "Running..." : "Run Inference"}
        </button>
      </div>

      {/* Right panel — results */}
      <div className="space-y-4">
        {error && (
          <div className="border border-red rounded-[var(--radius)] bg-[var(--red-subtle,rgba(255,54,33,0.06))] p-4">
            <p className="text-sm font-medium text-red">Error</p>
            <p className="text-sm text-text-dim mt-1">{error}</p>
          </div>
        )}

        {result && (
          <>
            {/* Summary */}
            <div className="bg-bg-card border border-border rounded-[var(--radius)] p-4">
              <div className="flex items-center justify-between mb-3">
                <span className="th-material">Result</span>
                <div className="flex items-center gap-3 text-xs text-text-muted">
                  {latency != null && <span>{latency}ms (round-trip)</span>}
                  {result.latency_ms != null && <span>{result.latency_ms}ms (server)</span>}
                </div>
              </div>

              {result.type === "noul" && result.noul != null && (
                <div className="text-center py-4">
                  <div className="text-4xl font-bold text-accent">
                    {(result.noul * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-text-muted mt-1">probability of yes</div>
                  <div className="mt-3 h-3 bg-bg-hover rounded-full overflow-hidden">
                    <div
                      className="h-full bg-accent rounded-full transition-all"
                      style={{ width: `${result.noul * 100}%` }}
                    />
                  </div>
                </div>
              )}

              {result.type === "choice" && result.choice && (
                <div>
                  <div className="text-sm mb-1">
                    Selected: <span className="font-medium text-accent">{result.choice}</span>
                    {result.confidence != null && (
                      <span className="text-text-muted ml-2">
                        (confidence: {(result.confidence * 100).toFixed(0)}%)
                      </span>
                    )}
                  </div>
                </div>
              )}

              {result.type === "score" && result.score != null && (
                <div className="text-center py-2">
                  <div className="text-4xl font-bold text-accent">
                    {result.score.toFixed(2)}
                  </div>
                  <div className="text-sm text-text-muted mt-1">predicted score</div>
                </div>
              )}
            </div>

            {/* Probability chart */}
            {chartData.length > 0 && (
              <div className="bg-bg-card border border-border rounded-[var(--radius)] p-4">
                <span className="th-material block mb-3">Probabilities</span>
                <ResponsiveContainer width="100%" height={Math.max(120, chartData.length * 36)}>
                  <BarChart data={chartData} layout="vertical" margin={{ left: 0, right: 20 }}>
                    <XAxis type="number" domain={[0, 100]} tickFormatter={(v) => `${v}%`}
                      tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
                    <YAxis type="category" dataKey="name" width={120}
                      tick={{ fill: "var(--text-dim)", fontSize: 12 }} />
                    <Tooltip formatter={(v) => `${v}%`}
                      contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border-color)", borderRadius: "var(--radius)" }} />
                    <Bar dataKey="value" radius={[0, 3, 3, 0]}>
                      {chartData.map((_, i) => (
                        <Cell key={i} fill={i === 0 ? "var(--accent)" : "var(--text-muted)"} fillOpacity={i === 0 ? 1 : 0.4} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Raw JSON toggle */}
            <div>
              <button
                onClick={() => setShowJson((s) => !s)}
                className="text-xs text-text-muted hover:text-text transition-colors"
              >
                {showJson ? "Hide" : "Show"} raw JSON
              </button>
              {showJson && (
                <pre className="mt-2 p-3 text-xs bg-bg-card border border-border rounded-[var(--radius)] overflow-x-auto font-mono">
                  {JSON.stringify(result, null, 2)}
                </pre>
              )}
            </div>
          </>
        )}

        {!result && !error && !loading && (
          <div className="border border-border rounded-[var(--radius)] p-8 text-center text-text-muted">
            <p className="text-sm">Select a preset or fill in the fields and click Run Inference</p>
          </div>
        )}

        {loading && (
          <div className="border border-border rounded-[var(--radius)] p-8 text-center text-text-muted animate-pulse">
            <p className="text-sm">Running inference...</p>
          </div>
        )}
      </div>
    </div>
  );
}
