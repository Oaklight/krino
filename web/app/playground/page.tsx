import PlaygroundClient from "@/components/PlaygroundClient";
import ErrorBoundary from "@/components/ErrorBoundary";

export default function PlaygroundPage() {
  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold mb-2">Playground</h1>
      <p className="text-sm text-text-dim mb-6">
        Run live inference against Krino decision models on HuggingFace Spaces.
        Select a preset or build your own query.
      </p>
      <ErrorBoundary>
        <PlaygroundClient />
      </ErrorBoundary>
    </div>
  );
}
