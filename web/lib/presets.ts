export interface Preset {
  name: string;
  state: string;
  questionType: "noul" | "choice" | "score";
  instructions: string;
  options: string;
}

export const PRESETS: Preset[] = [
  {
    name: "Banking Intent",
    state: "Customer says: I want to cancel my transfer to John",
    questionType: "choice",
    instructions: "Which intent does this message express?",
    options:
      "cancel_transfer: Wants to cancel a money transfer\ntrack_order: Wants to track a delivery\nbilling: Has a billing question\naccount_info: Wants account information",
  },
  {
    name: "Sentiment Detection",
    state: "The movie was absolutely terrible. I walked out after 20 minutes.",
    questionType: "noul",
    instructions: "Does this review express a negative sentiment?",
    options: "",
  },
  {
    name: "Medical Triage",
    state: "Patient presents with chest pain, shortness of breath, and left arm numbness.",
    questionType: "choice",
    instructions: "What is the urgency level?",
    options:
      "critical: Requires immediate emergency intervention\nurgent: Needs prompt medical attention\nroutine: Can wait for scheduled appointment",
  },
  {
    name: "Satisfaction Score",
    state: "The restaurant had great ambiance but the food was mediocre and overpriced.",
    questionType: "score",
    instructions: "Rate the overall satisfaction level",
    options:
      "1: Very dissatisfied\n2: Dissatisfied\n3: Neutral\n4: Satisfied\n5: Very satisfied",
  },
  {
    name: "Code Classification",
    state: 'def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)',
    questionType: "choice",
    instructions: "What programming paradigm does this code primarily use?",
    options:
      "recursive: Uses recursion as the primary pattern\nimperative: Sequential step-by-step logic\nfunctional: Pure functions with no side effects\noop: Object-oriented with classes",
  },
];

export const MODELS = [
  { id: "oaklight/krino-ettin-150m-heads", label: "Ettin-150m (fastest)", params: "150M" },
  { id: "oaklight/krino-modernbert-base-heads", label: "ModernBERT-base", params: "149M" },
  { id: "oaklight/krino-qwen3-0.6b-heads", label: "Qwen3-0.6B", params: "0.6B" },
  { id: "oaklight/krino-qwen3.5-4b-heads", label: "Qwen3.5-4B", params: "4B" },
  { id: "oaklight/krino-qwen3-reranker-4b-heads", label: "Qwen3-reranker-4B", params: "4B" },
  { id: "oaklight/krino-qwen3-reranker-0.6b-heads", label: "Qwen3-reranker-0.6B", params: "0.6B" },
];
