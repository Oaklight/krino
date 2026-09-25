export interface QuestionPreset {
  type: "noul" | "choice" | "score";
  instructions: string;
  options: string;
}

export interface Preset {
  name: string;
  state: string;
  questions: QuestionPreset[];
}

export const PRESETS: Preset[] = [
  {
    name: "Banking Intent",
    state: "Customer says: I want to cancel my transfer to John",
    questions: [
      {
        type: "noul",
        instructions: "Is this a cancellation request?",
        options: "",
      },
      {
        type: "choice",
        instructions: "Which intent does this message express?",
        options:
          "cancel_transfer: Wants to cancel a money transfer\ntrack_order: Wants to track a delivery\nbilling: Has a billing question\naccount_info: Wants account information",
      },
      {
        type: "score",
        instructions: "Rate the urgency level",
        options: "1: Not urgent\n2: Slightly urgent\n3: Moderately urgent\n4: Urgent\n5: Critical",
      },
    ],
  },
  {
    name: "Movie Review",
    state: "The movie was absolutely terrible. I walked out after 20 minutes.",
    questions: [
      {
        type: "noul",
        instructions: "Does this review express a negative sentiment?",
        options: "",
      },
      {
        type: "choice",
        instructions: "What is the overall sentiment?",
        options: "positive: Positive sentiment\nnegative: Negative sentiment\nneutral: Neutral sentiment",
      },
      {
        type: "score",
        instructions: "Rate the sentiment from most negative to most positive",
        options: "1: Very negative\n2: Negative\n3: Neutral\n4: Positive\n5: Very positive",
      },
    ],
  },
  {
    name: "Medical Triage",
    state: "Patient presents with chest pain, shortness of breath, and left arm numbness.",
    questions: [
      {
        type: "noul",
        instructions: "Does this require immediate emergency intervention?",
        options: "",
      },
      {
        type: "choice",
        instructions: "What is the urgency level?",
        options:
          "critical: Requires immediate emergency intervention\nurgent: Needs prompt medical attention\nroutine: Can wait for scheduled appointment",
      },
      {
        type: "score",
        instructions: "Rate the severity",
        options: "1: Minimal\n2: Mild\n3: Moderate\n4: Severe\n5: Life-threatening",
      },
    ],
  },
  {
    name: "Restaurant Review",
    state: "The restaurant had great ambiance but the food was mediocre and overpriced.",
    questions: [
      {
        type: "noul",
        instructions: "Would the reviewer recommend this restaurant?",
        options: "",
      },
      {
        type: "choice",
        instructions: "What aspect is the main complaint about?",
        options: "food_quality: Food quality\nprice: Pricing\nservice: Service\nambiance: Ambiance",
      },
      {
        type: "score",
        instructions: "Rate the overall satisfaction level",
        options:
          "1: Very dissatisfied\n2: Dissatisfied\n3: Neutral\n4: Satisfied\n5: Very satisfied",
      },
    ],
  },
  {
    name: "Code Classification",
    state: "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
    questions: [
      {
        type: "noul",
        instructions: "Does this code use recursion?",
        options: "",
      },
      {
        type: "choice",
        instructions: "What programming paradigm does this code primarily use?",
        options:
          "recursive: Uses recursion as the primary pattern\nimperative: Sequential step-by-step logic\nfunctional: Pure functions with no side effects\noop: Object-oriented with classes",
      },
    ],
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
