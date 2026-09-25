"""Krino inference demo — Gradio app for HuggingFace ZeroGPU or self-hosted GPU."""

import importlib.util
import json
import time

import gradio as gr
import torch
from huggingface_hub import hf_hub_download

try:
    import spaces
    HAS_ZEROGPU = True
except ImportError:
    HAS_ZEROGPU = False

MODELS = {
    "oaklight/krino-ettin-150m-heads": "Ettin-150m (fastest)",
    "oaklight/krino-modernbert-base-heads": "ModernBERT-base",
    "oaklight/krino-qwen3-0.6b-heads": "Qwen3-0.6B",
    "oaklight/krino-qwen3.5-4b-heads": "Qwen3.5-4B",
    "oaklight/krino-qwen3-reranker-4b-heads": "Qwen3-reranker-4B",
}

loaded_models: dict = {}


def _load_krino_module(model_id: str):
    """Import the KrinoModel class from a model's krino.py."""
    krino_path = hf_hub_download(model_id, "krino.py")
    spec = importlib.util.spec_from_file_location(f"krino_{model_id}", krino_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_model(model_id: str):
    """Get or load a KrinoModel, placing it on CUDA."""
    if model_id not in loaded_models:
        mod = _load_krino_module(model_id)
        loaded_models[model_id] = mod.KrinoModel.from_pretrained(model_id)
    return loaded_models[model_id]


# Preload default model at module scope (free under ZeroGPU CUDA emulation)
DEFAULT_MODEL_ID = "oaklight/krino-ettin-150m-heads"
_get_model(DEFAULT_MODEL_ID)


def _parse_question(question_type: str, instructions: str, options_text: str) -> dict:
    """Build a question dict from form inputs."""
    question: dict = {"type": question_type, "instructions": instructions}
    if question_type in ("choice", "score"):
        parsed = {}
        for line in options_text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                key, desc = line.split(":", 1)
                parsed[key.strip()] = desc.strip()
            else:
                parsed[line] = line
        question["criteria" if question_type == "choice" else "legend"] = parsed
    return question


def _predict_impl(model_id, state, question_type, instructions, options_text):
    """Core prediction logic."""
    if model_id not in MODELS:
        return {"error": f"Unknown model: {model_id}"}
    question = _parse_question(question_type, instructions, options_text)
    try:
        model = _get_model(model_id)
        t0 = time.perf_counter()
        answer = model.predict(state=state, question=question)
        answer["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        answer["model"] = MODELS.get(model_id, model_id)
        return answer
    except Exception as e:
        return {"error": str(e)}


# Apply @spaces.GPU decorator only when running on ZeroGPU
if HAS_ZEROGPU:
    predict = spaces.GPU(duration=15)(_predict_impl)
else:
    predict = _predict_impl


EXAMPLES = [
    [
        "oaklight/krino-ettin-150m-heads",
        "Customer says: I want to cancel my transfer to John",
        "choice",
        "Which intent does this message express?",
        "cancel_transfer: Wants to cancel a money transfer\ntrack_order: Wants to track a delivery\nbilling: Has a billing question\naccount_info: Wants account information",
    ],
    [
        "oaklight/krino-ettin-150m-heads",
        "The movie was absolutely terrible. I walked out after 20 minutes.",
        "noul",
        "Does this review express a negative sentiment?",
        "",
    ],
    [
        "oaklight/krino-ettin-150m-heads",
        "Patient presents with chest pain, shortness of breath, and left arm numbness.",
        "choice",
        "What is the urgency level?",
        "critical: Requires immediate emergency intervention\nurgent: Needs prompt medical attention\nroutine: Can wait for scheduled appointment",
    ],
    [
        "oaklight/krino-ettin-150m-heads",
        "The restaurant had great ambiance but the food was mediocre and overpriced.",
        "score",
        "Rate the overall satisfaction level",
        "1: Very dissatisfied\n2: Dissatisfied\n3: Neutral\n4: Satisfied\n5: Very satisfied",
    ],
]

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(choices=list(MODELS.keys()), value=DEFAULT_MODEL_ID, label="Model"),
        gr.Textbox(label="State / Input Text", lines=4, placeholder="Enter the text to analyze..."),
        gr.Radio(choices=["noul", "choice", "score"], value="choice", label="Question Type"),
        gr.Textbox(label="Instructions", placeholder="What question should the model answer?"),
        gr.Textbox(label="Options (one per line, format: key: description)", lines=5,
                   placeholder="option1: Description\noption2: Description"),
    ],
    outputs=gr.JSON(label="Result"),
    title="Krino — Decision Model Demo",
    description="Run typed decision queries against open Krino models. Returns calibrated probabilities instead of free text.",
    examples=EXAMPLES,
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
