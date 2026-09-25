"""Krino inference demo — Gradio app for HuggingFace Spaces."""

import json
import time
from pathlib import Path

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

MODELS = {
    "Ettin-150m (fastest)": "oaklight/krino-ettin-150m-heads",
    "ModernBERT-base": "oaklight/krino-modernbert-base-heads",
    "Qwen3-0.6B": "oaklight/krino-qwen3-0.6b-heads",
    "Qwen3.5-4B": "oaklight/krino-qwen3.5-4b-heads",
    "Qwen3-reranker-4B": "oaklight/krino-qwen3-reranker-4b-heads",
}

loaded_models: dict = {}


def load_model(model_id: str):
    """Load a Krino model from HuggingFace."""
    if model_id in loaded_models:
        return loaded_models[model_id]

    config_path = hf_hub_download(model_id, "config.json")
    config = json.loads(Path(config_path).read_text())
    backbone_name = config["backbone"]["name"]

    krino_path = hf_hub_download(model_id, "krino.py")
    import importlib.util
    spec = importlib.util.spec_from_file_location("krino", krino_path)
    krino_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(krino_mod)

    model = krino_mod.KrinoModel.from_pretrained(model_id)
    loaded_models[model_id] = model
    return model


def predict(model_name: str, state: str, question_type: str,
            instructions: str, options_text: str):
    """Run inference and return structured results."""
    model_id = MODELS.get(model_name)
    if not model_id:
        return json.dumps({"error": f"Unknown model: {model_name}"}, indent=2)

    question: dict = {"type": question_type, "instructions": instructions}

    if question_type == "noul":
        pass
    elif question_type == "choice":
        criteria = {}
        for line in options_text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                key, desc = line.split(":", 1)
                criteria[key.strip()] = desc.strip()
            else:
                criteria[line] = line
        question["criteria"] = criteria
    elif question_type == "score":
        legend = {}
        for line in options_text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                key, desc = line.split(":", 1)
                legend[key.strip()] = desc.strip()
            else:
                legend[line] = line
        question["legend"] = legend

    try:
        model = load_model(model_id)
        t0 = time.perf_counter()
        answer = model.predict(state=state, question=question)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        answer["latency_ms"] = latency_ms
        answer["model"] = model_name
        return json.dumps(answer, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


EXAMPLES = [
    [
        "Ettin-150m (fastest)",
        "Customer says: I want to cancel my transfer to John",
        "choice",
        "Which intent does this message express?",
        "cancel_transfer: Wants to cancel a money transfer\ntrack_order: Wants to track a delivery\nbilling: Has a billing question\naccount_info: Wants account information",
    ],
    [
        "Ettin-150m (fastest)",
        "The movie was absolutely terrible. I walked out after 20 minutes.",
        "noul",
        "Does this review express a negative sentiment?",
        "",
    ],
    [
        "Ettin-150m (fastest)",
        "Patient presents with chest pain, shortness of breath, and left arm numbness.",
        "choice",
        "What is the urgency level?",
        "critical: Requires immediate emergency intervention\nurgent: Needs prompt medical attention\nroutine: Can wait for scheduled appointment",
    ],
    [
        "Ettin-150m (fastest)",
        "The restaurant had great ambiance but the food was mediocre and overpriced.",
        "score",
        "Rate the overall satisfaction level",
        "1: Very dissatisfied\n2: Dissatisfied\n3: Neutral\n4: Satisfied\n5: Very satisfied",
    ],
]

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(choices=list(MODELS.keys()), value="Ettin-150m (fastest)", label="Model"),
        gr.Textbox(label="State / Input Text", lines=4, placeholder="Enter the text to analyze..."),
        gr.Radio(choices=["noul", "choice", "score"], value="choice", label="Question Type"),
        gr.Textbox(label="Instructions", placeholder="What question should the model answer?"),
        gr.Textbox(label="Options (one per line, format: key: description)", lines=5,
                   placeholder="option1: Description\noption2: Description"),
    ],
    outputs=gr.JSON(label="Result"),
    title="Krino — Decision Model Demo",
    description="Run typed decision queries against open Krino models. Returns calibrated probabilities.",
    examples=EXAMPLES,
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
