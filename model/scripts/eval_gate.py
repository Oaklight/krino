#!/usr/bin/env python3
"""One-command evaluation gate for Krino v3.

Loads a checkpoint and writes one JSON with every acceptance field: accuracy,
ECE, Brier, Brier skill, QWK, RPS, MAE, per-type, per-source, per-option-count,
per-schema, schema holdout, risk-coverage, option-order robustness, and an
optional K-sweep.

Acceptance is JevBench public. Banking77 is diagnostic only.

Examples:

    python -m model.scripts.eval_gate --repo oaklight/krino-qwen3-reranker-0.6b-heads \
        --jevbench --output results/gate/reranker06_jevbench.json

    python -m model.scripts.eval_gate --model Qwen/Qwen3-Reranker-0.6B \
        --heads checkpoints/heads.pt --data data/benchmarks/stsb.jsonl \
        --arm reranker06_stsb --output results/gate/reranker06_stsb.json

The arm name, data digest, and every configuration field are written into the
JSON so a number can never be separated from the run that produced it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.pipeline import load_jevbench, load_jsonl
from model.evaluation.gate import (
    Prediction,
    k_bucket,
    order_sensitivity,
    risk_coverage,
    schema_key,
    summarize,
    summarize_by,
    summarize_schema_split,
    chance_corrected_accuracy,
)
from model.src.backbone import load_causal_lm, load_encoder, load_qwen35_base
from model.src.decision_model import DecisionModel
from model.src.gpu_config import GPUConfig
from model.scripts.train_residual import fusion_logits, head_logits


# ---------------------------------------------------------------------------
# Raw Qwen3-Reranker baseline
# ---------------------------------------------------------------------------

RAW_RERANKER_INSTRUCTION = (
    "Classify: which option correctly answers the question for the given situation? "
    "Rate how well each option fits."
)


def _raw_options(
    question: dict[str, Any], reverse: bool = False
) -> list[tuple[str, str]]:
    criteria = question.get("criteria", {})
    q_type = question.get("type")
    if q_type == "noul":
        options = [
            ("no", str(criteria.get("false", criteria.get("no", "No")))),
            ("yes", str(criteria.get("true", criteria.get("yes", "Yes")))),
        ]
    elif q_type == "choice":
        options = [(str(key), str(value)) for key, value in criteria.items()]
    elif q_type == "score":
        options = [(str(i), str(description)) for i, description in enumerate(criteria)]
    else:
        raise ValueError(f"Unsupported question type: {q_type}")
    return list(reversed(options)) if reverse else options


class RawReranker:
    """Raw Qwen3-Reranker, judged with yes/no logits.

    This is the baseline the goal's +10 pp training target is measured against.
    Options are scored independently and their P(yes) values are normalised into
    a distribution so the same gate metrics apply.
    """

    def __init__(self, model_name: str, device: str) -> None:
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16
        ).to(device)
        self.model.eval()
        self.true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.false_id = self.tokenizer.convert_tokens_to_ids("no")
        prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query "
            "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
            "<|im_end|>\n<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
        self.max_length = 8192

    @torch.inference_mode()
    def _score_option(
        self,
        state: str,
        instructions: str,
        q_type: str,
        label: str,
        description: str,
    ) -> float:
        query = f"{state}\n\nQuestion: {instructions}"
        if q_type == "score":
            document = f"Rubric level {label}: {description}"
        else:
            document = f"Answer option {label}: {description}"
        text = (
            f"<Instruct>: {RAW_RERANKER_INSTRUCTION}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
        )
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        budget = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        input_ids = self.prefix_tokens + tokens[:budget] + self.suffix_tokens
        inputs = {
            "input_ids": torch.tensor([input_ids], device=self.device),
            "attention_mask": torch.ones(1, len(input_ids), device=self.device),
        }
        logits = self.model(**inputs).logits[:, -1, :]
        true_logit = logits[:, self.true_id].item()
        false_logit = logits[:, self.false_id].item()
        return torch.softmax(
            torch.tensor([false_logit, true_logit]), dim=0
        )[1].item()

    @torch.inference_mode()
    def score_many(
        self,
        state: str,
        instructions: str,
        q_type: str,
        options: list[tuple[str, str]],
        *,
        group_size: int = 16,
    ) -> list[float]:
        """P(yes) for every option, sharing one prefix KV cache.

        ``_score_option`` re-encodes the identical instruct+query prefix for
        every option, so its cost is O(K) full forwards.  Here the prefix is
        encoded once and the documents are scored as rows of a batched
        forward: same computation, O(1) prefix passes.
        """
        query = f"{state}\n\nQuestion: {instructions}"
        # End the prefix at the colon and lead every document with the
        # space, so prefix + document tokenises exactly like the full string
        # in _score_option; a prefix ending in a space merges that space into
        # the first document token and changes the prompt.
        prefix_text = (
            f"<Instruct>: {RAW_RERANKER_INSTRUCTION}\n"
            f"<Query>: {query}\n"
            "<Document>:"
        )
        prefix_ids = self.prefix_tokens + self.tokenizer.encode(
            prefix_text, add_special_tokens=False
        )
        prefix_len = len(prefix_ids)
        budget = self.max_length - prefix_len - len(self.suffix_tokens)
        if budget <= 0:
            return [
                self._score_option(state, instructions, q_type, label, desc)
                for label, desc in options
            ]

        docs: list[list[int]] = []
        for label, description in options:
            if q_type == "score":
                document = f" Rubric level {label}: {description}"
            else:
                document = f" Answer option {label}: {description}"
            ids = self.tokenizer.encode(document, add_special_tokens=False)[:budget]
            docs.append(ids + self.suffix_tokens)
        if not docs:
            return []

        prefix_tensor = torch.tensor([prefix_ids], device=self.device)
        prefix_mask = torch.ones(1, prefix_len, dtype=torch.long, device=self.device)
        prefix_out = self.model(
            input_ids=prefix_tensor, attention_mask=prefix_mask, use_cache=True
        )
        cache = prefix_out.past_key_values
        rows = min(group_size, len(docs))
        if rows > 1:
            cache.batch_repeat_interleave(rows)
        widest = max(len(doc) for doc in docs)
        base = getattr(self.model, "model", None)

        scores: list[float] = []
        for start in range(0, len(docs), group_size):
            group = docs[start : start + group_size]
            padded = group + [group[-1]] * (rows - len(group))
            ids = torch.zeros((rows, widest), dtype=torch.long, device=self.device)
            mask = torch.zeros(
                (rows, prefix_len + widest), dtype=torch.long, device=self.device
            )
            mask[:, :prefix_len] = 1
            positions = torch.full(
                (rows, widest), prefix_len, dtype=torch.long, device=self.device
            )
            last = torch.zeros(rows, dtype=torch.long, device=self.device)
            for row, doc in enumerate(padded):
                n = len(doc)
                ids[row, :n] = torch.tensor(doc, device=self.device)
                mask[row, prefix_len : prefix_len + n] = 1
                positions[row, :n] = torch.arange(
                    prefix_len, prefix_len + n, device=self.device
                )
                last[row] = n - 1
            inputs = {
                "input_ids": ids,
                "attention_mask": mask,
                "position_ids": positions,
                "past_key_values": cache,
            }
            if base is not None:
                hidden = base(**inputs).last_hidden_state
                picked = hidden[torch.arange(rows, device=self.device), last]
                logits = self.model.lm_head(picked)
            else:
                logits = self.model(**inputs).logits[:, -1, :]
            pair = torch.stack(
                [logits[:, self.false_id], logits[:, self.true_id]], dim=-1
            )
            scores.extend(torch.softmax(pair, dim=-1)[:, 1].tolist())
            cache.crop(-widest)
        return scores[: len(docs)]

    def predict_row(
        self,
        item: Any,
        question: dict[str, Any] | None = None,
        item_id: str | None = None,
        reverse: bool = False,
    ) -> Prediction:
        q = question or item.question
        options = _raw_options(q, reverse=reverse)
        scores = [
            self._score_option(
                item.state, q.get("instructions", ""), q["type"], label, description
            )
            for label, description in options
        ]
        total = sum(scores)
        if total > 0:
            probs = tuple(score / total for score in scores)
        else:
            probs = tuple(1.0 / len(scores) for _ in scores)
        return Prediction(
            item_id=item_id or item.id,
            q_type=q["type"],
            label=item.label,
            probs=probs,
            keys=tuple(label for label, _ in options),
            source=item.source,
            schema=schema_key(q),
            tier=jevbench_tier(item),
        )


_JEVBENCH_TIERS = ("easy", "hard", "original")


def jevbench_tier(item: Any) -> str:
    """Tier of a JevBench item, from its id prefix.

    The public suite ships three files whose ids start with the tier name.
    ``load_jevbench`` flattens them into one stream and keeps no tier field,
    so the id is the only stable signal.
    """
    if getattr(item, "source", "") != "jevbench":
        return ""
    head = str(getattr(item, "id", "")).split("-", 1)[0]
    return head if head in _JEVBENCH_TIERS else ""


class ResidualScorer:
    """Base reranker plus a zero-gated residual head.

    ``logits = log(p_base) + alpha_type * head_logits``.  With the alphas a
    residual checkpoint carries, the arm is exactly the base wherever an
    alpha is 0.
    """

    def __init__(self, base_model_name: str, device: str, checkpoint_path: Path) -> None:
        from model.src.backbone import load_causal_lm

        self.device = device
        self.base = RawReranker(base_model_name, device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        backbone, tokenizer = load_causal_lm(
            config["backbone"], device=device, freeze=True, flash_attention=False
        )
        gpu = GPUConfig(
            flash_attention=False,
            bf16_compute=True,
            max_length=config["max_length"],
            gradient_checkpointing=False,
        )
        self.model = DecisionModel(
            backbone=backbone,
            tokenizer=tokenizer,
            rank=config["rank"],
            mlp_layers=0,
            gpu_config=gpu,
            max_length=config["max_length"],
            # A marker arm must be rebuilt with the same readout it trained
            # with, or the head's input and its saved weights disagree.
            option_marker=bool(config.get("option_marker", False)),
            option_chunk_size=int(config.get("option_chunk_size") or 1),
            option_max_tokens=int(config.get("option_max_tokens") or 64),
            option_batch=int(config.get("option_batch") or 1),
            # LoRA modules must exist before their weights can load.
            lora_layers=config.get("lora_layers"),
            lora_rank=int(config.get("lora_rank") or 16),
            lora_alpha=int(config.get("lora_alpha") or 32),
            answerable_head=bool(config.get("answerable_head", False)),
        )
        self.model.load_state_dict(checkpoint["heads"], strict=False)
        self.model.eval()
        self.alpha = {k: float(v) for k, v in checkpoint["alpha"].items()}

    @torch.inference_mode()
    def predict_row(
        self,
        item: Any,
        question: dict[str, Any] | None = None,
        item_id: str | None = None,
        reverse: bool = False,
    ) -> Prediction:
        q = question or item.question
        options = _raw_options(q, reverse=reverse)
        scores = [
            self.base._score_option(
                item.state, q.get("instructions", ""), q["type"], label, description
            )
            for label, description in options
        ]
        total = sum(scores) or 1.0
        base = torch.tensor([score / total for score in scores], dtype=torch.float32)
        head = head_logits(self.model, item, q).reshape(-1).cpu()
        alpha = torch.tensor(self.alpha.get(q["type"], 0.0))
        logits = fusion_logits(base, head, alpha)
        probs = tuple(torch.softmax(logits, dim=-1).tolist())
        return Prediction(
            item_id=item_id or item.id,
            q_type=q["type"],
            label=item.label,
            probs=probs,
            keys=tuple(label for label, _ in options),
            source=item.source,
            schema=schema_key(q),
            tier=jevbench_tier(item),
        )

    @torch.inference_mode()
    def p_answerable(self, item: Any) -> float | None:
        """Answerable probability, or None when the checkpoint has no head."""
        if getattr(self.model, "answerable_head", None) is None:
            return None
        logit = self.model.forward_answerable(
            item.state, item.question.get("instructions", "")
        )
        return float(torch.sigmoid(logit).reshape(()).item())


def write_prediction_records(
    path: Path, rows: list[Prediction], items: list[Any], model: Any
) -> int:
    """Write the per-item JSONL ``eval_abstention.py`` consumes.

    Fields: id, q_type, n_options, gold, pred, correct, max_prob,
    p_answerable, reason.  ``p_answerable`` is null unless the model carries
    the answerable head.
    """
    by_id = {item.id: item for item in items}
    p_answerable = getattr(model, "p_answerable", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            item = by_id.get(row.item_id)
            record = {
                "id": row.item_id,
                "q_type": row.q_type,
                "n_options": len(row.probs),
                "gold": row.label,
                "pred": row.predicted_key,
                "correct": bool(row.correct),
                "max_prob": float(row.confidence),
                "p_answerable": (
                    p_answerable(item) if (p_answerable and item is not None) else None
                ),
                "reason": getattr(item, "reason", None) if item is not None else None,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _backbone_loader(loader_type: str, name: str, device: str, dtype: torch.dtype):
    if loader_type == "encoder":
        return load_encoder(name, device=device, dtype=dtype)
    if loader_type == "qwen35_base":
        return load_qwen35_base(name, device=device, dtype=dtype)
    return load_causal_lm(name, device=device, dtype=dtype)


def load_model(args: argparse.Namespace, device: str) -> tuple[DecisionModel, str]:
    dtype = torch.float32 if device == "cpu" or args.dtype == "float32" else torch.bfloat16

    if args.repo:
        config_path = hf_hub_download(args.repo, "config.json")
        with open(config_path) as handle:
            config = json.load(handle)
        backbone_name = config["backbone"]["name"]
        loader_type = config["backbone"].get("loader", "causal_lm")
        rank = int(config["heads"]["rank"])
        dropout = float(config["heads"].get("dropout", 0.1))
        backbone, tokenizer = _backbone_loader(loader_type, backbone_name, device, dtype)
        model = DecisionModel(
            backbone=backbone,
            tokenizer=tokenizer,
            rank=rank,
            dropout=dropout,
            readout=args.readout,
            readout_n=args.readout_n,
            readout_layer=args.readout_layer,
            lora_layers=args.lora_layers,
            lora_rank=args.lora_rank,
            max_length=args.max_length,
            option_marker=args.option_marker,
            option_chunk_size=args.option_chunk_size,
            option_max_tokens=args.option_max_tokens,
        )
        weights_path = hf_hub_download(args.repo, "head_weights.safetensors")
        model.load_state_dict(load_file(weights_path, device=device), strict=False)
    else:
        if not args.model or not args.heads:
            raise SystemExit("Specify --repo, or both --model and --heads")
        backbone, tokenizer = _backbone_loader(
            "encoder" if args.encoder else ("qwen35_base" if args.qwen35_base else "causal_lm"),
            args.model,
            device,
            dtype,
        )
        model = DecisionModel(
            backbone=backbone,
            tokenizer=tokenizer,
            rank=args.rank,
            dropout=args.dropout,
            readout=args.readout,
            readout_n=args.readout_n,
            readout_layer=args.readout_layer,
            lora_layers=args.lora_layers,
            lora_rank=args.lora_rank,
            max_length=args.max_length,
            option_marker=args.option_marker,
            option_chunk_size=args.option_chunk_size,
            option_max_tokens=args.option_max_tokens,
        )
        model.load_heads(args.heads)

    model.eval()
    return model, backbone_name if args.repo else args.model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@torch.inference_mode()
def predict_row(
    model: DecisionModel | RawReranker,
    item: Any,
    question: dict[str, Any] | None = None,
    item_id: str | None = None,
) -> Prediction:
    """Score one item without the rounding ``DecisionModel.predict`` applies."""
    if isinstance(model, (RawReranker, ResidualScorer)):
        return model.predict_row(item, question, item_id)
    q = question or item.question
    q_type = q.get("type")
    state = item.state
    instructions = q.get("instructions", "")

    if q_type == "noul":
        logit = model.forward_noul(state, instructions)
        prob_true = torch.sigmoid(logit).reshape(-1)[0].item()
        keys = ("no", "yes")
        probs = (1.0 - prob_true, prob_true)
    elif q_type == "choice":
        criteria = q["criteria"]
        keys = tuple(str(key) for key in criteria)
        texts = [criteria[key] or key for key in keys]
        logits = model.forward_choice(state, instructions, texts)
        probs = tuple(torch.softmax(logits, dim=-1)[0].tolist())
    elif q_type == "score":
        criteria = q["criteria"]
        keys = tuple(str(i) for i in range(len(criteria)))
        logits = model.forward_score(state, instructions, criteria)
        probs = tuple(torch.softmax(logits, dim=-1)[0].tolist())
    else:
        raise ValueError(f"Unsupported question type: {q_type}")

    return Prediction(
        item_id=item_id or item.id,
        q_type=q_type,
        label=item.label,
        probs=probs,
        keys=keys,
        source=item.source,
        schema=schema_key(q),
        tier=jevbench_tier(item),
    )


@torch.inference_mode()
def predict_reversed(model: DecisionModel | RawReranker, item: Any) -> Prediction:
    """Score the same item with its options or levels presented in reverse."""
    if isinstance(model, (RawReranker, ResidualScorer)):
        return model.predict_row(item, reverse=True)
    q = item.question
    q_type = q.get("type")
    if q_type == "choice":
        criteria = q["criteria"]
        keys = list(criteria)[::-1]
        texts = [criteria[key] or key for key in keys]
        logits = model.forward_choice(item.state, q.get("instructions", ""), texts)
        probs = tuple(torch.softmax(logits, dim=-1)[0].tolist())
        return Prediction(
            item_id=item.id,
            q_type="choice",
            label=item.label,
            probs=probs,
            keys=tuple(str(key) for key in keys),
            source=item.source,
            schema=schema_key(q),
            tier=jevbench_tier(item),
        )
    if q_type == "score":
        criteria = q["criteria"]
        n_levels = len(criteria)
        order = list(range(n_levels))[::-1]
        texts = [criteria[i] for i in order]
        logits = model.forward_score(item.state, q.get("instructions", ""), texts)
        probs = tuple(torch.softmax(logits, dim=-1)[0].tolist())
        # Logit position j holds the text of original level n_levels-1-j.
        keys = tuple(str(n_levels - 1 - j) for j in range(n_levels))
        return Prediction(
            item_id=item.id,
            q_type="score",
            label=item.label,
            probs=probs,
            keys=keys,
            source=item.source,
            schema=schema_key(q),
        )
    raise ValueError(f"Order sensitivity is undefined for {q_type}")


# ---------------------------------------------------------------------------
# K-sweep
# ---------------------------------------------------------------------------


def build_distractor_pool(items: list[Any]) -> list[tuple[str, str]]:
    """Deduplicated (key, text) distractors from every choice item."""
    pool: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in items:
        if item.question.get("type") != "choice":
            continue
        for key, text in item.question.get("criteria", {}).items():
            label = text or str(key)
            if label not in seen:
                seen.add(label)
                pool.append((str(key), label))
    return pool


def make_k_question(
    item: Any,
    pool: list[tuple[str, str]],
    k: int,
    seed: int,
) -> dict[str, Any] | None:
    """Keep the gold option and draw ``k-1`` distractors from other items."""
    criteria = item.question["criteria"]
    gold = str(item.label)
    if gold not in criteria:
        return None
    gold_text = criteria[gold] or gold
    candidates = [
        (key, text) for key, text in pool if key != gold and text != gold_text
    ]
    if len(candidates) < k - 1:
        return None
    rng = random.Random(f"{seed}:{item.id}:{k}")
    picked = rng.sample(candidates, k - 1)
    merged = {gold: gold_text}
    for j, (key, text) in enumerate(picked):
        merged[f"d{j}:{key}"] = text
    question = dict(item.question)
    question["criteria"] = merged
    return question


def run_k_sweep(
    model: DecisionModel,
    items: list[Any],
    pool_items: list[Any],
    ks: list[int],
    max_items: int,
    seed: int,
    n_bins: int,
) -> dict[str, dict[str, Any]]:
    pool = build_distractor_pool(pool_items)
    choices = [item for item in items if item.question.get("type") == "choice"]
    if max_items:
        choices = choices[:max_items]
    results: dict[str, dict[str, Any]] = {}
    for k in ks:
        rows: list[Prediction] = []
        for item in choices:
            question = make_k_question(item, pool, k, seed)
            if question is None:
                continue
            rows.append(
                predict_row(model, item, question=question, item_id=f"{item.id}-k{k}")
            )
        results[str(k)] = summarize(rows, n_bins)
        print(f"    K={k:<4} n={results[str(k)].get('n', 0)}")
    return results


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def items_digest(items: list[Any]) -> str:
    digest = hashlib.sha256()
    for item in items:
        digest.update(item.id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_items(args: argparse.Namespace) -> tuple[list[Any], str]:
    if args.jevbench:
        items = list(load_jevbench())
        return items, "jevbench"
    items: list[Any] = []
    for path in args.data or []:
        items.extend(load_jsonl(Path(path)))
    if not items:
        raise SystemExit("Specify --jevbench or --data")
    return items, ",".join(str(path) for path in args.data)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Krino v3 evaluation gate")
    parser.add_argument("--repo", help="HuggingFace repo with config.json and head_weights.safetensors")
    parser.add_argument("--model", help="Backbone name for a local checkpoint")
    parser.add_argument("--heads", type=Path, help="Local head checkpoint")
    parser.add_argument(
        "--residual-heads",
        type=Path,
        default=None,
        help="residual.pt from train_residual.py; fuses with the raw reranker",
    )
    parser.add_argument("--encoder", action="store_true", help="Backbone is an encoder")
    parser.add_argument("--qwen35-base", action="store_true", help="Backbone uses the Qwen3.5 base loader")
    parser.add_argument(
        "--raw-reranker",
        action="store_true",
        help="Score with Qwen3-Reranker yes/no logits instead of Krino heads",
    )
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--readout", default="last", help="last|mean_last_n|weighted|layer|fuse_pair")
    parser.add_argument("--readout-n", type=int, default=4)
    parser.add_argument("--readout-layer", type=int, default=None)
    parser.add_argument("--lora-layers", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument(
        "--option-marker",
        action="store_true",
        help="Score choice/score options at a marker token after each option",
    )
    parser.add_argument("--option-chunk-size", type=int, default=32)
    parser.add_argument("--option-max-tokens", type=int, default=64)
    parser.add_argument(
        "--option-batch",
        type=int,
        default=None,
        help="Override the checkpoint's option_batch (options scored per forward)",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--jevbench", action="store_true", help="Evaluate the JevBench public suite")
    parser.add_argument("--data", nargs="+", type=Path, help="One or more TypedQuestion JSONL files")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--bins", type=int, default=10, help="ECE bins")
    parser.add_argument("--no-order-sensitivity", action="store_true")
    parser.add_argument("--k-sweep", default="", help="Comma-separated option counts, e.g. 4,16,64,256")
    parser.add_argument("--k-sweep-max-items", type=int, default=50)
    parser.add_argument("--schema-split", type=Path, default=None,
                        help="JSON mapping schema hash to seen|heldout")
    parser.add_argument("--arm", default=None, help="Arm name written into the JSON")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=None,
        help="Per-item JSONL for eval_abstention.py (needs the answerable head)",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if args.residual_heads:
        raw_name = args.model or "Qwen/Qwen3-Reranker-4B"
        print(f"Residual arm on {raw_name}: {args.residual_heads}")
        model = ResidualScorer(raw_name, device, args.residual_heads)
        if args.option_batch:
            model.model.option_batch = args.option_batch
        backbone_name = f"{raw_name}+residual"
    else:
        model, backbone_name = load_model(args, device)
    items, data_label = load_items(args)
    all_items = items
    if args.max_items:
        items = items[: args.max_items]
    print(f"Items: {len(items)} ({data_label})")

    arm = args.arm or f"{backbone_name.split('/')[-1]}_{args.readout}"
    arm = arm.replace("/", "_").replace(" ", "_")
    output = args.output or Path(f"results/gate/{arm}.json")

    rows: list[Prediction] = []
    latencies: list[float] = []
    failed = 0
    for i, item in enumerate(items):
        start = time.monotonic()
        try:
            rows.append(predict_row(model, item))
        except Exception as error:  # noqa: BLE001 - report and keep the run alive
            print(f"  WARN: {item.id} failed: {error}", file=sys.stderr)
            failed += 1
            continue
        latencies.append(time.monotonic() - start)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(items)}")

    if args.predictions_out:
        written = write_prediction_records(args.predictions_out, rows, items, model)
        print(f"Predictions: {written} rows -> {args.predictions_out}")

    order: dict[str, Any] = {}
    if not args.no_order_sensitivity:
        source_rows = [row for row in rows if row.q_type in ("choice", "score")]
        permuted: list[Prediction] = []
        for item in items:
            if item.question.get("type") not in ("choice", "score"):
                continue
            try:
                permuted.append(predict_reversed(model, item))
            except Exception as error:  # noqa: BLE001
                print(f"  WARN: reversed {item.id} failed: {error}", file=sys.stderr)
        order = order_sensitivity(source_rows, permuted)

    sweep: dict[str, Any] = {}
    if args.k_sweep:
        ks = [int(value) for value in args.k_sweep.split(",") if value.strip()]
        print(f"K-sweep: {ks}")
        try:
            sweep = run_k_sweep(
                model, items, all_items, ks, args.k_sweep_max_items, args.seed, args.bins
            )
        except Exception as error:  # noqa: BLE001 - keep the main results
            # The 231-item pass is the acceptance number; a sweep failure must
            # not discard it after an hour of 4B inference.
            print(f"  WARN: K-sweep failed: {error}", file=sys.stderr)
            sweep = {"error": f"{type(error).__name__}: {error}"}

    schema_split: dict[str, str] | None = None
    if args.schema_split:
        with open(args.schema_split) as handle:
            schema_split = json.load(handle)

    by_schema = summarize_by(rows, lambda row: row.schema, args.bins)
    top_schemas = dict(
        sorted(by_schema.items(), key=lambda pair: -pair[1].get("n", 0))[:50]
    )

    latencies.sort()
    result = {
        "arm": arm,
        "config": {
            "backbone": backbone_name,
            "readout": args.readout,
            "readout_n": args.readout_n,
            "readout_layer": args.readout_layer,
            "option_marker": args.option_marker,
            "option_chunk_size": args.option_chunk_size if args.option_marker else None,
            "option_max_tokens": args.option_max_tokens if args.option_marker else None,
            "max_length": args.max_length,
            "lora_layers": args.lora_layers,
            "lora_rank": args.lora_rank,
            "seed": args.seed,
            "bins": args.bins,
            "data": data_label,
            "data_sha256": items_digest(items),
        },
        "overall": summarize(rows, args.bins),
        "intelligence": {
            **chance_corrected_accuracy(rows),
            # The public 231-item suite ships easy/hard/original only. The
            # official Intelligence also weights standard (0.28), judge
            # (0.28) and 308 sealed items; those tiers are not public, so
            # this is an unweighted per-item proxy over the visible 231.
            "proxy": "unweighted per-item chance-corrected accuracy, 231 public items",
            "chance_model": "uniform 1 / n_options per item; noul uses 0.5",
            "tiers_present": ["easy", "hard", "original"],
            "tiers_missing": ["standard", "judge", "sealed"],
            "official_weights_not_applied": {
                "easy": 0.14,
                "standard": 0.28,
                "judge": 0.28,
                "hard": 0.30,
            },
        },
        "by_tier": summarize_by(rows, lambda row: row.tier or "unknown", args.bins),
        "by_type": summarize_by(rows, lambda row: row.q_type, args.bins),
        "by_source": summarize_by(rows, lambda row: row.source, args.bins),
        "by_k_bucket": summarize_by(
            rows, lambda row: k_bucket(len(row.keys)), args.bins
        ),
        "schema": {
            "n_schemas": len(by_schema),
            "by_schema": top_schemas,
            "split": summarize_schema_split(rows, schema_split, args.bins),
        },
        "risk_coverage": risk_coverage(rows),
        "order_sensitivity": order,
        "k_sweep": sweep,
        "counts": {"items": len(items), "scored": len(rows), "failed": failed},
        "latency": {
            "mean_s": sum(latencies) / len(latencies) if latencies else 0.0,
            "median_s": latencies[len(latencies) // 2] if latencies else 0.0,
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nWrote {output}")

    overall = result["overall"]
    print(
        f"overall: n={overall.get('n', 0)} acc={overall.get('accuracy', 0):.4f} "
        f"ece={overall.get('ece', 0):.4f} brier={overall.get('brier', 0):.4f} "
        f"bss={overall.get('bss')}"
    )
    for name, metrics in result["by_type"].items():
        print(
            f"  {name:7s} n={metrics.get('n', 0):5d} acc={metrics.get('accuracy', 0):.4f} "
            f"ece={metrics.get('ece', 0):.4f}"
        )


if __name__ == "__main__":
    main()
