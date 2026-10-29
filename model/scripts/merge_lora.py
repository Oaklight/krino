#!/usr/bin/env python3
"""Merge a trained LoRA checkpoint into its backbone and save a plain HF model.

Training adapts the frozen backbone with low-rank factors that live in the
checkpoint's ``best_heads.pt``; nothing else can read them. That makes two
questions unanswerable from the checkpoints alone, and both matter for goal
item 4:

* does fine-tuning preserve the backbone's own reranker readout, or does it
  destroy the capability the base model has? and
* is the JevBench deficit in the trained heads or in the adapted backbone?

Answering them needs the fine-tuned backbone as a normal HuggingFace model, so
the reranker's native yes/no scoring can be run on it. This script produces that
model and then checks itself: it re-loads the saved directory, rebuilds the
decision heads on top with no LoRA, and verifies the choice logits match the
LoRA model's on the same items. A merge that silently loses the adapter, or
applies it twice, fails that comparison instead of producing a plausible number.

Example:
    python model/scripts/merge_lora.py \\
        --config model/configs/r2_reranker0.6b.yaml \\
        --model Qwen/Qwen3-Reranker-0.6B \\
        --load-heads model/checkpoints/arch_06b_loraAll_weighted/best_heads.pt \\
        --lora-layers -1 --lora-rank 16 --readout weighted \\
        --output model/merged/arch_06b_loraAll_weighted
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model.src.gpu_config import GPUConfig, estimate_model_billions, print_gpu_info
from model.src.lora import merge_lora
from model.src.model_factory import build_decision_model

PROBES = [
    (
        "The customer wants a refund for a duplicate charge on the last invoice.",
        "Which intent does this express?",
        ["billing_question", "cancel_order", "track_order", "report_damage"],
    ),
    (
        "A short product review that is clearly positive about the item.",
        "How positive is this review?",
        ["very negative", "negative", "neutral", "positive", "very positive"],
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", default=None, help="Backbone name; defaults to the config")
    parser.add_argument("--load-heads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Directory for the merged model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lora-layers", type=int, default=-1)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-targets", default=None)
    parser.add_argument("--readout", default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Largest acceptable choice-logit gap between the LoRA model and the "
        "re-loaded merged model, in logit units",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    model_name = args.model or cfg.get("model", "Qwen/Qwen3-Reranker-0.6B")
    max_length = args.max_length or cfg.get("max_length")
    gpu_config = GPUConfig.auto_detect(estimate_model_billions(model_name))
    gpu_config = gpu_config.apply_overrides(max_length=max_length)
    print_gpu_info(gpu_config)
    targets = args.lora_targets or cfg.get("lora_targets", "q_proj,v_proj")

    def build(**overrides: object):
        """Build the architecture the checkpoint was saved against."""
        kwargs: dict[str, object] = {
            "model_name": model_name,
            "gpu_config": gpu_config,
            "device": args.device,
            "use_encoder": cfg.get("encoder", False),
            "qwen35_base": cfg.get("qwen35_base", False),
            "rank": cfg.get("rank", 64),
            "readout": args.readout or cfg.get("readout", "last"),
            "max_length": max_length,
            "lora_layers": cfg.get("lora_layers"),
            "lora_rank": cfg.get("lora_rank", 16),
            "lora_alpha": cfg.get("lora_alpha", 32),
            "lora_targets": tuple(str(targets).split(",")),
        }
        kwargs.update(overrides)
        return build_decision_model(**kwargs)

    print(f"Building {model_name} with LoRA, then loading {args.load_heads}", flush=True)
    model = build_decision_model(
        model_name=model_name,
        gpu_config=gpu_config,
        device=args.device,
        use_encoder=cfg.get("encoder", False),
        qwen35_base=cfg.get("qwen35_base", False),
        rank=cfg.get("rank", 64),
        readout=args.readout or cfg.get("readout", "last"),
        max_length=max_length,
        lora_layers=args.lora_layers,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_targets=tuple(str(targets).split(",")),
    )
    model.load_heads(args.load_heads)
    model.eval()

    merged_layers = merge_lora(model.backbone)
    print(f"Merged {merged_layers} LoRA layers", flush=True)
    if merged_layers == 0:
        print("ERROR: no LoRALinear modules were found to merge", file=sys.stderr)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(args.output)
    model.tokenizer.save_pretrained(args.output)
    print(f"Saved merged backbone to {args.output}", flush=True)

    # Self-check: rebuild from the saved directory with no LoRA at all, load the
    # same heads, and compare the choice logits against the LoRA model that was
    # doing the work a moment ago.
    print("Verifying the saved model reproduces the LoRA model", flush=True)
    # The point of the check is that the *saved directory* carries the merged
    # weights, so it has to be the thing loaded -- building from the original
    # backbone name would put the unadapted base back and compare the LoRA
    # model against itself with the adapter removed.
    reloaded = build(lora_layers=None, model_name=str(args.output))
    reloaded.load_heads(args.load_heads)
    reloaded.eval()
    with torch.no_grad():
        lo_a = model.forward_choice(*PROBES[0]).float()
        lo_b = reloaded.forward_choice(*PROBES[0]).float()
    reloaded_gap = (lo_a - lo_b).abs().max().item()
    print(f"  max |logit difference| = {reloaded_gap:.4f} (tolerance {args.tolerance})")
    if reloaded_gap > args.tolerance:
        print(
            "ERROR: the reloaded merged model does not reproduce the LoRA model",
            file=sys.stderr,
        )
        return 1
    print("merge verified", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
