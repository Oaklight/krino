"""Decision model: frozen backbone + trainable typed heads.

Combines a pretrained LM backbone (frozen) with lightweight decision
heads (trainable) for typed-question scoring. Supports both causal
LM and encoder backbones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .backbone import is_hybrid_model
from .gpu_config import GPUConfig, print_gpu_info
from .heads import (
    AttentionHead,
    ChoiceHead,
    LayerMixer,
    MLPProjector,
    NoulHead,
    ScoreHead,
    UnifiedHead,
)
from .lora import inject_lora


def _is_encoder_model(backbone: PreTrainedModel) -> bool:
    """Detect whether backbone is an encoder (bidirectional) or causal decoder."""
    return not getattr(backbone.config, "is_decoder", True)


class DecisionModel(nn.Module):
    """Frozen backbone + trainable decision heads.

    Supports both causal LM (Qwen, Llama, Gemma) and encoder (ModernBERT, BERT)
    backbones. Pooling strategy is detected automatically:
      - Causal: last-token pooling (last token sees full sequence)
      - Encoder: mean pooling (all tokens see all tokens)

    Three ways to attach the heads to the backbone, usable together:
      - ``readout``: which backbone layers feed the heads (see ``LayerMixer``).
      - ``lora_layers``: inject LoRA adapters into the backbone, either into
        every block or only the last N, so the backbone representation can
        adapt while the pretrained weights stay frozen.
      - ``mlp_layers``: an MLP projector between backbone and heads.

    A second backbone can be attached with ``second_backbone``.  Both encode
    the same token ids and their readout tensors are concatenated along the
    hidden dimension, then a learned projection maps the result back to
    ``hidden_size`` so the existing heads work unchanged.  ``fusion="sum"``
    is available when both backbones share a hidden size.
    """

    def __init__(
        self,
        backbone: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        hidden_size: int | None = None,
        rank: int = 64,
        rival_aware: bool = False,
        dropout: float = 0.1,
        mlp_layers: int = 0,
        mlp_dim: int | None = None,
        noul_rank: int | None = None,
        max_length: int | None = None,
        gpu_config: GPUConfig | None = None,
        shared_attention: bool = False,
        unified_head: bool = False,
        readout: str = "last",
        readout_n: int = 4,
        readout_layer: int | None = None,
        readout_pair: tuple[int, int] | None = None,
        readout_fusion: str = "concat",
        lora_layers: int | None = None,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        lora_targets: tuple[str, ...] = ("q_proj", "v_proj"),
        second_backbone: PreTrainedModel | None = None,
        second_readout: str = "last",
        second_readout_n: int = 4,
        second_readout_layer: int | None = None,
        second_readout_pair: tuple[int, int] | None = None,
        second_readout_fusion: str = "concat",
        second_lora_layers: int | None = None,
        fusion: str = "concat",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size or backbone.config.hidden_size
        self.is_encoder = _is_encoder_model(backbone)
        self.is_hybrid = is_hybrid_model(backbone.config)
        self.shared_attention = shared_attention
        self._unified_head = unified_head

        # GPU optimization config — auto-detect if not provided
        if gpu_config is None:
            gpu_config = GPUConfig.auto_detect()
        self.gpu_config = gpu_config

        # max_length priority: explicit arg > gpu_config > backbone native.
        # Both paths clamp to the backbone's positional-embedding limit: an
        # explicit request above it would overflow the position table at
        # forward time, which is harder to diagnose than a shorter sequence.
        backbone_max = min(
            getattr(backbone.config, "max_position_embeddings", 32768),
            32768,
        )
        if max_length is not None:
            self.max_length = min(max_length, backbone_max)
        else:
            self.max_length = min(gpu_config.max_length, backbone_max)

        # Enable gradient checkpointing on backbone if configured.  LoRA keeps
        # the backbone graph alive, so activations must be recomputed even on
        # backbones the auto-detect leaves uncheckpointed — a frozen backbone
        # needs no graph at all, but an adapted one does, and without this the
        # 0.6B model OOMs at its 16K max length.  The reentrant implementation
        # (the default) needs an input that requires grad; LoRA's adapters sit
        # inside the blocks, so it would silently produce no gradients.
        needs_checkpointing = gpu_config.gradient_checkpointing or lora_layers is not None
        if needs_checkpointing:
            if hasattr(backbone, "gradient_checkpointing_enable"):
                gc_kwargs: dict = {}
                if lora_layers is not None:
                    gc_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
                backbone.gradient_checkpointing_enable(**gc_kwargs)
                if lora_layers is not None and not gpu_config.gradient_checkpointing:
                    print(
                        "  Gradient checkpointing: forced on for LoRA "
                        "(auto-detect left it off)",
                        flush=True,
                    )

        print_gpu_info(gpu_config)
        print(f"  max_length={self.max_length} (effective)", flush=True)

        n_hidden = getattr(backbone.config, "num_hidden_layers", 0)
        if readout != "last" and n_hidden == 0:
            raise ValueError(
                f"readout={readout!r} needs backbone.config.num_hidden_layers, "
                "which this backbone does not define"
            )
        self.readout = LayerMixer(
            n_hidden + 1,
            readout,
            readout_n,
            layer=readout_layer,
            pair=readout_pair,
            fusion=readout_fusion,
        )
        # A two-layer concatenation is twice the backbone width; bring it back
        # so the heads and every other readout mode keep one input width.
        self.readout_project = (
            nn.Linear(self.hidden_size * self.readout.out_mult, self.hidden_size)
            if self.readout.out_mult > 1
            else nn.Identity()
        )
        if readout != "last":
            detail = ""
            if readout == "mean_last_n":
                detail = f" (n={readout_n})"
            elif readout == "layer":
                detail = f" (layer={readout_layer})"
            elif readout == "fuse_pair":
                detail = f" (layers={readout_pair}, {readout_fusion})"
            print(
                f"  Readout: {readout}{detail} over {n_hidden + 1} hidden states",
                flush=True,
            )

        # Optional second backbone.  Both encode the same ids; their readout
        # tensors are fused and projected back to ``hidden_size``.
        self.second_backbone = second_backbone
        self.second_hidden_size = 0
        self.second_readout = None
        self.second_readout_project = None
        self.fusion_project = None
        self.fusion = fusion
        self.second_backbone_requires_grad = False
        if second_backbone is not None:
            if fusion not in ("concat", "sum"):
                raise ValueError(f"fusion={fusion!r}; expected 'concat' or 'sum'")
            if _is_encoder_model(second_backbone) != self.is_encoder:
                raise ValueError(
                    "primary and second backbone must both be causal or both "
                    "be encoders"
                )
            n_hidden2 = getattr(second_backbone.config, "num_hidden_layers", 0)
            if second_readout != "last" and n_hidden2 == 0:
                raise ValueError(
                    f"second_readout={second_readout!r} needs "
                    "second_backbone.config.num_hidden_layers"
                )
            self.second_hidden_size = second_backbone.config.hidden_size
            self.second_readout = LayerMixer(
                n_hidden2 + 1,
                second_readout,
                second_readout_n,
                layer=second_readout_layer,
                pair=second_readout_pair,
                fusion=second_readout_fusion,
            )
            self.second_readout_project = (
                nn.Linear(
                    self.second_hidden_size * self.second_readout.out_mult,
                    self.second_hidden_size,
                )
                if self.second_readout.out_mult > 1
                else nn.Identity()
            )
            if fusion == "sum":
                if self.hidden_size != self.second_hidden_size:
                    raise ValueError(
                        "fusion='sum' needs equal hidden sizes, got "
                        f"{self.hidden_size} and {self.second_hidden_size}"
                    )
                self.fusion_project = nn.Identity()
            else:
                self.fusion_project = nn.Linear(
                    self.hidden_size + self.second_hidden_size, self.hidden_size
                )
            print(
                f"  Second backbone: hidden={self.second_hidden_size}, "
                f"readout={second_readout}, fusion={fusion} "
                f"({self.hidden_size} + {self.second_hidden_size} -> "
                f"{self.hidden_size})",
                flush=True,
            )

        if mlp_layers > 0:
            self.projector = MLPProjector(
                self.hidden_size, mlp_dim, mlp_layers, dropout
            )
        else:
            self.projector = nn.Identity()

        if unified_head:
            # Single AttentionHead for all question types.  Noul is handled
            # via "true"/"false" option encoding in forward_noul.
            self.unified = UnifiedHead(self.hidden_size, rank, rival_aware, dropout)
            # choice_head and score_head alias the unified head — same
            # forward(ctx, opt, mask) signature, so training code works as-is.
            self.choice_head = self.unified
            self.score_head = self.unified
            # No separate noul_head — noul is handled in forward_noul.
            self.noul_head = None  # type: ignore[assignment]
            noul_saved = sum(
                p.numel()
                for p in NoulHead(self.hidden_size, rank=noul_rank, dropout=dropout).parameters()
            )
            attn_saved = sum(
                p.numel()
                for p in AttentionHead(self.hidden_size, rank, rival_aware=False, dropout=dropout).parameters()
            )
            print(
                f"  Unified head: single AttentionHead for all types "
                f"(noul via \"true\"/\"false\" options, "
                f"saved {noul_saved + attn_saved:,} params vs separate heads)",
                flush=True,
            )
        elif shared_attention:
            self.unified = None  # type: ignore[assignment]
            self.noul_head = NoulHead(self.hidden_size, rank=noul_rank, dropout=dropout)
            shared_attn = AttentionHead(self.hidden_size, rank, rival_aware, dropout)
            self.choice_head = ChoiceHead(
                self.hidden_size, rank, rival_aware, dropout, attention=shared_attn
            )
            self.score_head = ScoreHead(
                self.hidden_size, rank, dropout, attention=shared_attn
            )
            saved = sum(p.numel() for p in AttentionHead(self.hidden_size, rank, rival_aware=False, dropout=dropout).parameters())
            print(
                f"  Shared attention: choice+score share AttentionHead (saved {saved:,} params)",
                flush=True,
            )
        else:
            self.unified = None  # type: ignore[assignment]
            self.noul_head = NoulHead(self.hidden_size, rank=noul_rank, dropout=dropout)
            self.choice_head = ChoiceHead(self.hidden_size, rank, rival_aware, dropout)
            self.score_head = ScoreHead(self.hidden_size, rank, dropout)

        for param in self.backbone.parameters():
            param.requires_grad_(False)

        # Inject LoRA after the freeze pass — the adapters are the one part
        # of the backbone that must stay trainable.  A negative value means
        # "every block"; a positive one means "the last N blocks".
        self.lora_layers = lora_layers
        if lora_layers is not None:
            adapted = inject_lora(
                self.backbone,
                targets=lora_targets,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                last_n_layers=lora_layers if lora_layers > 0 else None,
            )
            n_lora = sum(
                p.numel() for p in self.backbone.parameters() if p.requires_grad
            )
            scope = f"last {lora_layers} blocks" if lora_layers > 0 else "all blocks"
            print(
                f"  LoRA: {adapted} layers ({scope}, r={lora_rank}, "
                f"alpha={lora_alpha}, targets={','.join(lora_targets)}), "
                f"{n_lora:,} trainable params in backbone",
                flush=True,
            )

        # The second backbone is frozen the same way; LoRA on it is opt-in
        # and independent of the primary backbone's LoRA scope.
        if self.second_backbone is not None:
            for param in self.second_backbone.parameters():
                param.requires_grad_(False)
            if second_lora_layers is not None:
                adapted2 = inject_lora(
                    self.second_backbone,
                    targets=lora_targets,
                    rank=lora_rank,
                    alpha=lora_alpha,
                    dropout=lora_dropout,
                    last_n_layers=second_lora_layers if second_lora_layers > 0 else None,
                )
                scope2 = (
                    f"last {second_lora_layers} blocks"
                    if second_lora_layers > 0
                    else "all blocks"
                )
                print(
                    f"  Second-backbone LoRA: {adapted2} layers ({scope2}), "
                    f"r={lora_rank}, targets={','.join(lora_targets)}",
                    flush=True,
                )
            self.second_backbone_requires_grad = any(
                p.requires_grad for p in self.second_backbone.parameters()
            )

        self.backbone_requires_grad = any(
            p.requires_grad for p in self.backbone.parameters()
        )

        device = next(self.backbone.parameters()).device
        self.readout = self.readout.to(device)
        self.readout_project = self.readout_project.to(device).float()
        if self.second_readout is not None:
            self.second_readout = self.second_readout.to(device)
        if self.second_readout_project is not None:
            self.second_readout_project = self.second_readout_project.to(device).float()
        if self.fusion_project is not None:
            self.fusion_project = self.fusion_project.to(device).float()
        self.projector = self.projector.to(device).float()
        if self.noul_head is not None:
            self.noul_head = self.noul_head.to(device).float()
        if self.unified is not None:
            self.unified = self.unified.to(device).float()
        self.choice_head = self.choice_head.to(device).float()
        self.score_head = self.score_head.to(device).float()

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    def _run_backbone(
        self,
        backbone: PreTrainedModel,
        inputs: dict,
        needs_grad: bool,
    ) -> Any:
        """Run one backbone forward pass with optional bf16 autocast.

        Args:
            backbone: The backbone to run (primary or second).
            inputs: Tokenized inputs dict (on device).
            needs_grad: Whether the backbone graph must be kept alive.

        Returns:
            Model outputs with hidden states.
        """
        fwd_kwargs = dict(**inputs, output_hidden_states=True)
        if not self.is_encoder:
            fwd_kwargs["use_cache"] = False

        use_autocast = (
            self.gpu_config.bf16_compute
            and self.device.type == "cuda"
        )
        # LoRA adapters live inside the backbone, so their graph needs grad
        # enabled even though every pretrained weight stays frozen.  An
        # enclosing no_grad (evaluation) still wins.
        needs = needs_grad and torch.is_grad_enabled()
        with torch.enable_grad() if needs else torch.no_grad():
            if use_autocast:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return backbone(**fwd_kwargs)
            else:
                return backbone(**fwd_kwargs)

    def _backbone_forward(self, inputs: dict) -> Any:
        """Run the primary backbone forward pass."""
        return self._run_backbone(
            self.backbone, inputs, self.backbone_requires_grad
        )

    def _hidden_sequence(self, inputs: dict) -> torch.Tensor:
        """Backbone readout fused across both backbones, [batch, seq, hidden].

        The readout layers return bf16 under autocast; the projections and
        the heads are fp32, so upcast before either projection runs.
        """
        outputs = self._run_backbone(
            self.backbone, inputs, self.backbone_requires_grad
        )
        hidden = self.readout_project(self.readout(outputs.hidden_states).float())
        if self.second_backbone is None:
            return hidden
        outputs2 = self._run_backbone(
            self.second_backbone, inputs, self.second_backbone_requires_grad
        )
        hidden2 = self.second_readout_project(
            self.second_readout(outputs2.hidden_states).float()
        )
        if self.fusion == "sum":
            return self.fusion_project(hidden + hidden2)
        return self.fusion_project(torch.cat([hidden, hidden2], dim=-1))

    def _encode_text(
        self, text: str | list[str], max_length: int | None = None
    ) -> torch.Tensor:
        """Encode text and return pooled hidden state per sequence.

        Causal LM: last-token pooling (last token attended to full sequence).
        Encoder: mean pooling (all tokens see all tokens bidirectionally).
        """
        max_length = max_length or self.max_length
        if isinstance(text, str):
            text = [text]
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(self.device)
        hidden = self._hidden_sequence(inputs)
        if self.is_encoder:
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        else:
            seq_lengths = inputs["attention_mask"].sum(dim=1) - 1
            pooled = hidden[
                torch.arange(hidden.size(0), device=hidden.device), seq_lengths
            ]
        return self.projector(pooled.float())

    def _encode_with_sequence(self, text: str, max_length: int | None = None) -> torch.Tensor:
        """Encode text and return full sequence hidden states [1, seq_len, hidden]."""
        max_length = max_length or self.max_length
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        ).to(self.device)
        hidden = self._hidden_sequence(inputs)
        return self.projector(hidden.float())

    def forward_noul(self, state: str, instructions: str) -> torch.Tensor:
        """Returns logit [1, 1] for noul prediction.

        When unified_head is enabled, encodes "false" and "true" as two
        options and uses cross-attention.  The returned logit is the
        difference (true - false) so that sigmoid gives P(true), which is
        mathematically equivalent to softmax over the two options.
        """
        if self.unified is not None:
            context_text = f"{state} {instructions}"
            context_hidden = self._encode_with_sequence(context_text)
            option_pooled = self._encode_text(["false", "true"])
            option_hidden = option_pooled.unsqueeze(0)  # [1, 2, hidden]
            return self.unified.noul_logit(context_hidden, option_hidden)
        text = f"{state} {instructions}"
        pooled = self._encode_text(text)
        return self.noul_head(pooled)

    def forward_choice(
        self, state: str, instructions: str, option_texts: list[str]
    ) -> torch.Tensor:
        """Returns logits [1, n_options] for choice prediction."""
        context_text = f"{state} {instructions}"
        context_hidden = self._encode_with_sequence(context_text)
        option_pooled = self._encode_text(option_texts)
        option_hidden = option_pooled.unsqueeze(0)
        return self.choice_head(context_hidden, option_hidden)

    def forward_score(
        self, state: str, instructions: str, level_texts: list[str]
    ) -> torch.Tensor:
        """Returns logits [1, n_levels] for score prediction."""
        context_text = f"{state} {instructions}"
        context_hidden = self._encode_with_sequence(context_text)
        level_pooled = self._encode_text(level_texts)
        level_hidden = level_pooled.unsqueeze(0)
        return self.score_head(context_hidden, level_hidden)

    def predict(self, state: str, question: dict[str, Any]) -> dict[str, Any]:
        """Single-question inference for eval suite compatibility."""
        q_type = question.get("type", "noul")
        instructions = question.get("instructions", "")

        if q_type == "noul":
            logit = self.forward_noul(state, instructions)
            noul = torch.sigmoid(logit).item()
            return {"type": "noul", "noul": round(max(0.01, min(0.99, noul)), 2)}

        elif q_type == "choice":
            criteria = question["criteria"]
            keys = list(criteria.keys())
            option_texts = [v or k for k, v in criteria.items()]
            logits = self.forward_choice(state, instructions, option_texts)
            probs = torch.softmax(logits, dim=-1)[0].tolist()
            prob_dict = {k: round(p, 2) for k, p in zip(keys, probs)}
            return {
                "type": "choice",
                "choice": max(prob_dict, key=prob_dict.get),
                "probabilities": prob_dict,
            }

        elif q_type == "score":
            criteria = question["criteria"]
            logits = self.forward_score(state, instructions, criteria)
            probs = torch.softmax(logits, dim=-1)[0].tolist()
            prob_dict = {str(i): round(p, 2) for i, p in enumerate(probs)}
            legend = {str(i): desc for i, desc in enumerate(criteria)}
            score_val = sum(i * p for i, p in enumerate(probs))
            return {
                "type": "score",
                "score": round(score_val, 2),
                "probabilities": prob_dict,
                "legend": legend,
            }

        raise ValueError(f"Unknown question type: {q_type}")

    def save_heads(self, path: Path | str) -> None:
        """Save the trainable parameters to a checkpoint.

        Frozen backbone weights are excluded.  LoRA adapters are included even
        though their names sit under ``backbone.`` — without them a LoRA
        checkpoint could not be reloaded.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        trainable = {name for name, p in self.named_parameters() if p.requires_grad}
        head_state = {
            k: v
            for k, v in self.state_dict().items()
            if "backbone" not in k or k in trainable
        }
        torch.save(head_state, path)

    def load_heads(self, path: Path | str) -> None:
        """Load head parameters from a checkpoint."""
        head_state = torch.load(Path(path), map_location=self.device, weights_only=True)
        self.load_state_dict(head_state, strict=False)

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def frozen_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)
