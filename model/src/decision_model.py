"""Decision model: frozen backbone + trainable typed heads.

Combines a pretrained LM backbone (frozen) with lightweight decision
heads (trainable) for typed-question scoring. Supports both causal
LM and encoder backbones.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .backbone import is_hybrid_model
from .choice_prompts import fit_option_prompts
from .gpu_config import GPUConfig, print_gpu_info
from .heads import (
    AttentionHead,
    ChoiceHead,
    CrossEncoderHead,
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
        lora_layers: int | None = None,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        lora_targets: tuple[str, ...] = ("q_proj", "v_proj"),
        choice_mode: str = "tower",
    ) -> None:
        super().__init__()
        if choice_mode not in ("tower", "cross_encoder"):
            raise ValueError(
                f"choice_mode={choice_mode!r} is not 'tower' or 'cross_encoder'"
            )
        self.choice_mode = choice_mode
        #: Items whose cross-encoder prompts had to shorten the state to keep
        #: the option inside max_length. Reported once, then counted.
        self.truncated_prompts = 0
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
        #
        # An *explicit* "off" is a different matter: the caller has asked for
        # the memory-for-speed trade, and overriding it silently made
        # --no-gradient-checkpointing a no-op whenever LoRA was on.
        explicit_off = (
            gpu_config.gradient_checkpointing_explicit
            and not gpu_config.gradient_checkpointing
        )
        needs_checkpointing = gpu_config.gradient_checkpointing or (
            lora_layers is not None and not explicit_off
        )
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
        elif lora_layers is not None and explicit_off:
            print(
                "  Gradient checkpointing: disabled by request with LoRA — "
                "every adapted block keeps its activations, so memory use is "
                "much higher",
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
        self.readout = LayerMixer(n_hidden + 1, readout, readout_n)
        if readout != "last":
            detail = f" (n={readout_n})" if readout == "mean_last_n" else ""
            print(
                f"  Readout: {readout}{detail} over {n_hidden + 1} hidden states",
                flush=True,
            )

        if mlp_layers > 0:
            self.projector = MLPProjector(
                self.hidden_size, mlp_dim, mlp_layers, dropout
            )
        else:
            self.projector = nn.Identity()

        if choice_mode == "cross_encoder":
            # One prompt per option, each carrying the state; the score is a
            # linear read-out of that prompt's pooled hidden state. Score and
            # noul keep the default path — only the choice slot changes.
            self.unified = None  # type: ignore[assignment]
            self.noul_head = NoulHead(self.hidden_size, rank=noul_rank, dropout=dropout)
            self.choice_head = CrossEncoderHead(self.hidden_size, dropout)
            self.score_head = ScoreHead(self.hidden_size, rank, dropout)
            params = sum(p.numel() for p in self.choice_head.parameters())
            print(
                f"  Choice head: cross-encoder (one prompt per option, "
                f"state in the prompt), {params:,} params",
                flush=True,
            )
        elif unified_head:
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

        self.backbone_requires_grad = any(
            p.requires_grad for p in self.backbone.parameters()
        )

        device = next(self.backbone.parameters()).device
        self.readout = self.readout.to(device)
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

    def _backbone_forward(self, inputs: dict) -> Any:
        """Run backbone forward pass with optional bf16 autocast.

        Args:
            inputs: Tokenized inputs dict (on device).

        Returns:
            Model outputs with hidden states.
        """
        fwd_kwargs = dict(**inputs, output_hidden_states=True)
        if not self.is_encoder:
            fwd_kwargs["use_cache"] = False
            # The projection to vocabulary is pure waste here: nothing in this
            # model reads ``.logits`` — the heads take ``hidden_states`` and the
            # prefix-cache path takes the cache. Qwen3's vocabulary is 151,669,
            # so one 16,384-token sequence materialises 4.6 GiB in bf16 before
            # any batching; sixteen of them is 74 GiB, which is how a run OOMed
            # while asking for a single 26.13 GiB allocation (~5.6 sequences).
            #
            # Only pass this when the backbone actually has a vocabulary head.
            # ``is_encoder`` answers "is this bidirectional", not "can this
            # forward project": ``load_qwen35_base`` returns
            # ``AutoModelForCausalLM(...).model``, which has no ``lm_head`` and
            # would raise on the kwarg. Hidden states and the KV cache are
            # unaffected by the slice.
            #
            # Do NOT add this to ``model/src/logit_scorer.py``: that module reads
            # ``ctx_out.logits[0, -1]`` and the shifted ``out.logits[:, :-1, :]``
            # over option tokens, so keeping one position would silently zero its
            # token scores.
            if getattr(self.backbone, "lm_head", None) is not None:
                fwd_kwargs["logits_to_keep"] = 1

        use_autocast = (
            self.gpu_config.bf16_compute
            and self.device.type == "cuda"
        )
        # LoRA adapters live inside the backbone, so their graph needs grad
        # enabled even though every pretrained weight stays frozen.  An
        # enclosing no_grad (evaluation) still wins.
        needs_grad = self.backbone_requires_grad and torch.is_grad_enabled()
        with torch.enable_grad() if needs_grad else torch.no_grad():
            if use_autocast:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return self.backbone(**fwd_kwargs)
            else:
                return self.backbone(**fwd_kwargs)

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
        outputs = self._backbone_forward(inputs)
        hidden = self.readout(outputs.hidden_states)
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
        outputs = self._backbone_forward(inputs)
        hidden = self.readout(outputs.hidden_states)
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
        if self.choice_mode == "cross_encoder":
            # Each option is scored from the whole prompt, so the state and the
            # option interact through the full depth of the backbone rather
            # than through a rank-limited attention. One batched forward for
            # all options; the softmax across them lives in the loss.
            #
            # The option is the suffix of the prompt and the tokenizer truncates
            # from the right, so a state that fills the window would drop the
            # option — silently, and only on the longest items. The state is
            # shortened instead, and a truncated prompt is reported once.
            prompts, dropped = fit_option_prompts(
                state,
                instructions,
                option_texts,
                self.max_length,
                encode=lambda text: self.tokenizer(text, add_special_tokens=False)["input_ids"],
                decode=lambda ids: self.tokenizer.decode(ids, skip_special_tokens=True),
            )
            if dropped:
                self.truncated_prompts += 1
                if self.truncated_prompts == 1:
                    warnings.warn(
                        "cross-encoder prompts exceeded max_length: dropped "
                        f"{dropped} leading state tokens so the option survives. "
                        "Long states are shortened, never the option.",
                        stacklevel=2,
                    )
            option_pooled = self._encode_text(prompts)
            return self.choice_head(option_pooled).unsqueeze(0)
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
