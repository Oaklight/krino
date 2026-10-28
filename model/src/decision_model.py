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
    OptionMarkerHead,
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

    ``option_marker`` selects a different option readout for choice and score
    questions.  The default readout encodes each option on its own and pools
    it to one vector, then ``AttentionHead`` cross-attends the pooled options
    to the context.  The option-marker readout instead appends each option to
    the same sequence as the state and instructions, puts a marker token right
    after the option, and scores the backbone hidden state at that marker.
    The softmax over all marker logits is still the option distribution, so
    the losses and evaluation code do not change.

    Many-candidate cost model (``option_marker`` only).  Let ``L`` be the
    context length, ``K`` the option count, ``B`` ``option_chunk_size``, and
    ``S`` the token length of one option plus its marker.  The context is
    encoded once into a KV cache; options are then appended in chunks of at
    most ``B`` options each, so the backbone runs ``1 + ceil(K / B)`` times:
    one prefix forward over ``L`` tokens and ``ceil(K / B)`` suffix forwards
    over about ``B * S`` tokens each.  Every suffix token attends to the
    cached prefix, so the extra attention work is
    ``O(K * S * L)`` per layer, but the prefix is never re-encoded.  The
    cache costs ``2 * n_layers * n_kv_heads * head_dim * L * dtype_bytes``;
    for Qwen3-0.6B (28 layers, 8 KV heads, 128 head dim, bf16) that is about
    112 KiB per context token (1.8 GiB at 16K tokens).  The suffix cache is
    cropped after every chunk, so it adds only ``B * S`` tokens at a time.
    Without the cache the same question would cost ``K`` full-context forward
    passes and ``K * L`` tokens instead of ``1 + ceil(K / B)`` forwards.
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
        option_marker: bool = False,
        option_chunk_size: int = 32,
        option_max_tokens: int = 64,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size or backbone.config.hidden_size
        self.is_encoder = _is_encoder_model(backbone)
        self.is_hybrid = is_hybrid_model(backbone.config)
        self.shared_attention = shared_attention
        self._unified_head = unified_head
        self.option_marker = option_marker
        self.option_chunk_size = max(1, option_chunk_size)
        self.option_max_tokens = max(1, option_max_tokens)
        if option_marker and unified_head:
            raise ValueError(
                "option_marker and unified_head are mutually exclusive: "
                "the marker readout already uses one head for choice and score"
            )
        if option_marker and self.is_encoder:
            raise ValueError(
                "option_marker is implemented for causal backbones; encoder "
                "backbones need a different sequence layout"
            )
        self.marker_token_id = None
        if option_marker:
            self.marker_token_id = (
                getattr(tokenizer, "eos_token_id", None)
                or getattr(tokenizer, "sep_token_id", None)
                or getattr(tokenizer, "pad_token_id", None)
            )
            if self.marker_token_id is None:
                raise ValueError(
                    "option_marker needs a tokenizer with an eos, sep or pad token "
                    "to use as the option marker"
                )

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
        if option_marker and lora_layers is not None:
            # HF turns use_cache off when checkpointing is enabled, and the
            # marker readout's whole cost argument depends on reusing the
            # context KV cache.  A checkpointed prefix pass would therefore
            # fall back to re-encoding the context once per chunk.  Keep the
            # cache and store the (short) suffix activations instead.
            needs_checkpointing = False
            print(
                "  Option-marker + LoRA: gradient checkpointing disabled "
                "(KV caching is incompatible with checkpointing)",
                flush=True,
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

        if option_marker:
            # Noul keeps the binary pooled head: it has no candidate set, so
            # a marker readout would add two full option sequences for no
            # benefit.  Choice and score share one marker scorer, which is
            # what shared_attention would have done for the attention head.
            self.unified = None  # type: ignore[assignment]
            self.noul_head = NoulHead(
                self.hidden_size, rank=noul_rank, dropout=dropout
            )
            self.marker_head = OptionMarkerHead(self.hidden_size, rank, dropout)
            self.choice_head = self.marker_head
            self.score_head = self.marker_head
            marker_params = sum(p.numel() for p in self.marker_head.parameters())
            print(
                f"  Option-marker readout: marker={self.marker_token_id}, "
                f"chunk={self.option_chunk_size}, "
                f"max_option_tokens={self.option_max_tokens}, "
                f"scorer={marker_params:,} params",
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

    def _backbone_forward(
        self,
        inputs: dict,
        *,
        output_hidden_states: bool = True,
        use_cache: bool | None = None,
    ) -> Any:
        """Run backbone forward pass with optional bf16 autocast.

        Args:
            inputs: Tokenized inputs dict (on device).
            output_hidden_states: Whether the backbone should return every
                layer's hidden states.  The marker readout's prefix pass only
                needs the KV cache, so it turns this off to avoid retaining
                ``n_layers`` copies of the context.
            use_cache: Causal-LM KV cache flag.  ``None`` keeps the historical
                behaviour (off); the marker readout sets it explicitly.

        Returns:
            Model outputs with hidden states.
        """
        fwd_kwargs = dict(**inputs, output_hidden_states=output_hidden_states)
        if not self.is_encoder:
            fwd_kwargs["use_cache"] = False if use_cache is None else use_cache

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

    def _tokenize_context(
        self, state: str, instructions: str, max_length: int
    ) -> dict[str, torch.Tensor]:
        """Tokenize state+instructions, keeping the instructions when truncating.

        The common path is the same string concatenation the other readouts
        use.  When it does not fit, dropping tokens from the right would drop
        the instructions first because they sit at the end of the string, so
        the fallback truncates the state from the left and keeps the full
        question.
        """
        combined = self.tokenizer(
            f"{state} {instructions}", return_tensors="pt", truncation=False
        )
        if combined["input_ids"].shape[1] <= max_length:
            attention_mask = combined.get(
                "attention_mask", torch.ones_like(combined["input_ids"])
            )
            return {
                "input_ids": combined["input_ids"].to(self.device),
                "attention_mask": attention_mask.to(self.device),
            }

        instruction_ids = self.tokenizer(
            " " + instructions, add_special_tokens=False
        )["input_ids"]
        state_ids = self.tokenizer(state, add_special_tokens=False)["input_ids"]
        if len(instruction_ids) >= max_length:
            kept = instruction_ids[:max_length]
        else:
            state_budget = max_length - len(instruction_ids)
            # ``state_ids[-0:]`` would return the whole list, so handle the
            # no-room-for-state case explicitly.
            kept = (
                state_ids[-state_budget:] if state_budget > 0 else []
            ) + instruction_ids
        input_ids = torch.tensor([kept], dtype=torch.long, device=self.device)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }

    def _crop_cache(self, cache: Any, n_tokens: int) -> None:
        """Remove the suffix K/V appended by the last chunk.

        transformers 5.x takes a negative count; older releases take the
        target length.  Trying the new form first keeps the fast path on the
        pinned version and still works on a 4.x install.
        """
        if n_tokens == 0:
            return
        try:
            cache.crop(-n_tokens)
        except (TypeError, ValueError):
            cache.crop(cache.get_seq_length() - n_tokens)

    def _tokenize_option_ids(self, option_texts: list[str]) -> list[list[int]]:
        """Tokenize options without padding or special tokens.

        Each option is capped at ``option_max_tokens`` so one pathological
        option cannot consume the whole sequence budget.  The marker token is
        appended separately, so the returned lists exclude it.
        """
        option_ids: list[list[int]] = []
        for text in option_texts:
            encoded = self.tokenizer(
                text or "",
                add_special_tokens=False,
                truncation=True,
                max_length=self.option_max_tokens,
            )
            option_ids.append(list(encoded["input_ids"]))
        return option_ids

    def _chunk_option_ids(self, option_ids: list[list[int]]) -> list[list[list[int]]]:
        """Split tokenized options into fixed-size chunks."""
        chunks: list[list[list[int]]] = []
        current: list[list[int]] = []
        for ids in option_ids:
            if len(current) >= self.option_chunk_size:
                chunks.append(current)
                current = []
            current.append(ids)
        if current:
            chunks.append(current)
        return chunks

    def _build_marker_suffix(
        self, chunk: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build one suffix sequence and the positions of its marker tokens.

        The suffix concatenates every option in the chunk followed by the
        marker token, so the backbone can score all options in the chunk with
        one forward pass over the cached prefix.
        """
        ids: list[int] = []
        positions: list[int] = []
        for option_ids in chunk:
            ids.extend(option_ids)
            ids.append(self.marker_token_id)
            positions.append(len(ids) - 1)
        return (
            torch.tensor([ids], dtype=torch.long, device=self.device),
            torch.tensor(positions, dtype=torch.long, device=self.device),
        )

    def _forward_options_no_cache(
        self,
        prefix_inputs: dict,
        chunks: list[list[list[int]]],
    ) -> torch.Tensor:
        """Fallback that re-encodes the context for each option chunk.

        Used when the backbone cannot return a KV cache (for example when a
        caller forces gradient checkpointing).  It still chunks the options,
        so memory stays bounded, but the prefix is paid once per chunk.
        """
        if not getattr(self, "_warned_no_cache", False):
            print(
                "  Warning: backbone returned no KV cache for the option-marker "
                "readout; re-encoding the context per chunk",
                flush=True,
            )
            self._warned_no_cache = True
        context_len = prefix_inputs["input_ids"].shape[1]
        all_logits: list[torch.Tensor] = []
        for chunk in chunks:
            suffix_ids, positions = self._build_marker_suffix(chunk)
            input_ids = torch.cat([prefix_inputs["input_ids"], suffix_ids], dim=1)
            attention_mask = torch.cat(
                [prefix_inputs["attention_mask"], torch.ones_like(suffix_ids)], dim=1
            )
            outputs = self._backbone_forward(
                {"input_ids": input_ids, "attention_mask": attention_mask}
            )
            hidden = self.projector(self.readout(outputs.hidden_states).float())
            marker_hidden = hidden[0, positions + context_len].unsqueeze(0)
            all_logits.append(self.marker_head(marker_hidden))
        return torch.cat(all_logits, dim=-1)

    def _forward_options_by_marker(
        self,
        state: str,
        instructions: str,
        option_texts: list[str],
    ) -> torch.Tensor:
        """Score every option at the marker token that follows it.

        Args:
            state: The content to decide about.
            instructions: The question text.
            option_texts: Candidate option descriptions.

        Returns:
            logits: [1, n_options] — one score per option, in input order.
        """
        if not option_texts:
            raise ValueError("option_marker readout needs at least one option")

        option_ids = self._tokenize_option_ids(option_texts)
        chunks = self._chunk_option_ids(option_ids)
        max_suffix = max(
            sum(len(ids) + 1 for ids in chunk) for chunk in chunks
        )
        if max_suffix >= self.max_length:
            raise ValueError(
                f"option_marker needs room for a suffix of {max_suffix} tokens "
                f"inside max_length={self.max_length}; lower --option-chunk-size "
                "or --option-max-tokens"
            )
        prefix_budget = self.max_length - max_suffix
        prefix_inputs = self._tokenize_context(state, instructions, prefix_budget)

        prefix_outputs = self._backbone_forward(
            prefix_inputs,
            output_hidden_states=False,
            use_cache=True,
        )
        cache = getattr(prefix_outputs, "past_key_values", None)
        if cache is None or not hasattr(cache, "crop"):
            return self._forward_options_no_cache(prefix_inputs, chunks)

        all_logits: list[torch.Tensor] = []
        for chunk in chunks:
            suffix_ids, positions = self._build_marker_suffix(chunk)
            attention_mask = torch.cat(
                [prefix_inputs["attention_mask"], torch.ones_like(suffix_ids)], dim=1
            )
            outputs = self._backbone_forward(
                {
                    "input_ids": suffix_ids,
                    "attention_mask": attention_mask,
                    "past_key_values": cache,
                }
            )
            hidden = self.projector(self.readout(outputs.hidden_states).float())
            marker_hidden = hidden[0, positions].unsqueeze(0)
            all_logits.append(self.marker_head(marker_hidden))
            # Qwen3 appends the suffix K/V to the cache even with
            # use_cache=False.  Drop them so every chunk starts from the same
            # prefix instead of seeing the previous chunk's options.
            self._crop_cache(cache, suffix_ids.shape[1])
        return torch.cat(all_logits, dim=-1)

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
        if self.option_marker:
            return self._forward_options_by_marker(state, instructions, option_texts)
        context_text = f"{state} {instructions}"
        context_hidden = self._encode_with_sequence(context_text)
        option_pooled = self._encode_text(option_texts)
        option_hidden = option_pooled.unsqueeze(0)
        return self.choice_head(context_hidden, option_hidden)

    def forward_score(
        self, state: str, instructions: str, level_texts: list[str]
    ) -> torch.Tensor:
        """Returns logits [1, n_levels] for score prediction."""
        if self.option_marker:
            return self._forward_options_by_marker(state, instructions, level_texts)
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
        """Load head parameters from a checkpoint.

        A ``strict=False`` load is deliberate: the backbone is loaded from the
        pretrained model, so its weights are absent from the checkpoint.  A
        missing or unexpected non-backbone key, however, usually means the
        checkpoint was trained with a different head architecture and part of
        its head silently did not load.  Report those keys instead of
        accepting a random head.
        """
        head_state = torch.load(Path(path), map_location=self.device, weights_only=True)
        result = self.load_state_dict(head_state, strict=False)
        missing = [k for k in result.missing_keys if not k.startswith("backbone.")]
        unexpected = [
            k for k in result.unexpected_keys if not k.startswith("backbone.")
        ]
        if missing or unexpected:
            print(
                f"  Warning: checkpoint {path} does not match the model's heads; "
                f"missing={missing[:5]}{'...' if len(missing) > 5 else ''} "
                f"unexpected={unexpected[:5]}{'...' if len(unexpected) > 5 else ''}",
                flush=True,
            )

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def frozen_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)
