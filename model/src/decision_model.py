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
from .shape_buckets import (
    PREFIX_BUCKETS,
    bucket_length,
    bucket_rows,
    left_pad,
)
from .heads import (
    AttentionHead,
    ChoiceHead,
    CrossEncoderHead,
    LayerMixer,
    MLPProjector,
    NoulHead,
    OptionMarkerHead,
    ScoreHead,
    UnifiedHead,
)
from .lora import inject_lora


_CAUSAL_CLASS_SUFFIXES = ("ForCausalLM", "LMHeadModel", "ForConditionalGeneration")


def _is_encoder_model(backbone: PreTrainedModel) -> bool:
    """Detect whether backbone is an encoder (bidirectional) or causal decoder.

    ``config.is_decoder`` is unreliable: HuggingFace causal configs such as
    Qwen3 and ModernBERT both leave it unset, so the old ``not is_decoder``
    test classified every ModernBERT reranker as causal.  The class or
    declared architecture name is the discriminating signal, because the
    causal loaders return a ``*ForCausalLM`` wrapper while the encoder loader
    returns a bare ``*Model``.
    """
    config = backbone.config
    if getattr(config, "is_decoder", False) is True:
        return False
    # Qwen3.5 text towers are causal decoders even when the nested text
    # config carries no ``architectures`` entry (the 4B family exposes only
    # ``Qwen3_5TextModel``, which has no causal suffix).  GDN hybrids are
    # decoder-only by construction.
    if is_hybrid_model(config) or str(
        getattr(config, "model_type", "")
    ).startswith("qwen3_5"):
        return False
    names = list(getattr(config, "architectures", None) or ())
    names.append(type(backbone).__name__)
    for name in names:
        if name.endswith(_CAUSAL_CLASS_SUFFIXES):
            return False
    return True


def _noul_option_texts(criteria: Any) -> list[str]:
    """Option texts for noul in marker mode: false first, true second.

    Criteria may use true/false or yes/no keys; either shape maps onto the
    same two-slot order that ``forward_noul`` expects.  A bare question with
    no criteria keeps the literal words.
    """
    if not isinstance(criteria, dict):
        criteria = {}
    false_text = criteria.get("false") or criteria.get("no") or "No"
    true_text = criteria.get("true") or criteria.get("yes") or "Yes"
    return [str(false_text), str(true_text)]


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

    ``option_marker`` selects a different option readout for choice and score
    questions.  The default readout encodes each option on its own and pools
    it to one vector, then ``AttentionHead`` cross-attends the pooled options
    to the context.  The option-marker readout instead appends each option to
    the same sequence as the state and instructions, puts a marker token right
    after the option, and scores the backbone hidden state at that marker.
    The softmax over all marker logits is still the option distribution, so
    the losses and evaluation code do not change.

    Exactness note: options are concatenated into one sequence per chunk, so
    within a chunk each option attends to the other options' text.  The score
    of an option then depends on its chunk-mates and on ``K``, which a
    reranker-style decision should not allow.  ``option_chunk_size=1`` makes
    every option its own sequence and is the only exact setting; larger
    chunks trade that invariance for fewer backbone forwards.

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

    The existing pooled readout is not a per-option loop: it encodes the
    context and then the whole option batch in two backbone forwards, so its
    forward count is 2 for any ``K``.  The marker readout uses
    ``1 + ceil(K / B)`` forwards instead, and every option token attends to
    the cached prefix in every layer.  That attention adds about
    ``4 * n_layers * hidden * K * S * L`` FLOPs.  The marker readout is
    therefore a representation and memory-bounding change, not a FLOP
    reduction: the ``K`` full-context passes it avoids are the naive
    per-option alternative, which the pooled batch never paid.  On
    launch-bound hardware the extra forwards can dominate wall-clock.
    Treat the counts above as a FLOP and memory model, not a latency
    guarantee, and measure both.
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
        option_marker: bool = False,
        noul_marker: bool = False,
        option_chunk_size: int = 32,
        option_max_tokens: int = 64,
        option_batch: int = 1,
        train_safe_buckets: bool = False,
        choice_mode: str = "tower",
        answerable_head: bool = False,
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
        self.option_marker = option_marker
        if noul_marker and not option_marker:
            raise ValueError("noul_marker needs option_marker=True")
        self.noul_marker = bool(noul_marker)
        self.option_chunk_size = max(1, option_chunk_size)
        self.option_max_tokens = max(1, option_max_tokens)
        self.option_batch = max(1, option_batch)
        #: Padding-only bucket variant for training: keeps the semantic input
        #: (context budget and reserve) identical, but pads the option block
        #: to a multiple of 8 and the row count to a power-of-two bucket.
        self.train_safe_buckets = bool(train_safe_buckets)
        if option_marker and unified_head:
            raise ValueError(
                "option_marker and unified_head are mutually exclusive: "
                "the marker readout already uses one head for choice and score"
            )
        if option_marker and second_backbone is not None:
            raise ValueError(
                "option_marker scores a single sequence, so it does not "
                "support a second backbone"
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

        if option_marker:
            # Noul keeps the binary pooled head by default: it has no
            # candidate set, so a marker readout would add two option
            # sequences per item.  ``noul_marker`` instead scores the
            # explicit "false"/"true" options with the same marker scorer
            # choice and score use, so all three types share one readout and
            # noul sees its options in context.
            self.unified = None  # type: ignore[assignment]
            self.noul_head = (
                None
                if noul_marker
                else NoulHead(self.hidden_size, rank=noul_rank, dropout=dropout)
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
            )
        elif choice_mode == "cross_encoder":
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

        # The answerable head is a Bernoulli decision over the whole item, not
        # over an option set, so it reads the pooled context like noul and is
        # independent of the choice/score readout.  At inference it combines
        # with the option distribution as a mixture, never as an extra option:
        # p(abstain) = 1 - sigmoid(logit), p(option_i) = sigmoid(logit) * p_i.
        self.answerable_head = (
            NoulHead(self.hidden_size, rank=noul_rank, dropout=dropout)
            if answerable_head
            else None
        )
        if self.answerable_head is not None:
            ans_params = sum(p.numel() for p in self.answerable_head.parameters())
            print(
                f"  Answerable head: Bernoulli over the pooled context, "
                f"{ans_params:,} params",
                flush=True,
            )

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
            # An adapted second backbone needs its graph kept alive, so it
            # needs checkpointing for the same reason the primary one does.
            if gpu_config.gradient_checkpointing or second_lora_layers is not None:
                if hasattr(self.second_backbone, "gradient_checkpointing_enable"):
                    gc_kwargs2: dict = {}
                    if second_lora_layers is not None:
                        gc_kwargs2["gradient_checkpointing_kwargs"] = {
                            "use_reentrant": False
                        }
                    self.second_backbone.gradient_checkpointing_enable(**gc_kwargs2)
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
        if self.answerable_head is not None:
            self.answerable_head = self.answerable_head.to(device).float()

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    def _run_backbone(
        self,
        backbone: PreTrainedModel,
        inputs: dict,
        needs_grad: bool,
        *,
        output_hidden_states: bool = True,
        use_cache: bool | None = None,
    ) -> Any:
        """Run one backbone forward pass with optional bf16 autocast.

        Args:
            backbone: The backbone to run (primary or second).
            inputs: Tokenized inputs dict (on device).
            needs_grad: Whether the backbone graph must be kept alive.
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
        needs = needs_grad and torch.is_grad_enabled()
        with torch.enable_grad() if needs else torch.no_grad():
            if use_autocast:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return backbone(**fwd_kwargs)
            else:
                return backbone(**fwd_kwargs)

    def _backbone_forward(
        self,
        inputs: dict,
        *,
        output_hidden_states: bool = True,
        use_cache: bool | None = None,
    ) -> Any:
        """Run the primary backbone forward pass."""
        return self._run_backbone(
            self.backbone,
            inputs,
            self.backbone_requires_grad,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
        )

    def _hidden_sequence(self, inputs: dict) -> torch.Tensor:
        """Backbone readout fused across both backbones, [batch, seq, hidden].

        The readout layers return bf16 under autocast; the projections and
        the heads are fp32, so upcast before either projection runs.
        """
        outputs = self._backbone_forward(inputs)
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

    @torch.inference_mode()
    def warmup_prefix_buckets(self) -> None:
        """Run one dummy forward per shape bucket to load its CUDA kernels.

        A shape's first forward costs 0.4-1.3 s at 4B, and the cost is per
        shape rather than per request.  Paying it at construction keeps every
        request at the steady-state latency.  The dummy masks are all ones:
        the kernel shape depends on tensor dimensions, not values.
        """
        device = self.device
        prefix_buckets = [b for b in PREFIX_BUCKETS if b <= self.max_length]
        width = 64
        rows = min(8, max(1, self.option_batch))
        for length in prefix_buckets:
            # An exact-fit request has no padding and takes SDPA's
            # all-visible path; a shorter one carries masked pad positions
            # and takes the masked path.  They load different kernels, so
            # warm both.  The pooled readout pads on the right, the marker
            # readout on the left.
            for padded in (False, True):
                pooled_inputs = {
                    "input_ids": torch.zeros(
                        (1, length), dtype=torch.long, device=device
                    ),
                    "attention_mask": torch.ones(
                        (1, length), dtype=torch.long, device=device
                    ),
                    "position_ids": torch.arange(length, device=device).unsqueeze(0),
                }
                if padded:
                    pooled_inputs["attention_mask"][0, -1] = 0
                    pooled_inputs["position_ids"][0, -1] = 0
                pooled_hidden = self._hidden_sequence(dict(pooled_inputs))
                mask = pooled_inputs["attention_mask"]
                if self.is_encoder:
                    weights = mask.unsqueeze(-1).float()
                    pooled = (pooled_hidden * weights).sum(1) / weights.sum(
                        1
                    ).clamp(min=1e-9)
                else:
                    lengths = mask.sum(dim=1) - 1
                    pooled = pooled_hidden[
                        torch.arange(pooled_hidden.size(0), device=device), lengths
                    ]
                pooled = self.projector(pooled.float())
                if self.noul_head is not None:
                    self.noul_head(pooled)
                if self.answerable_head is not None:
                    self.answerable_head(pooled)
                if not self.option_marker:
                    continue
                marker_inputs = {
                    "input_ids": torch.zeros(
                        (1, length), dtype=torch.long, device=device
                    ),
                    "attention_mask": torch.ones(
                        (1, length), dtype=torch.long, device=device
                    ),
                    "position_ids": torch.arange(length, device=device).unsqueeze(0),
                }
                if padded:
                    marker_inputs["attention_mask"][0, 0] = 0
                outputs = self._backbone_forward(
                    marker_inputs, output_hidden_states=False, use_cache=True
                )
                cache = getattr(outputs, "past_key_values", None)
                if cache is None:
                    continue
                if rows > 1:
                    cache.batch_repeat_interleave(rows)
                suffix = {
                    "input_ids": torch.zeros(
                        (rows, width), dtype=torch.long, device=device
                    ),
                    "attention_mask": torch.ones(
                        (rows, length + width), dtype=torch.long, device=device
                    ),
                    "position_ids": torch.arange(
                        length, length + width, device=device
                    ).unsqueeze(0).repeat(rows, 1),
                    "past_key_values": cache,
                }
                if padded:
                    suffix["attention_mask"][:, 0] = 0
                out = self._backbone_forward(suffix)
                hidden = self.projector(
                    self.readout_project(self.readout(out.hidden_states).float())
                )
                picked = hidden[torch.arange(rows, device=device), 0]
                self.marker_head(picked.unsqueeze(0))
                self._crop_cache(cache, width)
                del cache, outputs, out, hidden, picked

    def _pad_to_bucket(self, inputs: dict, max_length: int) -> dict:
        """Right-pad a tokenized batch to a fixed length bucket.

        A new sequence length costs a 0.4-1.3 s first kernel load on H100;
        an evaluation that walks items with distinct lengths pays it once per
        item.  Right padding keeps the last real token at index
        ``real_length - 1``, which is exactly the index the causal pooling
        below reads, and the padded positions are masked out.
        """
        ids = inputs["input_ids"]
        mask = inputs.get("attention_mask", torch.ones_like(ids))
        real = ids.shape[1]
        target = bucket_length(real, PREFIX_BUCKETS, limit=max_length)
        if target <= real:
            return inputs
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", None) or 0
        pad = target - real
        padded_ids = torch.cat(
            [
                ids,
                torch.full(
                    (ids.shape[0], pad),
                    pad_id,
                    dtype=ids.dtype,
                    device=ids.device,
                ),
            ],
            dim=1,
        )
        padded_mask = torch.cat(
            [
                mask,
                torch.zeros(
                    (mask.shape[0], pad), dtype=mask.dtype, device=mask.device
                ),
            ],
            dim=1,
        )
        out = {"input_ids": padded_ids, "attention_mask": padded_mask}
        if not self.is_encoder:
            # Real tokens keep positions 0..L-1; pads sit at 0 and are masked.
            out["position_ids"] = (padded_mask.cumsum(-1) - 1).clamp(min=0) * padded_mask
        return out

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
        inputs = self._pad_to_bucket(inputs, max_length)
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
        # JevBench hard items carry a structured dict state; the concatenation
        # above stringifies it, so the fallback must do the same or the
        # tokenizer raises on a non-str input.
        state_ids = self.tokenizer(str(state), add_special_tokens=False)["input_ids"]
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
        *,
        warn: bool = True,
    ) -> torch.Tensor:
        """Fallback that re-encodes the context for each option chunk.

        Used when the backbone cannot return a KV cache (for example when a
        caller forces gradient checkpointing).  It still chunks the options,
        so memory stays bounded, but the prefix is paid once per chunk.
        """
        if warn and not getattr(self, "_warned_no_cache", False):
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
            hidden = self.projector(
                self.readout_project(self.readout(outputs.hidden_states).float())
            )
            marker_hidden = hidden[0, positions + context_len].unsqueeze(0)
            all_logits.append(self.marker_head(marker_hidden))
        return torch.cat(all_logits, dim=-1)

    def _forward_options_batched(
        self,
        state: str,
        instructions: str,
        option_texts: list[str],
        *,
        group_size: int,
    ) -> torch.Tensor:
        """Score options in parallel along the batch dimension.

        The sequential marker path runs one suffix forward per option, each
        carrying a single query token, so the GPU is mostly idle.  This path
        expands the prefix KV cache to ``group_size`` rows and scores that
        many options in one forward.  Every row carries the shared prefix plus
        exactly one ``option + marker`` block, so an option still sees only
        the prefix and itself -- the property that forces chunk=1 in the
        concatenating path.
        """
        option_ids = self._tokenize_option_ids(option_texts)
        blocks = [ids + [self.marker_token_id] for ids in option_ids]
        widest = max(len(block) for block in blocks)
        if widest >= self.max_length:
            raise ValueError(
                f"option_marker needs room for a suffix of {widest} tokens "
                f"inside max_length={self.max_length}; lower --option-max-tokens"
            )
        # Fixed shapes, for the same reason as the base reranker: a new
        # tensor shape costs a 0.4-1.3 s first kernel load on H100.  The
        # option block is a fixed 64 tokens (65 for a maximum-length option)
        # and the context is tokenized against that reserve, so every long
        # item lands on the same 1984-token prefix bucket instead of an
        # exact length.  Every padded position is masked out.
        #
        # ``train_safe_buckets`` keeps the same context budget and reserve,
        # so the tokens the model sees are identical; it only shrinks the
        # padded suffix to a multiple of 8.
        reserve = max(64, widest)
        width = ((widest + 7) // 8) * 8 if self.train_safe_buckets else reserve
        prefix_inputs = self._tokenize_context(
            state, instructions, self.max_length - reserve
        )
        real_prefix_len = prefix_inputs["input_ids"].shape[1]
        prefix_bucket = bucket_length(
            real_prefix_len, PREFIX_BUCKETS, limit=self.max_length - reserve
        )
        if prefix_bucket + width > self.max_length:
            prefix_bucket, width = real_prefix_len, widest
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", None) or 0
        padded_ids, pad = left_pad(
            prefix_inputs["input_ids"][0].tolist(), prefix_bucket, pad_id
        )
        prefix_inputs = {
            "input_ids": torch.tensor(
                [padded_ids], dtype=torch.long, device=self.device
            ),
            "attention_mask": torch.cat(
                [
                    torch.zeros((1, pad), dtype=torch.long, device=self.device),
                    torch.ones(
                        (1, real_prefix_len), dtype=torch.long, device=self.device
                    ),
                ],
                dim=1,
            ),
            "position_ids": torch.cat(
                [
                    torch.zeros((1, pad), dtype=torch.long, device=self.device),
                    torch.arange(real_prefix_len, device=self.device).unsqueeze(0),
                ],
                dim=1,
            ),
        }

        prefix_outputs = self._backbone_forward(
            prefix_inputs,
            output_hidden_states=False,
            use_cache=True,
        )
        cache = getattr(prefix_outputs, "past_key_values", None)
        if cache is None or not hasattr(cache, "batch_repeat_interleave"):
            return self._forward_options_no_cache(
                prefix_inputs, self._chunk_option_ids(option_ids)
            )

        # Eight rows, matching the base path: at long context the suffix
        # attention dominates and 16 rows cost twice the 8-row time.  The
        # training-safe variant buckets the row count to the actual option
        # count, so small K does not pay for eight rows.
        rows = (
            bucket_rows(min(8, len(blocks)))
            if self.train_safe_buckets
            else min(8, group_size)
        )
        group_size = rows
        if rows > 1:
            cache.batch_repeat_interleave(rows)

        all_logits: list[torch.Tensor] = []
        for start in range(0, len(blocks), group_size):
            group = blocks[start : start + group_size]
            padded = group + [group[-1]] * (rows - len(group))
            input_ids = torch.zeros(
                (rows, width), dtype=torch.long, device=self.device
            )
            attention_mask = torch.zeros(
                (rows, prefix_bucket + width),
                dtype=torch.long,
                device=self.device,
            )
            attention_mask[:, :prefix_bucket] = prefix_inputs["attention_mask"]
            position_ids = torch.full(
                (rows, width), real_prefix_len, dtype=torch.long, device=self.device
            )
            marker_index = torch.zeros(rows, dtype=torch.long, device=self.device)
            for row, block in enumerate(padded):
                n = len(block)
                input_ids[row, :n] = torch.tensor(block, device=self.device)
                attention_mask[row, prefix_bucket : prefix_bucket + n] = 1
                position_ids[row, :n] = torch.arange(
                    real_prefix_len, real_prefix_len + n, device=self.device
                )
                marker_index[row] = n - 1
            outputs = self._backbone_forward(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "past_key_values": cache,
                }
            )
            hidden = self.projector(
                self.readout_project(self.readout(outputs.hidden_states).float())
            )
            picked = hidden[torch.arange(rows, device=self.device), marker_index]
            all_logits.append(self.marker_head(picked.unsqueeze(0)))
            self._crop_cache(cache, width)
        return torch.cat(all_logits, dim=-1)[:, : len(blocks)]

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
        if self.option_batch > 1:
            return self._forward_options_batched(
                state, instructions, option_texts, group_size=self.option_batch
            )

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

        if self.is_encoder:
            # Encoders have no KV cache; re-encoding the prefix per chunk is
            # the expected path, not a fallback worth warning about.
            return self._forward_options_no_cache(prefix_inputs, chunks, warn=False)
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
            hidden = self.projector(
                self.readout_project(self.readout(outputs.hidden_states).float())
            )
            marker_hidden = hidden[0, positions].unsqueeze(0)
            all_logits.append(self.marker_head(marker_hidden))
            # Qwen3 appends the suffix K/V to the cache even with
            # use_cache=False.  Drop them so every chunk starts from the same
            # prefix instead of seeing the previous chunk's options.
            self._crop_cache(cache, suffix_ids.shape[1])
        return torch.cat(all_logits, dim=-1)

    def forward_noul(
        self,
        state: str,
        instructions: str,
        criteria: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Returns logit [1, 1] for noul prediction.

        With ``noul_marker``, the two answers are scored as options by the
        shared marker readout, so noul sees "false"/"true" in the same
        context-conditioned representation as choice and score.  Otherwise
        the pooled ``NoulHead`` (or the unified attention head) is used.

        The returned logit is the difference (true - false) so that sigmoid
        gives P(true), which is mathematically equivalent to softmax over
        the two options.
        """
        if self.noul_marker:
            option_texts = _noul_option_texts(criteria)
            logits = self._forward_options_by_marker(
                state, instructions, option_texts
            )
            return logits[:, 1:2] - logits[:, 0:1]
        if self.unified is not None:
            context_text = f"{state} {instructions}"
            context_hidden = self._encode_with_sequence(context_text)
            option_pooled = self._encode_text(["false", "true"])
            option_hidden = option_pooled.unsqueeze(0)  # [1, 2, hidden]
            return self.unified.noul_logit(context_hidden, option_hidden)
        text = f"{state} {instructions}"
        pooled = self._encode_text(text)
        return self.noul_head(pooled)

    def forward_answerable(self, state: str, instructions: str) -> torch.Tensor:
        """Returns the answerable logit [1] for the whole item.

        The decision is Bernoulli over the pooled context: there is no option
        set to score.  At inference ``sigmoid(logit)`` is the answerable
        probability and ``1 - sigmoid(logit)`` is the abstain probability; the
        option distribution is scaled by ``sigmoid(logit)`` as a mixture, not
        extended with a ``none`` key.
        """
        if self.answerable_head is None:
            raise RuntimeError(
                "forward_answerable needs answerable_head=True at construction"
            )
        pooled = self._encode_text(f"{state} {instructions}")
        return self.answerable_head(pooled).reshape(-1)

    def forward_choice(
        self, state: str, instructions: str, option_texts: list[str]
    ) -> torch.Tensor:
        """Returns logits [1, n_options] for choice prediction."""
        if self.option_marker:
            return self._forward_options_by_marker(state, instructions, option_texts)
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
                # One tokenizer call for all K options instead of K calls per
                # item: the per-call path is what keeps the GPU waiting.
                encode_batch=lambda texts: self.tokenizer(
                    texts, add_special_tokens=False
                )["input_ids"],
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
            logit = self.forward_noul(
                state, instructions, question.get("criteria")
            )
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
        """Load head parameters from a checkpoint, refusing a mismatched architecture.

        A checkpoint records the architecture that produced it.  Loading it into a
        model built with different head settings used to succeed silently:
        ``load_state_dict(strict=False)`` leaves every key it cannot match at its
        initial value, so the run reports a number for a head that was never
        trained.  Two measured cases:

        * ``eval_authored.yaml`` mirrors a 4B run whose ``noul_rank`` is 128, but
          the config did not say so.  The model got a linear noul head, the
          checkpoint's ``noul_head.proj.*`` went nowhere, and the set scored
          26/51 on the flag family — chance — against 34/51 with the head loaded.
        * An eval of the tower checkpoint ``arch_4b_32k_split138_s123`` under a
          cross-encoder config scored 0.2680 on choice instead of 0.1765: the
          cross-encoder choice head stayed random.

        A key the model cannot hold, or one whose shape differs, is always a
        mismatch, so those raise.  A model parameter the checkpoint does not carry
        only warns: that is what a legitimate partial load looks like (frozen
        backbone weights are never saved, and a heads-only checkpoint says
        nothing about adapters).

        Raises:
            ValueError: If the checkpoint carries parameters this model does not
                have, or with different shapes.  The message names the keys and
                says which setting produced them, because the mismatch is almost
                always a config that does not mirror the training run.
        """
        head_state = torch.load(Path(path), map_location=self.device, weights_only=True)
        own = self.state_dict()

        shape_mismatches = [
            f"{name}: checkpoint {tuple(value.shape)} vs model {tuple(own[name].shape)}"
            for name, value in head_state.items()
            if name in own and own[name].shape != value.shape
        ]
        unexpected = sorted(name for name in head_state if name not in own)
        if shape_mismatches or unexpected:
            parts = []
            if unexpected:
                shown = ", ".join(unexpected[:8])
                more = f" (+{len(unexpected) - 8} more)" if len(unexpected) > 8 else ""
                parts.append(f"{len(unexpected)} key(s) the model does not have: {shown}{more}")
            if shape_mismatches:
                shown = "; ".join(shape_mismatches[:4])
                more = f" (+{len(shape_mismatches) - 4} more)" if len(shape_mismatches) > 4 else ""
                parts.append(f"{len(shape_mismatches)} shape mismatch(es): {shown}{more}")
            raise ValueError(
                f"{Path(path)} does not match this model's architecture — "
                + "; ".join(parts)
                + ". Check that the config's rank/noul_rank/choice_mode/readout "
                "match the run that wrote the checkpoint."
            )

        # What `save_heads` would have written: every non-backbone entry, plus the
        # trainable backbone parameters (the LoRA adapters).
        trainable_names = {name for name, p in self.named_parameters() if p.requires_grad}
        expected = {
            name
            for name in own
            if "backbone" not in name or name in trainable_names
        }
        missing = sorted(expected - set(head_state))
        if missing:
            shown = ", ".join(missing[:8])
            more = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
            warnings.warn(
                f"{Path(path)} has no value for {len(missing)} parameter(s) this "
                f"model expects: {shown}{more}. They keep their initial values.",
                stacklevel=2,
            )

        self.load_state_dict(head_state, strict=False)


    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def frozen_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)
