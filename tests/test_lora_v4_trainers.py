"""LoRA and KL wiring in the v4 trainers.

The v4 arms predate LoRA: they built the model without adapters and the eval
scripts rebuilt it the same way.  These tests pin the three pieces that make an
adapted run reproducible:

* the CLI defaults stay frozen (``--lora-layers`` omitted),
* a zero-initialised adapter with the LoRA KL weight at 0 reproduces the
  frozen model's logits exactly, and
* a checkpoint's config reconstructs the adapter in a fresh model.

The tiny Llama keeps the identity check offline and CPU-only.  It is not the
Qwen3.5 production backbone, but the LoRA path is backbone-agnostic: the
adapter wraps ``q_proj``/``v_proj`` and starts at zero.
"""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
llama = pytest.importorskip("transformers.models.llama.modeling_llama")

from data.format import TypedQuestion  # noqa: E402
from model.scripts import train_listwise, train_pairwise  # noqa: E402
from model.src.decision_model import DecisionModel  # noqa: E402
from model.src.gpu_config import GPUConfig  # noqa: E402
from model.src.listwise import ListwiseMarkerReadout  # noqa: E402
from model.src.lora import lora_kwargs_from_config  # noqa: E402
from model.src.pairwise import PairwiseMarkerReadout  # noqa: E402

VOCAB = 128
MARKER_ID = 1
STATE = "a fairly long state " * 4
INSTRUCTIONS = "pick one"
OPTIONS = ["alpha", "beta", "gamma", "delta"]


class _Encoding(dict):
    def to(self, device):
        return self


class _CharTokenizer:
    """One token per byte, with the marker as its own id."""

    eos_token_id = MARKER_ID
    pad_token_id = 0

    def __call__(
        self,
        text,
        return_tensors=None,
        add_special_tokens=True,
        truncation=False,
        max_length=None,
        padding=False,
        **kwargs,
    ):
        ids = [2 + (b % (VOCAB - 4)) for b in str(text).encode()]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        if return_tensors == "pt":
            return _Encoding(
                input_ids=torch.tensor([ids], dtype=torch.long),
                attention_mask=torch.ones(1, len(ids), dtype=torch.long),
            )
        return {"input_ids": ids}


def _tiny_decision_model(**lora_kwargs):
    """Build an identical tiny backbone, optionally LoRA-adapted.

    ``torch.manual_seed(0)`` before each build makes the two models
    bit-identical when no adapter is injected.
    """
    torch.manual_seed(0)
    config = llama.LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=4096,
    )
    backbone = llama.LlamaForCausalLM(config).eval()
    gpu = GPUConfig(
        flash_attention=False,
        bf16_compute=False,
        max_length=256,
        gradient_checkpointing=False,
    )
    return DecisionModel(
        backbone=backbone,
        tokenizer=_CharTokenizer(),
        rank=8,
        gpu_config=gpu,
        option_marker=True,
        option_chunk_size=1,
        **lora_kwargs,
    ).eval()


def _choice_item(n_options: int = 4, gold: int = 2) -> TypedQuestion:
    criteria = {f"option_{i}": f"option text {i}" for i in range(n_options)}
    return TypedQuestion(
        id="lora-v4-test-1",
        state=STATE,
        question={
            "type": "choice",
            "instructions": INSTRUCTIONS,
            "criteria": criteria,
        },
        label=f"option_{gold}",
        source="lora_v4_test",
        split="train",
    )


def _parser_defaults(parser, extra: list[str]):
    base = ["--items", "a.jsonl", "--base-probs", "b.jsonl", "--output", "out"]
    args = parser.parse_args(base + extra)
    return args


class TestParserDefaults:
    """Every new flag must default to the pre-LoRA behaviour."""

    @pytest.mark.parametrize(
        "module", [train_listwise, train_pairwise], ids=["listwise", "pairwise"]
    )
    def test_lora_is_off_by_default(self, module) -> None:
        args = _parser_defaults(module.build_parser(), [])
        assert args.lora_layers is None
        assert args.lora_kl_weight == 0.0
        assert args.lora_rank == 16
        assert args.lora_alpha == 32
        assert args.lora_targets == ("q_proj", "v_proj")
        assert args.torch_seed is None

    @pytest.mark.parametrize(
        "module", [train_listwise, train_pairwise], ids=["listwise", "pairwise"]
    )
    def test_lora_flags_are_accepted(self, module) -> None:
        args = _parser_defaults(
            module.build_parser(),
            [
                "--lora-layers", "1",
                "--lora-rank", "4",
                "--lora-alpha", "8",
                "--lora-targets", "q_proj, o_proj",
                "--lora-kl-weight", "0.5",
                "--torch-seed", "7",
            ],
        )
        assert args.lora_layers == 1
        assert args.lora_rank == 4
        assert args.lora_alpha == 8
        assert args.lora_targets == ("q_proj", "o_proj")
        assert args.lora_kl_weight == 0.5
        assert args.torch_seed == 7

    def test_dry_run_is_off_by_default(self) -> None:
        args = _parser_defaults(train_listwise.build_parser(), [])
        assert args.dry_run is False

    def test_dry_run_is_accepted(self) -> None:
        args = _parser_defaults(train_listwise.build_parser(), ["--dry-run"])
        assert args.dry_run is True

    def test_holdout_ids_is_off_by_default(self) -> None:
        args = _parser_defaults(train_listwise.build_parser(), [])
        assert args.holdout_ids is None

    def test_holdout_ids_is_accepted(self) -> None:
        args = _parser_defaults(
            train_listwise.build_parser(),
            ["--holdout-ids", "heldout.json"],
        )
        assert str(args.holdout_ids) == "heldout.json"


class TestConfigRoundTrip:
    def test_a_frozen_config_adds_no_kwargs(self) -> None:
        config = {"lora_layers": None, "lora_rank": 16, "lora_alpha": 32}
        assert lora_kwargs_from_config(config) == {}

    def test_an_adapted_config_rebuilds_the_scope(self) -> None:
        config = {
            "lora_layers": -1,
            "lora_rank": 4,
            "lora_alpha": 8,
            "lora_targets": ["q_proj", "v_proj"],
            "lora_dropout": 0.0,
        }
        assert lora_kwargs_from_config(config) == {
            "lora_layers": -1,
            "lora_rank": 4,
            "lora_alpha": 8,
            "lora_targets": ("q_proj", "v_proj"),
            "lora_dropout": 0.0,
        }


class TestZeroInitialisedAdapter:
    """The issue's acceptance test: LoRA on, KL 0, logits unchanged."""

    def test_readout_logits_match_the_frozen_model(self) -> None:
        frozen = _tiny_decision_model()
        adapted = _tiny_decision_model(lora_layers=-1)

        with torch.no_grad():
            expected = ListwiseMarkerReadout(frozen).score(
                STATE, INSTRUCTIONS, OPTIONS
            )
            got = ListwiseMarkerReadout(adapted).score(STATE, INSTRUCTIONS, OPTIONS)

        assert torch.equal(expected, got), "a zero-init adapter changed the logits"

    def test_listwise_loss_matches_the_frozen_model_at_kl_zero(self) -> None:
        frozen = _tiny_decision_model()
        adapted = _tiny_decision_model(lora_layers=-1)
        item = _choice_item()
        base = torch.full((len(OPTIONS),), 0.25)
        alpha = torch.zeros(())

        frozen_loss = train_listwise.item_loss(
            ListwiseMarkerReadout(frozen), item, base, alpha,
            random.Random(0), 1.0, 0.2, 2, False, None, 0.0,
        )
        adapted_loss = train_listwise.item_loss(
            ListwiseMarkerReadout(adapted), item, base, alpha,
            random.Random(0), 1.0, 0.2, 2, False, None, 0.0,
        )

        assert torch.equal(frozen_loss[0], adapted_loss[0])

    def test_pairwise_loss_matches_the_frozen_model_at_kl_zero(self) -> None:
        frozen = _tiny_decision_model()
        adapted = _tiny_decision_model(lora_layers=-1)
        base = torch.tensor([0.25, 0.25, 0.25, 0.25])
        alpha = torch.zeros(())

        def loss(model):
            return PairwiseMarkerReadout(model).pair_loss(
                STATE, INSTRUCTIONS, OPTIONS, 2, 0, base, alpha,
                random.Random(0), kl_weight=1.0, ce_weight=0.2,
                order_sensitive=True, lora_kl_weight=0.0,
            )

        frozen_loss = loss(frozen)
        adapted_loss = loss(adapted)
        assert torch.equal(frozen_loss[0], adapted_loss[0])

    def test_the_anchor_term_is_off_at_zero_and_on_above_zero(self) -> None:
        model = _tiny_decision_model(lora_layers=-1)
        item = _choice_item()
        base = torch.full((len(OPTIONS),), 0.25)
        alpha = torch.zeros(())

        def run(weight: float):
            return train_listwise.item_loss(
                ListwiseMarkerReadout(model), item, base, alpha,
                random.Random(0), 1.0, 0.2, 2, False, None, weight,
            )[0]

        off = run(0.0)
        on = run(0.5)
        # The term is KL(base || readout), which is non-zero for a random
        # head, so a positive weight must change the loss.
        assert not torch.equal(off, on)


class TestGradEnabledHybridCache:
    """A grad-enabled GDN backbone must not share its prefix cache.

    The hybrid suffix forward overwrites the fork's conv/recurrent states in
    place.  That bumps the version of tensors the prefix graph saved for
    backward, and fla's chunked delta-rule backward raises.  The fix is to
    re-encode the context, so no call carries ``past_key_values``.
    """

    def _spy(self, model):
        calls = []
        real = model._backbone_forward

        def spy(inputs, **kwargs):
            calls.append("past_key_values" in inputs)
            return real(inputs, **kwargs)

        model._backbone_forward = spy
        model.is_hybrid = True
        model.backbone_requires_grad = True
        return calls

    def test_listwise_reencodes_the_context(self) -> None:
        model = _tiny_decision_model(lora_layers=-1)
        calls = self._spy(model)
        logits = ListwiseMarkerReadout(model).score(STATE, INSTRUCTIONS, OPTIONS)
        assert torch.isfinite(logits).all()
        # The combined call re-encodes the prefix, so the cache-only prefix
        # forward is skipped: one backbone call, and it carries no cache.
        assert len(calls) == 1
        assert not any(calls)

    def test_the_marker_path_reencodes_the_context(self) -> None:
        model = _tiny_decision_model(lora_layers=-1)
        calls = self._spy(model)
        logits = model.forward_choice(STATE, INSTRUCTIONS, OPTIONS)
        assert torch.isfinite(logits).all()
        # chunk_size=1 re-encodes once per option; none of them carries a cache.
        assert len(calls) == len(OPTIONS)
        assert not any(calls)

    def test_the_reencoded_context_has_no_fully_masked_rows(self) -> None:
        model = _tiny_decision_model(lora_layers=-1)
        seen = []
        real = model._backbone_forward

        def spy(inputs, **kwargs):
            seen.append(inputs)
            return real(inputs, **kwargs)

        model._backbone_forward = spy
        model.is_hybrid = True
        model.backbone_requires_grad = True
        ListwiseMarkerReadout(model).score(STATE, INSTRUCTIONS, OPTIONS)

        combined = seen[-1]
        mask = combined["attention_mask"]
        # Right padding: position 0 is a real token (a left-padded position 0
        # is fully masked under a causal mask, and cuDNN's SDPA backward
        # returns NaN for that row).  The mask is a run of ones then zeros.
        assert int(mask[0, 0]) == 1
        assert torch.all(mask[:, :-1] >= mask[:, 1:])
        assert int(combined["input_ids"][0, 0]) != model.tokenizer.pad_token_id

    def test_the_marker_path_has_no_fully_masked_rows(self) -> None:
        model = _tiny_decision_model(lora_layers=-1)
        seen = []
        real = model._backbone_forward

        def spy(inputs, **kwargs):
            seen.append(inputs)
            return real(inputs, **kwargs)

        model._backbone_forward = spy
        model.is_hybrid = True
        model.backbone_requires_grad = True
        model.forward_choice(STATE, INSTRUCTIONS, OPTIONS)

        assert int(seen[-1]["attention_mask"].min()) == 1

    def test_eval_keeps_the_cache(self) -> None:
        """No grad means the fork cannot break autograd, so reuse the prefix."""
        model = _tiny_decision_model(lora_layers=-1)
        calls = self._spy(model)
        with torch.no_grad():
            logits = ListwiseMarkerReadout(model).score(
                STATE, INSTRUCTIONS, OPTIONS
            )
        assert torch.isfinite(logits).all()
        assert calls == [False, True]

    def test_eval_marker_path_keeps_the_cache(self) -> None:
        model = _tiny_decision_model(lora_layers=-1)
        calls = self._spy(model)
        with torch.no_grad():
            model.forward_choice(STATE, INSTRUCTIONS, OPTIONS)
        assert calls[-1] is True


class TestCheckpointReload:
    """An eval rebuild must restore the adapter, not just the heads."""

    def test_the_adapter_survives_save_and_config_reload(self) -> None:
        trained = _tiny_decision_model(lora_layers=-1)
        # Move the adapters off their zero initialisation so a dropped
        # adapter is visible in the logits.
        with torch.no_grad():
            for name, param in trained.named_parameters():
                if "lora_b" in name:
                    param.fill_(0.25)

        trainable = {name for name, p in trained.named_parameters() if p.requires_grad}
        heads = {
            key: value.detach().cpu()
            for key, value in trained.state_dict().items()
            if "backbone" not in key or key in trainable
        }
        config = {
            "lora_layers": -1,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_targets": ["q_proj", "v_proj"],
        }

        reloaded = _tiny_decision_model(**lora_kwargs_from_config(config))
        reloaded.load_state_dict(heads, strict=False)

        with torch.no_grad():
            expected = ListwiseMarkerReadout(trained).score(
                STATE, INSTRUCTIONS, OPTIONS
            )
            got = ListwiseMarkerReadout(reloaded).score(STATE, INSTRUCTIONS, OPTIONS)

        assert torch.equal(expected, got)


def test_torch_seed_moves_the_global_rng(tmp_path, monkeypatch):
    """The flag must seed torch, not only be written into the config.

    A run that accepts ``--torch-seed`` without applying it records an
    initialisation it did not use.  The dry run returns before the backbone is
    loaded, so this exercises the real entry point cheaply.
    """
    import json
    import sys

    ids = [f"fixture-{index}" for index in range(5)]
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": item_id,
                    "state": STATE,
                    "question": {
                        "type": "choice",
                        "instructions": INSTRUCTIONS,
                        "criteria": {"alpha": "alpha", "beta": "beta"},
                    },
                    "label": "alpha",
                    "source": "fixture",
                    "split": "train",
                }
            )
            for item_id in ids
        )
        + "\n"
    )
    base_probs = tmp_path / "base.jsonl"
    base_probs.write_text(
        "\n".join(json.dumps({"id": item_id, "probs": [0.5, 0.5]}) for item_id in ids)
        + "\n"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_listwise",
            "--items",
            str(corpus),
            "--base-probs",
            str(base_probs),
            "--output",
            str(tmp_path / "out"),
            "--dry-run",
            "--limit",
            str(len(ids)),
            "--torch-seed",
            "12345",
        ],
    )

    torch.manual_seed(7)
    assert torch.initial_seed() == 7
    assert train_listwise.main() == 0
    assert torch.initial_seed() == 12345
