"""Tests for the cross-encoder choice head.

The point of the mode is that an option's score depends on the state, because
the state is inside the option's prompt. These tests pin that property, the
zero-initialised read-out, and the checkpoint inference that lets evaluation
rebuild the same head.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from model.src.heads import CrossEncoderHead  # noqa: E402


class _Config:
    hidden_size = 16
    num_hidden_layers = 2
    max_position_embeddings = 128
    is_decoder = True


class _Block(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)


class _FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _Config()
        self.layers = nn.ModuleList([_Block(16) for _ in range(2)])


class _WordTokenizer:
    """Word-level stand-in: the prompt fitting only needs counts and a decode."""

    def __call__(self, text, add_special_tokens=True, **_kwargs):
        texts = [text] if isinstance(text, str) else list(text)
        return {"input_ids": [[hash(word) for word in value.split()] for value in texts]}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"w{index}" for index in range(len(ids)))


def _model(choice_mode: str):
    from model.src.decision_model import DecisionModel
    from model.src.gpu_config import GPUConfig

    gpu = GPUConfig(
        flash_attention=False,
        bf16_compute=False,
        max_length=128,
        gradient_checkpointing=False,
    )
    return DecisionModel(
        backbone=_FakeBackbone(),
        tokenizer=_WordTokenizer(),
        rank=8,
        gpu_config=gpu,
        choice_mode=choice_mode,
    )


class TestCrossEncoderHead:
    def test_it_returns_one_logit_per_option(self):
        head = CrossEncoderHead(16)
        assert head(torch.randn(5, 16)).shape == (5,)

    def test_a_fresh_head_scores_every_option_identically(self):
        # Zero-initialised read-out: the first training step starts from a
        # uniform distribution, not from whichever option a random projection
        # happens to favour.
        head = CrossEncoderHead(16).eval()
        scores = head(torch.randn(7, 16))
        assert torch.allclose(scores, scores[0].expand_as(scores))

    def test_it_scores_options_differently_once_the_weights_move(self):
        head = CrossEncoderHead(16).eval()
        with torch.no_grad():
            head.score.weight.normal_()
        scores = head(torch.randn(7, 16))
        assert not torch.allclose(scores, scores[0].expand_as(scores))


class TestForwardChoice:
    def _stubbed(self, choice_mode: str, recorder: list[str]):
        model = _model(choice_mode)

        def fake_encode(text, max_length=None):
            texts = [text] if isinstance(text, str) else list(text)
            recorder.extend(texts)
            # Deterministic embedding that depends on the whole string, so a
            # prompt containing the state embeds differently from one without.
            rows = []
            for value in texts:
                seed = sum(ord(char) for char in value) % 997
                generator = torch.Generator().manual_seed(seed)
                rows.append(torch.randn(model.hidden_size, generator=generator))
            return torch.stack(rows)

        model._encode_text = fake_encode  # type: ignore[assignment]
        model._encode_with_sequence = lambda text, max_length=None: torch.randn(  # type: ignore[assignment]
            1, 4, model.hidden_size
        )
        return model

    def test_cross_encoder_puts_the_state_into_every_option_prompt(self):
        recorder: list[str] = []
        model = self._stubbed("cross_encoder", recorder)
        logits = model.forward_choice(
            "the customer is angry", "which intent?", ["refund", "cancel"]
        )
        assert logits.shape == (1, 2)
        assert len(recorder) == 2
        for prompt in recorder:
            assert "the customer is angry" in prompt
            assert "which intent?" in prompt

    def test_tower_mode_keeps_the_two_encode_calls(self):
        recorder: list[str] = []
        model = self._stubbed("tower", recorder)
        model._encode_with_sequence = lambda text, max_length=None: (  # type: ignore[assignment]
            recorder.append(text) or torch.randn(1, 4, model.hidden_size)
        )
        model.forward_choice("state", "instructions", ["a", "b"])
        # One context encode plus one batched option encode.
        assert recorder[0] == "state instructions"
        assert recorder[1:] == ["a", "b"]

    def test_the_score_depends_on_the_state_in_cross_encoder_mode(self):
        recorder: list[str] = []
        model = self._stubbed("cross_encoder", recorder)
        # The read-out starts at zero, so every score is identical until the
        # weights move; move them to observe the dependency under test.
        with torch.no_grad():
            model.choice_head.score.weight.normal_()
        first = model.forward_choice("state one", "q", ["yes", "no"])
        second = model.forward_choice("state two", "q", ["yes", "no"])
        assert not torch.allclose(first, second)

    def test_an_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError):
            _model("bilinear")


class _Item:
    """The subset of ``TypedQuestion`` the supervised losses touch."""

    def __init__(self, state: str, question: dict, label=None, teacher_probs=None) -> None:
        self.state = state
        self.question = question
        self.label = label
        self.teacher_probs = teacher_probs


class TestTrainingLoops:
    """The losses must reach the head through ``forward_choice``.

    They used to call ``model.choice_head(context_hidden, opt_hidden)``
    directly, which a cross-encoder head cannot accept — the first
    ``--choice-mode cross_encoder`` pilot died on the argument count before a
    single step ran. ``forward_choice`` owns the head's calling convention, so
    these tests go through the real loss functions.
    """

    def _stubbed(self, choice_mode: str, recorder: list[str]):
        model = _model(choice_mode)

        def fake_encode(text, max_length=None):
            texts = [text] if isinstance(text, str) else list(text)
            recorder.extend(texts)
            rows = []
            for value in texts:
                seed = sum(ord(char) for char in value) % 997
                generator = torch.Generator().manual_seed(seed)
                rows.append(torch.randn(model.hidden_size, generator=generator))
            return torch.stack(rows)

        model._encode_text = fake_encode  # type: ignore[assignment]
        model._encode_with_sequence = lambda text, max_length=None: torch.randn(  # type: ignore[assignment]
            1, 4, model.hidden_size
        )
        return model

    def test_a_choice_batch_reaches_the_cross_encoder_head(self):
        from model.training.supervised import _compute_choice_batch

        recorder: list[str] = []
        model = self._stubbed("cross_encoder", recorder)
        item = _Item(
            "the customer is angry about a late delivery",
            {"instructions": "pick the intent", "criteria": {"refund": "", "cancel": "cancel it"}},
            label="refund",
        )
        losses = _compute_choice_batch(model, [item])
        assert len(losses) == 1
        assert losses[0] is not None and torch.isfinite(losses[0])
        assert losses[0].requires_grad
        # Every encoded prompt carries the state, which only the
        # cross-encoder path does.
        assert len(recorder) == 2
        assert all("late delivery" in prompt for prompt in recorder)
        # ``v or k``: an empty value falls back to the key.
        assert any("cancel it" in prompt for prompt in recorder)

    def test_an_augmented_noul_batch_reaches_the_cross_encoder_head(self):
        from model.training.supervised import _compute_augmented_noul_batch

        recorder: list[str] = []
        model = self._stubbed("cross_encoder", recorder)
        options = {"decline": "decline", "accept": "accept", "maybe": "maybe"}
        item = _Item(
            "an invoice arrives for a service not ordered",
            {"augmented_options": options},
            teacher_probs={"decline": 0.7, "accept": 0.2, "maybe": 0.1},
        )
        losses = _compute_augmented_noul_batch(model, [item])
        assert len(losses) == 1
        assert torch.isfinite(losses[0]) and losses[0].requires_grad
        assert len(recorder) == 3

    def test_the_tower_path_still_batches_options(self):
        from model.training.supervised import _compute_choice_batch

        recorder: list[str] = []
        model = self._stubbed("tower", recorder)
        model._encode_with_sequence = lambda text, max_length=None: (  # type: ignore[assignment]
            recorder.append(text) or torch.randn(1, 4, model.hidden_size)
        )
        item = _Item(
            "state",
            {"instructions": "q", "criteria": {"a": "", "b": ""}},
            label="a",
        )
        losses = _compute_choice_batch(model, [item])
        assert losses[0] is not None and torch.isfinite(losses[0])
        # One context encode, then one batched encode of both options.
        assert recorder[0] == "state q"
        assert recorder[1:] == ["a", "b"]


class _Batch(dict):
    """Stand-in for a tokenizer's BatchEncoding: a dict that knows ``.to``."""

    def to(self, _device):
        return self


class TestPooling:
    """The last-token pool must land on the last *real* token.

    Options are batched with right padding, so a batch's shorter sequences end
    in pad tokens. Pooling by sequence length instead of by mask would read a
    pad position and score the wrong vector — silently, and only for the
    shorter options in a batch.
    """

    def test_causal_pooling_takes_the_last_real_token(self):
        model = _model("tower")
        model.is_encoder = False
        input_ids = torch.tensor([[5, 6, 7, 8], [5, 6, 0, 0]])
        attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
        model.tokenizer = lambda *args, **kwargs: _Batch(
            input_ids=input_ids, attention_mask=attention_mask
        )  # type: ignore[assignment]

        hidden = torch.zeros(2, 4, model.hidden_size)
        for batch_index in range(2):
            for position in range(4):
                hidden[batch_index, position] = batch_index * 10 + position

        class _Out:
            hidden_states = (hidden,)

        model._backbone_forward = lambda inputs: _Out()  # type: ignore[assignment]
        model.projector = nn.Identity()  # type: ignore[assignment]

        pooled = model._encode_text(["a b c d", "e f"])
        assert torch.equal(pooled[0], hidden[0, 3])
        assert torch.equal(pooled[1], hidden[1, 1])
