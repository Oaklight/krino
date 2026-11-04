"""Tests for the Bernoulli answerable (abstention) head.

The head is a whole-item decision over the pooled context, so these tests use
a deterministic stub backbone and assert the properties the abstention path
depends on: one logit, a checkpoint that round-trips the head, a strict refusal
when the head is absent, and the per-item JSONL fields the abstention eval
consumes.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from model.src.decision_model import DecisionModel
from model.src.gpu_config import GPUConfig


class _Encoding(dict):
    def to(self, device):
        return self


class _Tokenizer:
    eos_token_id = 2

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
        texts = [text] if isinstance(text, str) else list(text)
        rows = [[3 + (ord(ch) % 5) for ch in str(value)] for value in texts]
        if return_tensors == "pt":
            width = max(len(row) for row in rows)
            ids = [row + [0] * (width - len(row)) for row in rows]
            mask = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
            return _Encoding(
                input_ids=torch.tensor(ids, dtype=torch.long),
                attention_mask=torch.tensor(mask, dtype=torch.long),
            )
        return {"input_ids": rows[0]}


class _Config:
    hidden_size = 16
    num_hidden_layers = 2
    max_position_embeddings = 512
    is_decoder = True


class _Output:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states
        self.past_key_values = None


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _Config()
        self.embed = nn.Embedding(100, self.config.hidden_size)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        output_hidden_states=True,
        use_cache=False,
        **kwargs,
    ):
        hidden = self.embed(input_ids)
        states = tuple(hidden for _ in range(self.config.num_hidden_layers + 1))
        return _Output(states)


def _model(**kwargs) -> DecisionModel:
    gpu = GPUConfig(
        flash_attention=False,
        bf16_compute=False,
        max_length=512,
        gradient_checkpointing=False,
    )
    return DecisionModel(
        backbone=_Backbone(),
        tokenizer=_Tokenizer(),
        rank=8,
        gpu_config=gpu,
        **kwargs,
    )


class TestAnswerableHead:
    def test_forward_answerable_returns_one_logit(self) -> None:
        model = _model(answerable_head=True)
        logit = model.forward_answerable("the customer is angry", "is it answerable?")
        assert logit.shape == (1,)
        probability = torch.sigmoid(logit).item()
        assert 0.0 <= probability <= 1.0

    def test_a_missing_head_raises(self) -> None:
        model = _model()
        with pytest.raises(RuntimeError, match="answerable_head"):
            model.forward_answerable("state", "question?")

    def test_the_checkpoint_round_trips_the_head(self, tmp_path) -> None:
        model = _model(answerable_head=True)
        with torch.no_grad():
            for parameter in model.answerable_head.parameters():
                parameter.fill_(0.5)
        path = tmp_path / "heads.pt"
        model.save_heads(path)

        fresh = _model(answerable_head=True)
        with torch.no_grad():
            for parameter in fresh.answerable_head.parameters():
                parameter.zero_()
        fresh.load_heads(path)

        for saved, loaded in zip(
            model.answerable_head.parameters(), fresh.answerable_head.parameters()
        ):
            assert torch.equal(saved, loaded)

    def test_load_refuses_a_head_the_model_lacks(self, tmp_path) -> None:
        path = tmp_path / "heads.pt"
        _model(answerable_head=True).save_heads(path)
        with pytest.raises(ValueError, match="does not match this model's architecture"):
            _model().load_heads(path)


class TestPredictionRecords:
    def test_records_carry_the_abstention_fields(self, tmp_path) -> None:
        from model.evaluation.gate import Prediction
        from model.scripts.eval_gate import write_prediction_records

        row = Prediction(
            item_id="item-a",
            q_type="choice",
            label="refund",
            probs=(0.2, 0.8),
            keys=("refund", "cancel"),
        )

        class _Item:
            id = "item-a"
            state = "the customer is angry"
            question = {"type": "choice", "instructions": "pick"}
            reason = "insufficient"

        class _Model:
            def p_answerable(self, item):
                return 0.25

        out = tmp_path / "predictions.jsonl"
        written = write_prediction_records(out, [row], [_Item()], _Model())
        assert written == 1
        record = json.loads(out.read_text().strip())
        assert record["id"] == "item-a"
        assert record["q_type"] == "choice"
        assert record["n_options"] == 2
        assert record["pred"] == "cancel"
        assert record["correct"] is False
        assert record["max_prob"] == pytest.approx(0.8)
        assert record["p_answerable"] == pytest.approx(0.25)
        assert record["reason"] == "insufficient"

    def test_a_model_without_the_head_writes_null(self, tmp_path) -> None:
        from model.evaluation.gate import Prediction
        from model.scripts.eval_gate import write_prediction_records

        row = Prediction(
            item_id="item-b", q_type="noul", label=True, probs=(0.1, 0.9), keys=("false", "true")
        )
        out = tmp_path / "predictions.jsonl"
        write_prediction_records(out, [row], [], object())
        record = json.loads(out.read_text().strip())
        assert record["p_answerable"] is None
        assert record["correct"] is True


class TestAnswerableTarget:
    """The training label heuristic the abstention sets rely on."""

    def _item(self, label, criteria, answerable=None):
        class _Item:
            pass

        item = _Item()
        item.label = label
        item.answerable = answerable
        item.question = {"type": "choice", "criteria": criteria}
        return item

    def test_a_gold_carrying_item_is_answerable(self) -> None:
        from model.scripts.train_residual import answerable_target

        assert answerable_target(self._item("a", {"a": "", "b": ""})) is True

    def test_an_explicit_false_wins_over_the_gold(self) -> None:
        from model.scripts.train_residual import answerable_target

        item = self._item("a", {"a": "", "b": ""}, answerable=False)
        assert answerable_target(item) is False

    def test_a_removed_gold_is_not_answerable(self) -> None:
        from model.scripts.train_residual import answerable_target

        assert answerable_target(self._item("gone", {"a": "", "b": ""})) is False


class TestAnswerableLoss:
    def test_item_loss_with_pos_weight_is_scalar(self) -> None:
        """The BCE term must broadcast against a scalar logit.

        A [1]-shaped pos_weight against a [] logit raised
        "output with shape [] doesn't match the broadcast shape [1]" on the
        first real training step.
        """
        from model.scripts.train_residual import item_loss

        class _Item:
            id = "i"
            state = "the customer is angry"
            label = False
            answerable = False
            teacher_probs = None
            question = {"type": "noul", "instructions": "is it answerable?"}

        model = _model(answerable_head=True)
        base = torch.tensor([0.5, 0.5])
        alphas = {
            name: torch.zeros((), requires_grad=True)
            for name in ("noul", "choice", "score")
        }
        loss, _, _ = item_loss(model, _Item(), base, alphas, 1.0, 0.2, 1.0, 0.5)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
