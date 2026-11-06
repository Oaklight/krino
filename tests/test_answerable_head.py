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

    def test_an_invalid_unset_label_has_no_target(self) -> None:
        from model.scripts.train_residual import answerable_target

        # A malformed label means the row cannot be scored, not that the
        # answer is absent; it must not teach the head to abstain.
        assert answerable_target(self._item("gone", {"a": "", "b": ""})) is None

    def test_a_teacher_row_has_no_target(self) -> None:
        from model.scripts.train_residual import answerable_target

        item = self._item("a", {"a": "", "b": ""})
        item.provenance = "teacher"
        assert answerable_target(item) is None


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
        # At alpha=0 the fusion is the base distribution: KL is zero and the
        # option head's gradient is zero whatever the loss does, so the
        # assertions below could not tell a masked row from an unmasked one.
        alphas = {
            name: torch.ones((), requires_grad=True)
            for name in ("noul", "choice", "score")
        }
        loss, _, _ = item_loss(model, _Item(), base, alphas, 1.0, 0.2, 1.0, 0.5)
        assert loss.ndim == 0
        assert torch.isfinite(loss)


class TestTaskLossMask:
    """The task loss is masked on the loss, not by dropping rows.

    An ``answerable=false`` row keeps its provenance label and must train the
    abstain target only.  The option head still runs on it, so a gradient that
    reaches it would mean the mask sits on the input instead of the loss.
    """

    def _noul(
        self,
        label=False,
        answerable=False,
        reason="insufficient",
        provenance="execution",
        teacher_probs=None,
    ):
        class _Item:
            pass

        item = _Item()
        item.id = "mask-item"
        item.state = "the customer is angry"
        item.label = label
        item.answerable = answerable
        item.reason = reason
        item.provenance = provenance
        item.teacher_probs = teacher_probs
        item.teacher_name = "jev" if teacher_probs else None
        item.question = {
            "type": "noul",
            "instructions": "is the customer angry?",
            "criteria": None,
        }
        return item

    def _run(self, item, kl_weight=1.0, ce_weight=0.2, answerable_weight=1.0):
        from model.scripts.train_residual import item_loss

        model = _model(answerable_head=True)
        # Eval mode makes the NoulHead dropout identity so the BCE term can be
        # recomputed from a second forward; gradients are unchanged.
        model.eval()
        base = torch.tensor([0.5, 0.5])
        # At alpha=0 the fusion is the base distribution: KL is zero and the
        # option head's gradient is zero whatever the loss does, so the
        # assertions below could not tell a masked row from an unmasked one.
        alphas = {
            name: torch.ones((), requires_grad=True)
            for name in ("noul", "choice", "score")
        }
        loss, kl, ce = item_loss(
            model,
            item,
            base,
            alphas,
            kl_weight,
            ce_weight,
            answerable_weight,
            1.0,
        )
        loss.backward()
        return model, alphas, loss, kl, ce

    @staticmethod
    def _nonzero_grad(parameters):
        return any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
            for parameter in parameters
        )

    def test_abstention_row_zeroes_the_task_terms_and_option_gradient(self):
        model, alphas, loss, kl, ce = self._run(self._noul())
        assert kl == 0.0
        assert ce == 0.0
        # The row still carries the abstain target, so the step is not inert.
        assert loss.item() > 0.0
        logit = model.forward_answerable(
            "the customer is angry", "is the customer angry?"
        ).reshape(())
        abstain = torch.nn.functional.binary_cross_entropy_with_logits(
            logit, torch.tensor(0.0)
        )
        assert loss.item() == pytest.approx(abstain.item(), abs=1e-6)
        for parameter in model.noul_head.parameters():
            # No edge from the loss reaches the option path, so the grad is
            # None; a populated-but-zero grad would also pass.
            assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        assert alphas["noul"].grad is None or torch.count_nonzero(alphas["noul"].grad) == 0
        assert self._nonzero_grad(model.answerable_head.parameters())

    def test_a_batch_of_abstention_rows_stays_zero_on_the_option_path(self):
        from model.scripts.train_residual import answerable_target, item_loss

        items = [
            self._noul(reason="insufficient"),
            self._noul(reason="out_of_scope"),
            self._noul(reason="none_of_these"),
            self._noul(reason="support_removed", label="false"),
        ]
        assert all(answerable_target(item) is False for item in items)
        model = _model(answerable_head=True)
        model.eval()
        base = torch.tensor([0.5, 0.5])
        # At alpha=0 the fusion is the base distribution: KL is zero and the
        # option head's gradient is zero whatever the loss does, so the
        # assertions below could not tell a masked row from an unmasked one.
        alphas = {
            name: torch.ones((), requires_grad=True)
            for name in ("noul", "choice", "score")
        }
        kl_total = ce_total = 0.0
        for item in items:
            loss, kl_value, ce_value = item_loss(
                model, item, base, alphas, 1.0, 0.2, 1.0, 1.0
            )
            loss.backward()
            kl_total += kl_value
            ce_total += ce_value
        assert kl_total == 0.0
        assert ce_total == 0.0
        for parameter in model.noul_head.parameters():
            assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        assert alphas["noul"].grad is None or torch.count_nonzero(alphas["noul"].grad) == 0
        assert self._nonzero_grad(model.answerable_head.parameters())

    def test_a_masked_row_has_no_graph_when_the_head_is_off(self):
        """The A0 arm must skip backward() instead of calling it on a zero."""
        from model.scripts.train_residual import item_loss

        model = _model()  # no answerable head
        alphas = {
            name: torch.ones((), requires_grad=True)
            for name in ("noul", "choice", "score")
        }
        loss, kl, ce = item_loss(
            model,
            self._noul(),
            torch.tensor([0.5, 0.5]),
            alphas,
            1.0,
            0.2,
            1.0,
            1.0,
        )
        assert kl == 0.0
        assert ce == 0.0
        assert not loss.requires_grad

    def test_the_option_head_still_runs_on_an_abstention_row(self):
        """Masking on the input would skip this forward; the logits prove it ran."""
        from model.scripts.train_residual import head_logits

        model = _model(answerable_head=True)
        item = self._noul()
        head = head_logits(model, item)
        assert head.shape == (1, 2)
        assert torch.isfinite(head).all()

    def test_answerable_row_trains_the_positive_class_not_abstain(self):
        from model.scripts.train_residual import (
            answerable_target,
            item_loss,
        )

        item = self._noul(label=True, answerable=True, reason=None)
        model = _model(answerable_head=True)
        model.eval()
        base = torch.tensor([0.5, 0.5])
        # At alpha=0 the fusion is the base distribution: KL is zero and the
        # option head's gradient is zero whatever the loss does, so the
        # assertions below could not tell a masked row from an unmasked one.
        alphas = {
            name: torch.ones((), requires_grad=True)
            for name in ("noul", "choice", "score")
        }
        loss, kl, ce = item_loss(model, item, base, alphas, 0.0, 0.2, 1.0, 1.0)
        logit = model.forward_answerable(
            item.state, item.question["instructions"]
        ).reshape(())
        positive = torch.nn.functional.binary_cross_entropy_with_logits(
            logit, torch.tensor(1.0)
        )
        negative = torch.nn.functional.binary_cross_entropy_with_logits(
            logit, torch.tensor(0.0)
        )
        assert answerable_target(item) is True
        assert loss.item() == pytest.approx((0.2 * ce + positive).item(), abs=1e-6)
        assert abs(loss.item() - (0.2 * ce + negative).item()) > 1e-3

    def test_an_invalid_noul_label_contributes_no_hard_ce(self):
        item = self._noul(label="maybe", answerable=None, reason=None)
        _, _, _, _, ce = self._run(item, kl_weight=0.0, ce_weight=1.0)
        assert ce == 0.0

    def test_a_teacher_row_has_no_hard_ce_and_no_abstain_bce(self):
        item = self._noul(label=True, answerable=None, reason=None, provenance="teacher")
        model, _, loss, _, ce = self._run(item, kl_weight=0.0, answerable_weight=1.0)
        assert ce == 0.0
        assert loss.item() == 0.0
        assert not self._nonzero_grad(model.answerable_head.parameters())

    def test_a_teacher_row_still_trains_from_the_soft_target(self):
        from model.scripts.train_residual import fusion_logits, head_logits

        item = self._noul(
            label=False,
            answerable=None,
            reason=None,
            provenance="teacher",
            teacher_probs={"false": 0.25, "true": 0.75},
        )
        model = _model(answerable_head=True)
        model.eval()
        base = torch.tensor([0.5, 0.5])
        # At alpha=0 the fusion is the base distribution: KL is zero and the
        # option head's gradient is zero whatever the loss does, so the
        # assertions below could not tell a masked row from an unmasked one.
        alphas = {
            name: torch.ones((), requires_grad=True)
            for name in ("noul", "choice", "score")
        }
        from model.scripts.train_residual import item_loss

        loss, _, ce = item_loss(model, item, base, alphas, 0.0, 1.0, 1.0, 1.0)
        head = head_logits(model, item).reshape(-1)
        log_p = torch.log_softmax(fusion_logits(base, head, alphas["noul"]), dim=-1)
        soft = torch.tensor([0.25, 0.75])
        expected = torch.sum(soft * (torch.log(soft) - log_p))
        assert loss.item() == pytest.approx(expected.item(), abs=1e-6)
        assert ce == pytest.approx(expected.item(), abs=1e-6)


class TestEvaluateSkipsAbstentionRows:
    def test_unanswerable_rows_do_not_enter_the_accuracy(self):
        from model.scripts.train_residual import evaluate

        model = _model(answerable_head=True)
        answerable = TestTaskLossMask()._noul(
            label=True, answerable=True, reason=None
        )
        answerable.id = "answerable"
        abstain = TestTaskLossMask()._noul()
        abstain.id = "abstain"
        base = {
            "answerable": torch.tensor([0.5, 0.5]),
            "abstain": torch.tensor([0.5, 0.5]),
        }
        alphas = {name: torch.zeros(()) for name in ("noul", "choice", "score")}
        metrics = evaluate(model, [answerable, abstain], base, alphas)
        assert metrics["n"] == 1


class TestHardLabelRule:
    def _item(self, q_type, label, criteria):
        class _Item:
            pass

        item = _Item()
        item.label = label
        item.answerable = None
        item.provenance = "gold"
        item.question = {"type": q_type, "criteria": criteria}
        return item

    def test_noul_string_labels_are_parsed_not_truthy(self):
        from model.scripts.train_residual import _label_index

        for value, expected in (
            (False, 0),
            (True, 1),
            ("false", 0),
            ("FALSE", 0),
            ("true", 1),
        ):
            item = self._item("noul", value, None)
            assert _label_index(item) == expected

    def test_the_rule_rejects_bad_shapes_and_tiers(self):
        from model.scripts.train_residual import hard_label_eligible

        good_noul = self._item("noul", "true", None)
        assert hard_label_eligible(good_noul) is True

        bad_noul = self._item("noul", "maybe", None)
        assert hard_label_eligible(bad_noul) is False

        bad_choice = self._item("choice", "gone", {"a": "", "b": ""})
        assert hard_label_eligible(bad_choice) is False

        bad_score = self._item("score", 5, [1, 2, 3])
        assert hard_label_eligible(bad_score) is False

        good_score = self._item("score", 2, [1, 2, 3])
        assert hard_label_eligible(good_score) is True

        lesson = self._item("noul", True, None)
        lesson.answerable = False
        assert hard_label_eligible(lesson) is False

        teacher = self._item("noul", True, None)
        teacher.provenance = "teacher"
        assert hard_label_eligible(teacher) is False


class TestDedup:
    def _item(self, item_id, criteria):
        class _Item:
            pass

        item = _Item()
        item.id = item_id
        item.question = {"type": "choice", "criteria": criteria}
        return item

    def test_distinct_questions_with_the_same_id_survive(self):
        """RACE: 498 distinct questions over 144 passage ids."""
        from model.scripts.train_residual import dedup_by_id

        rows = [
            self._item("race-1", {"a": "x", "b": "y"}),
            self._item("race-1", {"a": "x", "b": "z"}),
            self._item("race-1", {"a": "x", "b": "y"}),  # true duplicate
            self._item("other", {"a": "x", "b": "y"}),
        ]
        kept, dropped = dedup_by_id(rows)
        assert len(kept) == 3
        assert dropped == 1
        assert [item.question["criteria"]["b"] for item in kept] == ["y", "z", "y"]

    def test_a_pre_seeded_key_is_dropped(self):
        from model.scripts.train_residual import _question_digest, dedup_by_id

        first = self._item("a", {"a": "x", "b": "y"})
        other = self._item("b", {"a": "x", "b": "y"})
        seen = {(first.id, _question_digest(first))}
        kept, dropped = dedup_by_id([first, other], seen)
        assert [item.id for item in kept] == ["b"]
        assert dropped == 1


class TestListwiseGuard:
    def test_the_listwise_loss_skips_an_abstention_row(self):
        """The paired corpus keeps the label, so only this guard stops the CE."""
        from model.scripts.train_listwise import item_loss as listwise_item_loss

        item = TestTaskLossMask()._noul(answerable=False)
        zero, kl, ce = listwise_item_loss(
            None,
            item,
            torch.tensor([0.5, 0.5]),
            torch.ones(()),
            None,
            1.0,
            1.0,
            2,
            False,
        )
        assert zero.item() == 0.0
        assert kl == 0.0
        assert ce == 0.0
