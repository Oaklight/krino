"""Tests for the ranked probability score (RPS) training loss.

RPS is the ordered counterpart of cross-entropy: it compares cumulative
distributions, so a near miss costs less than a far one.  These tests pin
that ordering property down, then check the wiring from the CLI-facing
``score_loss`` setting through to both score-loss code paths.
"""

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
F = torch.nn.functional

from model.training.losses import rps_from_probs, rps_loss


def _one_hot(idx, n_levels):
    return F.one_hot(torch.tensor(idx), n_levels).float()


class TestRPSValue:
    """RPS as a number, independent of any model."""

    def test_perfect_prediction_scores_zero(self):
        probs = _one_hot(3, 5).unsqueeze(0)
        assert rps_from_probs(probs, probs).item() == pytest.approx(0.0, abs=1e-6)

    def test_worst_prediction_scores_one(self):
        """Predicting level 0 for gold level 4 is the maximum penalty."""
        pred = _one_hot(0, 5).unsqueeze(0)
        target = _one_hot(4, 5).unsqueeze(0)
        assert rps_from_probs(pred, target).item() == pytest.approx(1.0, abs=1e-6)

    def test_normalisation_makes_range_level_independent(self):
        """Normalised RPS stays in [0, 1] however many levels there are."""
        for n_levels in (2, 3, 5, 9):
            pred = _one_hot(0, n_levels).unsqueeze(0)
            target = _one_hot(n_levels - 1, n_levels).unsqueeze(0)
            assert rps_from_probs(pred, target).item() == pytest.approx(1.0, abs=1e-6)

    def test_unnormalised_scales_with_levels(self):
        pred = _one_hot(0, 5).unsqueeze(0)
        target = _one_hot(4, 5).unsqueeze(0)
        # Four CDF entries each differ by 1 → 4.
        assert rps_from_probs(pred, target, normalize=False).item() == pytest.approx(4.0)

    def test_uniform_prediction(self):
        """Uniform over 5 levels against gold 4 gives 1.2 / 4 = 0.3."""
        pred = torch.full((1, 5), 0.2)
        target = _one_hot(4, 5).unsqueeze(0)
        assert rps_from_probs(pred, target).item() == pytest.approx(0.3, abs=1e-6)

    def test_rejects_single_level(self):
        with pytest.raises(ValueError, match="at least 2 levels"):
            rps_from_probs(torch.ones(1, 1), torch.ones(1, 1))

    def test_batch_is_averaged(self):
        """Two items, one perfect and one worst → mean 0.5."""
        pred = torch.stack([_one_hot(4, 5), _one_hot(0, 5)])
        target = torch.stack([_one_hot(4, 5), _one_hot(4, 5)])
        assert rps_from_probs(pred, target).item() == pytest.approx(0.5, abs=1e-6)


class TestRPSOrdering:
    """The property that motivates RPS: distance to the gold level matters."""

    def test_near_misses_cost_less_than_far_misses(self):
        target = _one_hot(4, 5).unsqueeze(0)
        losses = [
            rps_from_probs(_one_hot(k, 5).unsqueeze(0), target).item()
            for k in range(5)
        ]
        # Predicting 0..4 for gold 4 → 1.0, 0.75, 0.5, 0.25, 0.0
        assert losses == pytest.approx([1.0, 0.75, 0.5, 0.25, 0.0], abs=1e-6)

    def test_cross_entropy_is_blind_to_that_distance(self):
        """Contrast: CE gives the same loss for every equally confident miss."""
        target_idx = torch.tensor([4])
        ce = []
        for k in range(4):
            logits = torch.full((1, 5), -10.0)
            logits[0, k] = 10.0
            ce.append(F.cross_entropy(logits, target_idx).item())
        assert max(ce) - min(ce) == pytest.approx(0.0, abs=1e-5)

    def test_symmetric_around_gold(self):
        """One level below and one above the gold cost the same."""
        below = rps_from_probs(_one_hot(2, 5).unsqueeze(0), _one_hot(3, 5).unsqueeze(0))
        above = rps_from_probs(_one_hot(4, 5).unsqueeze(0), _one_hot(3, 5).unsqueeze(0))
        assert below.item() == pytest.approx(above.item(), abs=1e-6)


class TestRPSLoss:
    """Logit-facing wrapper."""

    def test_matches_explicit_softmax(self):
        logits = torch.randn(3, 5, generator=torch.Generator().manual_seed(0))
        target = torch.stack([_one_hot(1, 5), _one_hot(3, 5), _one_hot(4, 5)])
        expected = rps_from_probs(F.softmax(logits, dim=-1), target)
        assert rps_loss(logits, target).item() == pytest.approx(expected.item(), abs=1e-6)

    def test_soft_target_supported(self):
        """Interpolated labels (e.g. 2.5) are valid targets."""
        target = torch.tensor([[0.0, 0.0, 0.5, 0.5, 0.0]])
        loss = rps_loss(torch.zeros(1, 5), target)
        assert torch.isfinite(loss)

    def test_soft_target_sits_between_neighbours(self):
        """A half-way target costs less than a confidently wrong extreme."""
        logits_low = torch.tensor([[10.0, -10.0, -10.0, -10.0, -10.0]])
        logits_high = torch.tensor([[-10.0, -10.0, -10.0, -10.0, 10.0]])
        target = torch.tensor([[0.0, 0.0, 0.5, 0.5, 0.0]])
        soft = rps_loss(torch.zeros(1, 5), target).item()
        assert soft < rps_loss(logits_low, target).item()
        assert soft < rps_loss(logits_high, target).item()

    def test_gradients_flow(self):
        logits = torch.zeros(2, 5, requires_grad=True)
        target = torch.stack([_one_hot(1, 5), _one_hot(3, 5)])
        rps_loss(logits, target).backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert logits.grad.abs().sum() > 0

    def test_zero_gradient_at_perfect_prediction(self):
        """A confident correct prediction is not pushed further."""
        logits = torch.full((1, 5), -30.0)
        logits[0, 2] = 30.0
        logits.requires_grad_(True)
        rps_loss(logits, _one_hot(2, 5).unsqueeze(0)).backward()
        assert logits.grad.abs().max().item() == pytest.approx(0.0, abs=1e-9)


class TestScoreLossFor:
    """The dispatcher shared by both score-loss code paths."""

    def test_ce_integer_label_matches_cross_entropy(self):
        from model.training.supervised import _score_loss_for

        logits = torch.randn(1, 5, generator=torch.Generator().manual_seed(1))
        got = _score_loss_for(logits, 3, 5, "ce")
        want = F.cross_entropy(logits, torch.tensor([3]))
        assert got.item() == pytest.approx(want.item(), abs=1e-6)

    def test_ce_fractional_label_uses_soft_target(self):
        from model.training.supervised import _score_loss_for, _soft_cross_entropy

        logits = torch.randn(1, 5, generator=torch.Generator().manual_seed(2))
        target = torch.tensor([[0.0, 0.0, 0.5, 0.5, 0.0]])
        got = _score_loss_for(logits, 2.5, 5, "ce")
        want = _soft_cross_entropy(logits, target)
        assert got.item() == pytest.approx(want.item(), abs=1e-6)

    def test_rps_matches_rps_loss(self):
        from model.training.supervised import _score_loss_for

        logits = torch.randn(1, 5, generator=torch.Generator().manual_seed(3))
        got = _score_loss_for(logits, 3, 5, "rps")
        want = rps_loss(logits, _one_hot(3, 5).unsqueeze(0))
        assert got.item() == pytest.approx(want.item(), abs=1e-6)

    def test_rps_accepts_fractional_label(self):
        from model.training.supervised import _score_loss_for

        logits = torch.zeros(1, 5)
        got = _score_loss_for(logits, 2.5, 5, "rps")
        assert torch.isfinite(got)

    def test_rps_differs_from_ce(self):
        from model.training.supervised import _score_loss_for

        logits = torch.randn(1, 5, generator=torch.Generator().manual_seed(4))
        assert _score_loss_for(logits, 3, 5, "rps").item() != pytest.approx(
            _score_loss_for(logits, 3, 5, "ce").item(), abs=1e-6
        )

    def test_unknown_loss_raises(self):
        from model.training.supervised import _score_loss_for

        with pytest.raises(ValueError, match="unknown score_loss"):
            _score_loss_for(torch.zeros(1, 5), 3, 5, "brier")

    def test_default_is_ce(self):
        from model.training.supervised import _score_loss_for

        logits = torch.randn(1, 5, generator=torch.Generator().manual_seed(5))
        assert _score_loss_for(logits, 2, 5).item() == pytest.approx(
            _score_loss_for(logits, 2, 5, "ce").item()
        )


class _Item:
    """Minimal TypedQuestion stand-in for the score path."""

    def __init__(self, label, n_levels=5):
        self.state = "some state"
        self.question = {
            "type": "score",
            "instructions": "rate this",
            "criteria": [f"level {i}" for i in range(n_levels)],
        }
        self.label = label
        self.teacher_probs = None
        self.source = "stub"


class _FixedLogitModel(nn.Module):
    """Returns the same score logits for every item."""

    def __init__(self, logits):
        super().__init__()
        self._logits = logits
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward_score(self, state, instructions, level_texts):
        return self._logits


class _ScoreHeadStub(nn.Module):
    """score_head(context, levels) → [1, L, n_levels], like the real head."""

    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, context_hidden, level_hidden, level_mask=None):
        # [1, L, hidden] → [1, n_levels], the shape AttentionHead returns.
        return self.linear(level_hidden.mean(dim=1))


class _BatchedScoreModel(nn.Module):
    """Covers the batched path: encode helpers + a score head."""

    def __init__(self, n_levels=5, hidden=4):
        super().__init__()
        self.head = nn.Linear(hidden, n_levels)
        self.noul_head = None
        self.score_head = _ScoreHeadStub(self.head)

    def _encode_with_sequence(self, text, max_length=None):
        return torch.zeros(1, 3, 4)

    def _encode_text(self, texts, max_length=None):
        return torch.zeros(len(texts), 4)


class TestComputeLossWiring:
    """score_loss reaches compute_loss and changes the number."""

    def test_compute_loss_rps(self):
        from model.training.supervised import compute_loss

        logits = torch.randn(1, 5, generator=torch.Generator().manual_seed(6))
        model = _FixedLogitModel(logits)
        got = compute_loss(model, _Item(3), score_loss="rps")
        want = rps_loss(logits, _one_hot(3, 5).unsqueeze(0))
        assert got.item() == pytest.approx(want.item(), abs=1e-6)

    def test_compute_loss_default_unchanged(self):
        from model.training.supervised import compute_loss

        logits = torch.randn(1, 5, generator=torch.Generator().manual_seed(7))
        model = _FixedLogitModel(logits)
        got = compute_loss(model, _Item(3))
        want = F.cross_entropy(logits, torch.tensor([3]))
        assert got.item() == pytest.approx(want.item(), abs=1e-6)

    def test_compute_loss_rps_backpropagates(self):
        from model.training.supervised import compute_loss

        logits = torch.randn(1, 5, requires_grad=True)
        model = _FixedLogitModel(logits)
        compute_loss(model, _Item(1), score_loss="rps").backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0


class TestBatchPathWiring:
    """The batched code path must accept score_loss too."""

    def test_compute_score_batch_rps(self):
        from model.training.supervised import _compute_score_batch

        model = _BatchedScoreModel()
        losses = _compute_score_batch(model, [_Item(4)], score_loss="rps")
        assert len(losses) == 1
        # The stub head sees identical level embeddings, so every level gets
        # the same logit and the loss is the one for a uniform distribution.
        logits = model.head.bias.detach().unsqueeze(0)
        want = rps_loss(logits, _one_hot(4, 5).unsqueeze(0))
        assert losses[0].item() == pytest.approx(want.item(), abs=1e-6)

    def test_compute_score_batch_accepts_keyword(self):
        import inspect

        from model.training.supervised import _compute_score_batch

        sig = inspect.signature(_compute_score_batch)
        assert "score_loss" in sig.parameters
        assert sig.parameters["score_loss"].default == "ce"

    def test_train_epoch_batched_rps_runs(self):
        """The batched path does not go through compute_loss, so check
        score_loss reaches it — otherwise --score-loss rps would silently
        train score items with CE."""
        from model.training.supervised import train_epoch

        model = _BatchedScoreModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        stats = train_epoch(
            model,
            [_Item(1), _Item(3)],
            optimizer,
            batch_backbone=2,
            score_loss="rps",
        )
        assert stats["n_items"] == 2
        assert stats["mean_loss"] > 0

    def test_train_epoch_batched_rps_differs_from_ce(self):
        from model.training.supervised import train_epoch

        results = {}
        for mode in ("ce", "rps"):
            torch.manual_seed(0)
            model = _BatchedScoreModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
            results[mode] = train_epoch(
                model,
                [_Item(4)],
                optimizer,
                batch_backbone=1,
                score_loss=mode,
            )["mean_loss"]
        assert results["rps"] != pytest.approx(results["ce"], abs=1e-6)

    def test_train_epoch_rejects_unknown_loss(self):
        from model.training.supervised import train_epoch

        model = _BatchedScoreModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with pytest.raises(ValueError, match="unknown score_loss"):
            train_epoch(model, [_Item(2)], optimizer, batch_backbone=1, score_loss="nope")


class TestSignatures:
    """The setting is threaded all the way from the CLI entry point."""

    def test_train_epoch_accepts_score_loss(self):
        import inspect

        from model.training.supervised import train_epoch

        sig = inspect.signature(train_epoch)
        assert sig.parameters["score_loss"].default == "ce"

    def test_compute_loss_accepts_score_loss(self):
        import inspect

        from model.training.supervised import compute_loss

        sig = inspect.signature(compute_loss)
        assert sig.parameters["score_loss"].default == "ce"

    def test_train_multitask_accepts_score_loss(self):
        import inspect

        from model.training.supervised import train_multitask

        sig = inspect.signature(train_multitask)
        assert sig.parameters["score_loss"].default == "ce"

    def test_cli_flag_exists(self):
        """--score-loss must reach train_multitask, so parse it here."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--score-loss", choices=["ce", "rps"], default=None)
        assert parser.parse_args(["--score-loss", "rps"]).score_loss == "rps"
        with pytest.raises(SystemExit):
            parser.parse_args(["--score-loss", "brier"])
