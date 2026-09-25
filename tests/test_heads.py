"""Tests for decision head architecture (MLPProjector, NoulHead MLP, param counts)."""

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from model.src.heads import (
    AttentionHead,
    ChoiceHead,
    MLPProjector,
    NoulHead,
    ScoreHead,
)


class TestMLPProjector:
    def test_single_layer_preserves_shape(self):
        proj = MLPProjector(768, num_layers=1)
        x = torch.randn(2, 768)
        out = proj(x)
        assert out.shape == (2, 768)

    def test_multi_layer_preserves_shape(self):
        proj = MLPProjector(768, num_layers=3)
        x = torch.randn(2, 768)
        out = proj(x)
        assert out.shape == (2, 768)

    def test_custom_hidden_size(self):
        proj = MLPProjector(768, hidden_size=1024, num_layers=2)
        x = torch.randn(2, 768)
        out = proj(x)
        assert out.shape == (2, 768)

    def test_3d_input(self):
        proj = MLPProjector(768, num_layers=2)
        x = torch.randn(1, 10, 768)
        out = proj(x)
        assert out.shape == (1, 10, 768)

    def test_param_count_single_layer(self):
        proj = MLPProjector(768, num_layers=1)
        params = sum(p.numel() for p in proj.parameters())
        assert params == 768 * 768 + 768

    def test_param_count_two_layers(self):
        proj = MLPProjector(768, hidden_size=1024, num_layers=2)
        params = sum(p.numel() for p in proj.parameters())
        expected = (768 * 1024 + 1024) + (1024 * 768 + 768)
        assert params == expected


class TestNoulHeadMLP:
    def test_legacy_mode(self):
        head = NoulHead(768)
        x = torch.randn(2, 768)
        out = head(x)
        assert out.shape == (2, 1)

    def test_mlp_mode(self):
        head = NoulHead(768, rank=64)
        x = torch.randn(2, 768)
        out = head(x)
        assert out.shape == (2, 1)

    def test_legacy_param_count(self):
        head = NoulHead(768)
        params = sum(p.numel() for p in head.parameters())
        assert params == 768 + 1

    def test_mlp_param_count(self):
        head = NoulHead(768, rank=64)
        params = sum(p.numel() for p in head.parameters())
        expected = (768 * 64 + 64) + (64 + 1)
        assert params == expected

    def test_predict_returns_probability(self):
        head = NoulHead(768, rank=64)
        x = torch.randn(2, 768)
        probs = head.predict(x)
        assert probs.shape == (2,)
        assert (probs >= 0).all() and (probs <= 1).all()


class TestHeadBackwardCompat:
    def test_choice_head_unchanged(self):
        head = ChoiceHead(768, rank=64)
        ctx = torch.randn(1, 10, 768)
        opt = torch.randn(1, 4, 768)
        logits = head(ctx, opt)
        assert logits.shape == (1, 4)

    def test_score_head_unchanged(self):
        head = ScoreHead(768, rank=64)
        ctx = torch.randn(1, 10, 768)
        lvl = torch.randn(1, 5, 768)
        logits = head(ctx, lvl)
        assert logits.shape == (1, 5)

    def test_gradient_flows(self):
        proj = MLPProjector(768, num_layers=2)
        head = NoulHead(768, rank=64)
        x = torch.randn(2, 768, requires_grad=True)
        projected = proj(x)
        logits = head(projected)
        loss = logits.sum()
        loss.backward()
        assert x.grad is not None
        assert all(p.grad is not None for p in proj.parameters())
        assert all(p.grad is not None for p in head.parameters())
