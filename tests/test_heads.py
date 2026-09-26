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


class TestSharedAttention:
    """Tests for shared AttentionHead between ChoiceHead and ScoreHead."""

    def test_shared_attention_same_instance(self):
        """ChoiceHead and ScoreHead should reference the exact same AttentionHead."""
        shared = AttentionHead(768, rank=64)
        choice = ChoiceHead(768, attention=shared)
        score = ScoreHead(768, attention=shared)
        assert choice.attention is score.attention

    def test_shared_attention_param_count(self):
        """Shared heads should have fewer total params than separate heads."""
        shared_attn = AttentionHead(768, rank=64)
        choice_shared = ChoiceHead(768, attention=shared_attn)
        score_shared = ScoreHead(768, attention=shared_attn)
        choice_sep = ChoiceHead(768, rank=64)
        score_sep = ScoreHead(768, rank=64)

        # Use set() on data_ptr to deduplicate shared params
        shared_params = sum(
            p.numel()
            for p in {
                id(p): p
                for p in list(choice_shared.parameters())
                + list(score_shared.parameters())
            }.values()
        )
        sep_params = sum(p.numel() for p in choice_sep.parameters()) + sum(
            p.numel() for p in score_sep.parameters()
        )
        assert shared_params < sep_params

    def test_shared_attention_output_shapes(self):
        """Both heads should produce correct output shapes with shared attention."""
        shared = AttentionHead(768, rank=64)
        choice = ChoiceHead(768, attention=shared)
        score = ScoreHead(768, attention=shared)
        ctx = torch.randn(1, 10, 768)
        opts = torch.randn(1, 4, 768)
        lvls = torch.randn(1, 5, 768)
        assert choice(ctx, opts).shape == (1, 4)
        assert score(ctx, lvls).shape == (1, 5)

    def test_shared_attention_gradient_flows_to_both(self):
        """Gradients from both heads should reach the shared attention weights."""
        shared = AttentionHead(768, rank=64)
        choice = ChoiceHead(768, attention=shared)
        score = ScoreHead(768, attention=shared)

        # Choice forward + backward
        ctx = torch.randn(1, 10, 768, requires_grad=True)
        choice_out = choice(ctx, torch.randn(1, 4, 768))
        choice_out.sum().backward()
        grad1 = shared.query.weight.grad.clone()
        shared.query.weight.grad.zero_()

        # Score forward + backward
        ctx2 = torch.randn(1, 10, 768, requires_grad=True)
        score_out = score(ctx2, torch.randn(1, 5, 768))
        score_out.sum().backward()
        grad2 = shared.query.weight.grad.clone()

        # Both should produce non-zero gradients on shared params
        assert grad1.abs().sum() > 0
        assert grad2.abs().sum() > 0

    def test_no_duplicate_params_in_module(self):
        """PyTorch should not double-count shared params in nn.Module.parameters()."""
        shared = AttentionHead(768, rank=64)
        choice = ChoiceHead(768, attention=shared)
        score = ScoreHead(768, attention=shared)

        # Build a parent module containing both heads
        parent = nn.Module()
        parent.choice_head = choice
        parent.score_head = score

        # parameters() should deduplicate shared attention params
        param_ids = [id(p) for p in parent.parameters()]
        assert len(param_ids) == len(set(param_ids)), "Duplicate parameters detected"

    def test_backward_compat_no_shared(self):
        """Default construction (no shared attention) should create independent heads."""
        choice = ChoiceHead(768, rank=64)
        score = ScoreHead(768, rank=64)
        assert choice.attention is not score.attention


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
