"""Tests for decision head architecture (MLPProjector, NoulHead MLP, UnifiedHead, param counts)."""

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from model.src.heads import (
    AttentionHead,
    ChoiceHead,
    MLPProjector,
    NoulHead,
    ScoreHead,
    UnifiedHead,
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


class TestUnifiedHead:
    """Tests for UnifiedHead: single AttentionHead handling all question types."""

    def test_forward_matches_choice_shape(self):
        """forward() should produce [batch, n_options] logits like ChoiceHead."""
        head = UnifiedHead(768, rank=64)
        ctx = torch.randn(1, 10, 768)
        opts = torch.randn(1, 4, 768)
        logits = head(ctx, opts)
        assert logits.shape == (1, 4)

    def test_forward_matches_score_shape(self):
        """forward() should produce [batch, n_levels] logits like ScoreHead."""
        head = UnifiedHead(768, rank=64)
        ctx = torch.randn(1, 10, 768)
        lvls = torch.randn(1, 5, 768)
        logits = head(ctx, lvls)
        assert logits.shape == (1, 5)

    def test_noul_logit_shape(self):
        """noul_logit() should return [batch, 1] compatible with BCE loss."""
        head = UnifiedHead(768, rank=64)
        ctx = torch.randn(1, 10, 768)
        opts = torch.randn(1, 2, 768)  # false, true
        logit = head.noul_logit(ctx, opts)
        assert logit.shape == (1, 1)

    def test_noul_logit_batched(self):
        """noul_logit() should work with batch > 1."""
        head = UnifiedHead(768, rank=64)
        ctx = torch.randn(3, 10, 768)
        opts = torch.randn(3, 2, 768)
        logit = head.noul_logit(ctx, opts)
        assert logit.shape == (3, 1)

    def test_predict_noul_returns_probability(self):
        """predict_noul() should return values in [0, 1]."""
        head = UnifiedHead(768, rank=64)
        ctx = torch.randn(2, 10, 768)
        opts = torch.randn(2, 2, 768)
        probs = head.predict_noul(ctx, opts)
        assert probs.shape == (2,)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_noul_sigmoid_softmax_equivalence(self):
        """sigmoid(logit_true - logit_false) should equal softmax(logits)[true].

        This is the mathematical identity that makes the unified noul approach
        exactly equivalent to softmax over two options.  Uses eval mode to
        disable dropout so both forward passes produce identical logits.
        """
        head = UnifiedHead(768, rank=64)
        head.eval()
        ctx = torch.randn(4, 10, 768)
        opts = torch.randn(4, 2, 768)

        with torch.no_grad():
            # Get the full 2-option logits and the noul logit
            logits = head(ctx, opts)  # [4, 2]
            noul_logit = head.noul_logit(ctx, opts)  # [4, 1]

        # softmax(logits)[:, 1] should equal sigmoid(noul_logit)
        softmax_true = torch.softmax(logits, dim=-1)[:, 1]
        sigmoid_noul = torch.sigmoid(noul_logit).squeeze(-1)

        assert torch.allclose(softmax_true, sigmoid_noul, atol=1e-6)

    def test_predict_choice_returns_distribution(self):
        """predict_choice() should return probabilities summing to 1."""
        head = UnifiedHead(768, rank=64)
        ctx = torch.randn(1, 10, 768)
        opts = torch.randn(1, 4, 768)
        probs = head.predict_choice(ctx, opts)
        assert probs.shape == (1, 4)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-6)
        assert (probs >= 0).all()

    def test_predict_score_returns_expected_value(self):
        """predict_score() should return probabilities and expected value."""
        head = UnifiedHead(768, rank=64)
        ctx = torch.randn(1, 10, 768)
        lvls = torch.randn(1, 5, 768)
        probs, expected = head.predict_score(ctx, lvls)
        assert probs.shape == (1, 5)
        assert expected.shape == (1,)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-6)
        # Expected value should be in valid range [0, n_levels-1]
        assert expected.item() >= 0
        assert expected.item() <= 4

    def test_param_count_less_than_separate_heads(self):
        """UnifiedHead should have fewer params than NoulHead + ChoiceHead + ScoreHead."""
        unified = UnifiedHead(768, rank=64)
        noul = NoulHead(768, rank=64)
        choice = ChoiceHead(768, rank=64)
        score = ScoreHead(768, rank=64)

        unified_params = sum(p.numel() for p in unified.parameters())
        separate_params = (
            sum(p.numel() for p in noul.parameters())
            + sum(p.numel() for p in choice.parameters())
            + sum(p.numel() for p in score.parameters())
        )
        assert unified_params < separate_params

    def test_drop_in_replacement_for_choice_head(self):
        """UnifiedHead should be usable wherever ChoiceHead is expected."""
        unified = UnifiedHead(768, rank=64)
        ctx = torch.randn(1, 10, 768)
        opts = torch.randn(1, 4, 768)
        mask = torch.tensor([[True, True, True, False]])

        # Should accept same arguments as ChoiceHead.forward()
        logits = unified(ctx, opts, mask)
        assert logits.shape == (1, 4)
        # Masked position should be -inf
        assert logits[0, 3].item() == float("-inf")

    def test_drop_in_replacement_for_score_head(self):
        """UnifiedHead should be usable wherever ScoreHead is expected."""
        unified = UnifiedHead(768, rank=64)
        ctx = torch.randn(1, 10, 768)
        lvls = torch.randn(1, 5, 768)

        # Should accept same arguments as ScoreHead.forward()
        logits = unified(ctx, lvls)
        assert logits.shape == (1, 5)

    def test_gradient_flows_through_all_types(self):
        """Gradients from noul, choice, and score should all reach attention weights."""
        head = UnifiedHead(768, rank=64)
        ctx = torch.randn(1, 10, 768, requires_grad=True)

        # Noul
        noul_logit = head.noul_logit(ctx, torch.randn(1, 2, 768))
        noul_logit.sum().backward()
        grad_noul = head.attention.query.weight.grad.clone()
        head.attention.query.weight.grad.zero_()

        # Choice
        ctx2 = torch.randn(1, 10, 768, requires_grad=True)
        choice_logits = head(ctx2, torch.randn(1, 4, 768))
        choice_logits.sum().backward()
        grad_choice = head.attention.query.weight.grad.clone()
        head.attention.query.weight.grad.zero_()

        # Score
        ctx3 = torch.randn(1, 10, 768, requires_grad=True)
        score_logits = head(ctx3, torch.randn(1, 5, 768))
        score_logits.sum().backward()
        grad_score = head.attention.query.weight.grad.clone()

        assert grad_noul.abs().sum() > 0
        assert grad_choice.abs().sum() > 0
        assert grad_score.abs().sum() > 0

    def test_rival_aware_flag_propagates(self):
        """UnifiedHead should pass rival_aware to AttentionHead."""
        head = UnifiedHead(768, rank=64, rival_aware=True)
        assert head.attention.rival_aware is True
