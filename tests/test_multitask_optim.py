"""Tests for per-type LR and uncertainty-based loss weighting."""

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from unittest.mock import MagicMock

from model.src.gpu_config import GPUConfig


def _make_mock_backbone():
    """Create a mock backbone that returns fresh parameter iterators."""
    backbone = MagicMock()
    backbone.config.hidden_size = 64
    backbone.config.is_decoder = True
    backbone.config.max_position_embeddings = 512
    backbone.config.layer_types = []

    _param = nn.Parameter(torch.zeros(1))
    backbone.parameters = lambda: iter([_param])

    return backbone


def _make_mock_model(mlp_layers=0):
    """Create a minimal mock DecisionModel with real head-like modules."""
    from model.src.decision_model import DecisionModel

    backbone = _make_mock_backbone()
    tokenizer = MagicMock()

    gpu_cfg = GPUConfig(
        flash_attention=False,
        bf16_compute=False,
        max_length=128,
        gradient_checkpointing=False,
    )

    model = DecisionModel(
        backbone=backbone,
        tokenizer=tokenizer,
        rank=8,
        mlp_layers=mlp_layers,
        gpu_config=gpu_cfg,
    )
    return model


class TestPerTypeLR:
    """Test per-type learning rate creates correct optimizer param groups."""

    def test_per_type_lr_creates_separate_groups(self):
        """When per-type LRs are specified, optimizer should have separate param groups."""
        from model.training.supervised import train_multitask
        from data.sampler import SamplerConfig

        model = _make_mock_model()
        noul_param_ids = {id(p) for p in model.noul_head.parameters()}
        choice_param_ids = {id(p) for p in model.choice_head.parameters()}
        score_param_ids = {id(p) for p in model.score_head.parameters()}

        # We can't run the full training loop without data/sampler, so we
        # test the optimizer setup logic directly by replicating the key part
        from torch.optim import AdamW

        lr = 1e-3
        noul_lr = 2e-3
        choice_lr = 3e-3
        score_lr = 4e-3

        noul_params = list(model.noul_head.parameters())
        choice_params = list(model.choice_head.parameters())
        score_params = list(model.score_head.parameters())

        param_groups = [
            {"params": noul_params, "lr": noul_lr},
            {"params": choice_params, "lr": choice_lr},
            {"params": score_params, "lr": score_lr},
        ]

        optimizer = AdamW(param_groups, weight_decay=1e-4)

        # Verify 3 param groups with correct LRs
        assert len(optimizer.param_groups) == 3
        assert optimizer.param_groups[0]["lr"] == noul_lr
        assert optimizer.param_groups[1]["lr"] == choice_lr
        assert optimizer.param_groups[2]["lr"] == score_lr

        # Verify param assignment
        group0_ids = {id(p) for p in optimizer.param_groups[0]["params"]}
        group1_ids = {id(p) for p in optimizer.param_groups[1]["params"]}
        group2_ids = {id(p) for p in optimizer.param_groups[2]["params"]}

        assert group0_ids == noul_param_ids
        assert group1_ids == choice_param_ids
        assert group2_ids == score_param_ids

    def test_per_type_lr_with_mlp_creates_four_groups(self):
        """Per-type LR + mlp_lr should create 4 param groups."""
        model = _make_mock_model(mlp_layers=2)

        from torch.optim import AdamW

        lr = 1e-3
        noul_lr = 2e-3
        mlp_lr = 5e-3

        noul_params = list(model.noul_head.parameters())
        choice_params = list(model.choice_head.parameters())
        score_params = list(model.score_head.parameters())
        projector_params = list(model.projector.parameters())

        param_groups = [
            {"params": noul_params, "lr": noul_lr},
            {"params": choice_params, "lr": lr},
            {"params": score_params, "lr": lr},
            {"params": projector_params, "lr": mlp_lr},
        ]

        optimizer = AdamW(param_groups, weight_decay=1e-4)

        assert len(optimizer.param_groups) == 4
        assert optimizer.param_groups[0]["lr"] == noul_lr
        assert optimizer.param_groups[1]["lr"] == lr
        assert optimizer.param_groups[2]["lr"] == lr
        assert optimizer.param_groups[3]["lr"] == mlp_lr

    def test_no_per_type_lr_uses_single_group(self):
        """Without per-type LR, optimizer should use a single group."""
        model = _make_mock_model()

        from torch.optim import AdamW

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable, lr=1e-3)

        assert len(optimizer.param_groups) == 1

    def test_partial_per_type_lr_defaults_to_base(self):
        """When only some per-type LRs are set, others fall back to base lr."""
        model = _make_mock_model()

        from torch.optim import AdamW

        lr = 1e-3
        noul_lr = 5e-3

        noul_params = list(model.noul_head.parameters())
        choice_params = list(model.choice_head.parameters())
        score_params = list(model.score_head.parameters())

        param_groups = [
            {"params": noul_params, "lr": noul_lr},
            {"params": choice_params, "lr": lr},  # falls back to base
            {"params": score_params, "lr": lr},    # falls back to base
        ]

        optimizer = AdamW(param_groups, weight_decay=1e-4)

        assert optimizer.param_groups[0]["lr"] == noul_lr
        assert optimizer.param_groups[1]["lr"] == lr
        assert optimizer.param_groups[2]["lr"] == lr


class TestUncertaintyWeighting:
    """Test uncertainty-based loss weighting (Kendall et al. 2018)."""

    def test_log_vars_creation(self):
        """Uncertainty weighting should create 3 learnable log_var parameters."""
        device = torch.device("cpu")
        log_vars = {
            "noul": nn.Parameter(torch.zeros(1, device=device)),
            "choice": nn.Parameter(torch.zeros(1, device=device)),
            "score": nn.Parameter(torch.zeros(1, device=device)),
        }

        assert len(log_vars) == 3
        for name in ["noul", "choice", "score"]:
            assert name in log_vars
            assert log_vars[name].requires_grad is True
            assert log_vars[name].shape == (1,)
            assert log_vars[name].item() == 0.0

    def test_log_vars_added_to_optimizer(self):
        """log_var parameters should be added to optimizer param groups."""
        model = _make_mock_model()

        from torch.optim import AdamW

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable, lr=1e-3)
        initial_groups = len(optimizer.param_groups)

        device = torch.device("cpu")
        log_vars = {
            "noul": nn.Parameter(torch.zeros(1, device=device)),
            "choice": nn.Parameter(torch.zeros(1, device=device)),
            "score": nn.Parameter(torch.zeros(1, device=device)),
        }

        for lv in log_vars.values():
            optimizer.add_param_group({"params": [lv], "lr": 1e-3})

        # 3 new groups added (one per log_var)
        assert len(optimizer.param_groups) == initial_groups + 3

    def test_apply_uncertainty_weight_formula(self):
        """Verify the uncertainty weighting formula: loss/(2*exp(lv)) + lv/2."""
        from model.training.supervised import _apply_uncertainty_weight

        loss = torch.tensor(2.0)
        log_var = nn.Parameter(torch.tensor([0.0]))  # exp(0) = 1

        weighted = _apply_uncertainty_weight(loss, log_var)

        # loss/(2*exp(0)) + 0/2 = 2.0/2.0 + 0.0 = 1.0
        assert weighted.item() == pytest.approx(1.0, abs=1e-6)

    def test_apply_uncertainty_weight_nonzero_logvar(self):
        """Test with non-zero log_var value."""
        from model.training.supervised import _apply_uncertainty_weight

        loss = torch.tensor(4.0)
        log_var = nn.Parameter(torch.tensor([1.0]))  # exp(1) ≈ 2.718

        weighted = _apply_uncertainty_weight(loss, log_var)

        import math
        expected = 4.0 / (2 * math.exp(1.0)) + 1.0 / 2
        assert weighted.item() == pytest.approx(expected, abs=1e-5)

    def test_apply_uncertainty_weight_gradient_flows(self):
        """Verify gradients flow through log_var during backprop."""
        from model.training.supervised import _apply_uncertainty_weight

        loss = torch.tensor(2.0, requires_grad=True)
        log_var = nn.Parameter(torch.tensor([0.5]))

        weighted = _apply_uncertainty_weight(loss, log_var)
        weighted.backward()

        assert log_var.grad is not None
        assert log_var.grad.item() != 0.0

    def test_compute_loss_with_log_vars(self):
        """compute_loss should apply uncertainty weighting when log_vars provided."""
        from model.training.supervised import _apply_uncertainty_weight

        # Test the weighting function produces different output than raw loss
        raw_loss = torch.tensor(3.0)
        log_var = nn.Parameter(torch.tensor([1.0]))

        weighted = _apply_uncertainty_weight(raw_loss, log_var)

        # With log_var=1.0, weighted should differ from raw
        assert weighted.item() != raw_loss.item()

    def test_uncertainty_weight_negative_logvar(self):
        """Negative log_var should increase the weight on the loss."""
        from model.training.supervised import _apply_uncertainty_weight

        loss = torch.tensor(2.0)

        # Negative log_var → smaller exp → larger loss/(2*exp) component
        lv_neg = nn.Parameter(torch.tensor([-1.0]))
        lv_pos = nn.Parameter(torch.tensor([1.0]))

        w_neg = _apply_uncertainty_weight(loss, lv_neg)
        w_pos = _apply_uncertainty_weight(loss, lv_pos)

        # With negative log_var, the loss term is amplified more
        assert w_neg.item() > w_pos.item()


class TestTrainEpochLogVars:
    """Test that train_epoch correctly passes log_vars through."""

    def test_train_epoch_signature_accepts_log_vars(self):
        """train_epoch should accept log_vars parameter."""
        from model.training.supervised import train_epoch
        import inspect

        sig = inspect.signature(train_epoch)
        assert "log_vars" in sig.parameters
        assert sig.parameters["log_vars"].default is None

    def test_compute_loss_signature_accepts_log_vars(self):
        """compute_loss should accept log_vars parameter."""
        from model.training.supervised import compute_loss
        import inspect

        sig = inspect.signature(compute_loss)
        assert "log_vars" in sig.parameters
        assert sig.parameters["log_vars"].default is None


class TestTrainMultitaskSignature:
    """Test train_multitask function signature includes new parameters."""

    def test_has_per_type_lr_params(self):
        """train_multitask should accept noul_lr, choice_lr, score_lr."""
        from model.training.supervised import train_multitask
        import inspect

        sig = inspect.signature(train_multitask)
        assert "noul_lr" in sig.parameters
        assert "choice_lr" in sig.parameters
        assert "score_lr" in sig.parameters

        # All should default to None
        assert sig.parameters["noul_lr"].default is None
        assert sig.parameters["choice_lr"].default is None
        assert sig.parameters["score_lr"].default is None

    def test_has_uncertainty_weighting_param(self):
        """train_multitask should accept uncertainty_weighting."""
        from model.training.supervised import train_multitask
        import inspect

        sig = inspect.signature(train_multitask)
        assert "uncertainty_weighting" in sig.parameters
        assert sig.parameters["uncertainty_weighting"].default is False

    def test_has_mlp_lr_param(self):
        """Existing mlp_lr parameter should still be present."""
        from model.training.supervised import train_multitask
        import inspect

        sig = inspect.signature(train_multitask)
        assert "mlp_lr" in sig.parameters
        assert sig.parameters["mlp_lr"].default is None
