"""Tests for GPU configuration auto-detection and override logic."""

import pytest

torch = pytest.importorskip("torch")

from model.src.gpu_config import GPUConfig, estimate_model_billions, print_gpu_info


class TestGPUConfigAutoDetect:
    """Test GPUConfig.auto_detect() returns valid configs on any hardware."""

    def test_returns_gpu_config_instance(self):
        cfg = GPUConfig.auto_detect()
        assert isinstance(cfg, GPUConfig)

    def test_max_length_is_positive(self):
        cfg = GPUConfig.auto_detect()
        assert cfg.max_length > 0

    def test_fields_are_bool(self):
        cfg = GPUConfig.auto_detect()
        assert isinstance(cfg.flash_attention, bool)
        assert isinstance(cfg.bf16_compute, bool)
        assert isinstance(cfg.gradient_checkpointing, bool)

    def test_cpu_fallback(self):
        """When CUDA is unavailable, auto_detect should return safe CPU defaults."""
        if torch.cuda.is_available():
            pytest.skip("Test only meaningful without GPU")
        cfg = GPUConfig.auto_detect()
        assert cfg.flash_attention is False
        assert cfg.bf16_compute is False
        assert cfg.max_length == 512
        assert cfg.gradient_checkpointing is False

    def test_larger_model_gets_shorter_max_length(self):
        """A 4B model should get a shorter (or equal) max_length than a 0.6B model."""
        cfg_small = GPUConfig.auto_detect(model_params_billions=0.6)
        cfg_large = GPUConfig.auto_detect(model_params_billions=4.0)
        assert cfg_large.max_length <= cfg_small.max_length


class TestGPUConfigOverrides:
    """Test apply_overrides() selectively replaces auto-detected values."""

    def test_override_flash_attention(self):
        base = GPUConfig(
            flash_attention=True, bf16_compute=True,
            max_length=8192, gradient_checkpointing=False,
        )
        overridden = base.apply_overrides(flash_attention=False)
        assert overridden.flash_attention is False
        # Other fields unchanged
        assert overridden.bf16_compute is True
        assert overridden.max_length == 8192
        assert overridden.gradient_checkpointing is False

    def test_override_bf16(self):
        base = GPUConfig(
            flash_attention=False, bf16_compute=False,
            max_length=4096, gradient_checkpointing=False,
        )
        overridden = base.apply_overrides(bf16=True)
        assert overridden.bf16_compute is True

    def test_override_max_length(self):
        base = GPUConfig(
            flash_attention=True, bf16_compute=True,
            max_length=8192, gradient_checkpointing=False,
        )
        overridden = base.apply_overrides(max_length=2048)
        assert overridden.max_length == 2048

    def test_override_gradient_checkpointing(self):
        base = GPUConfig(
            flash_attention=True, bf16_compute=True,
            max_length=8192, gradient_checkpointing=False,
        )
        overridden = base.apply_overrides(gradient_checkpointing=True)
        assert overridden.gradient_checkpointing is True

    def test_none_overrides_preserve_original(self):
        base = GPUConfig(
            flash_attention=True, bf16_compute=False,
            max_length=4096, gradient_checkpointing=True,
        )
        overridden = base.apply_overrides()
        assert overridden == base

    def test_multiple_overrides(self):
        base = GPUConfig(
            flash_attention=True, bf16_compute=True,
            max_length=8192, gradient_checkpointing=False,
        )
        overridden = base.apply_overrides(
            flash_attention=False, max_length=2048, gradient_checkpointing=True,
        )
        assert overridden.flash_attention is False
        assert overridden.bf16_compute is True  # unchanged
        assert overridden.max_length == 2048
        assert overridden.gradient_checkpointing is True


class TestEstimateModelBillions:
    """Test model size estimation from model name strings."""

    def test_qwen3_reranker_4b(self):
        assert estimate_model_billions("Qwen/Qwen3-Reranker-4B") == 4.0

    def test_qwen3_reranker_06b(self):
        assert estimate_model_billions("Qwen/Qwen3-Reranker-0.6B") == 0.6

    def test_qwen35_08b(self):
        assert estimate_model_billions("Qwen/Qwen3.5-0.8B-Base") == 0.8

    def test_model_with_m_suffix(self):
        result = estimate_model_billions("some-org/ettin-150m")
        assert result == pytest.approx(0.15)

    def test_unknown_model_defaults_to_06(self):
        assert estimate_model_billions("some-org/mystery-model") == 0.6

    def test_case_insensitive(self):
        assert estimate_model_billions("Qwen/QWEN3-RERANKER-4B") == 4.0


class TestPrintGPUInfo:
    """Test that print_gpu_info() runs without error."""

    def test_prints_without_error(self, capsys):
        cfg = GPUConfig(
            flash_attention=False, bf16_compute=True,
            max_length=4096, gradient_checkpointing=False,
        )
        print_gpu_info(cfg)
        captured = capsys.readouterr()
        assert "flash_attention=False" in captured.out
        assert "bf16=True" in captured.out
        assert "max_length=4096" in captured.out


class TestGPUConfigWithDecisionModel:
    """Test that DecisionModel accepts and uses GPUConfig correctly."""

    def test_decision_model_accepts_gpu_config(self):
        """DecisionModel.__init__ should accept gpu_config parameter."""
        from unittest.mock import MagicMock

        # Create minimal mock backbone
        backbone = MagicMock()
        backbone.config.hidden_size = 768
        backbone.config.is_decoder = True
        backbone.config.max_position_embeddings = 32768
        # Use lambda to return fresh iterators (backbone.parameters() is
        # called multiple times during __init__)
        _param = torch.nn.Parameter(torch.zeros(1))
        backbone.parameters = lambda: iter([_param])
        # is_hybrid_model checks layer_types
        backbone.config.layer_types = []

        tokenizer = MagicMock()

        gpu_cfg = GPUConfig(
            flash_attention=False,
            bf16_compute=False,
            max_length=2048,
            gradient_checkpointing=False,
        )

        from model.src.decision_model import DecisionModel

        model = DecisionModel(
            backbone=backbone,
            tokenizer=tokenizer,
            gpu_config=gpu_cfg,
        )
        assert model.gpu_config is gpu_cfg
        assert model.max_length == 2048

    def test_decision_model_explicit_max_length_overrides_gpu_config(self):
        """Explicit max_length arg should take priority over gpu_config."""
        from unittest.mock import MagicMock

        backbone = MagicMock()
        backbone.config.hidden_size = 768
        backbone.config.is_decoder = True
        backbone.config.max_position_embeddings = 32768
        _param = torch.nn.Parameter(torch.zeros(1))
        backbone.parameters = lambda: iter([_param])
        backbone.config.layer_types = []

        tokenizer = MagicMock()

        gpu_cfg = GPUConfig(
            flash_attention=False,
            bf16_compute=False,
            max_length=8192,
            gradient_checkpointing=False,
        )

        from model.src.decision_model import DecisionModel

        model = DecisionModel(
            backbone=backbone,
            tokenizer=tokenizer,
            max_length=1024,
            gpu_config=gpu_cfg,
        )
        # Explicit max_length should win
        assert model.max_length == 1024
