"""Tests for MultitaskSampler."""

from data.format import TypedQuestion
from data.sampler import MultitaskSampler, SamplerConfig


def _make_items(source: str, q_type: str, n: int) -> list[TypedQuestion]:
    return [
        TypedQuestion(
            id=f"{source}-{i}",
            state=f"state {i}",
            question={"type": q_type, "instructions": "test"},
            label=True if q_type == "noul" else "a" if q_type == "choice" else 0.0,
            source=source,
            split="train",
        )
        for i in range(n)
    ]


class TestSamplerTypeBalance:
    def test_equal_ratios_produce_balanced_types(self):
        items = _make_items("s1", "noul", 100) + _make_items("s2", "choice", 100) + _make_items("s3", "score", 100)
        config = SamplerConfig(epoch_size=90)
        sampler = MultitaskSampler(items, config, seed=42)
        epoch = sampler.sample_epoch(0)
        types = {}
        for it in epoch:
            t = it.question["type"]
            types[t] = types.get(t, 0) + 1
        assert len(epoch) == 90
        for t in ("noul", "choice", "score"):
            assert 25 <= types.get(t, 0) <= 35, f"{t} count {types.get(t, 0)} out of expected range"

    def test_zero_ratio_excludes_type(self):
        items = _make_items("s1", "noul", 50) + _make_items("s2", "choice", 50) + _make_items("s3", "score", 50)
        config = SamplerConfig(type_ratios={"noul": 1.0, "choice": 1.0, "score": 0.0}, epoch_size=60)
        sampler = MultitaskSampler(items, config, seed=42)
        epoch = sampler.sample_epoch(0)
        types = {it.question["type"] for it in epoch}
        assert "score" not in types


class TestSamplerSourceCaps:
    def test_cap_limits_source_items(self):
        items = _make_items("big", "noul", 1000) + _make_items("small", "noul", 50)
        config = SamplerConfig(
            type_ratios={"noul": 1.0},
            source_caps={"big": 100},
            epoch_size=150,
        )
        sampler = MultitaskSampler(items, config, seed=42)
        epoch = sampler.sample_epoch(0)
        assert len(epoch) == 150


class TestSamplerReproducibility:
    def test_same_seed_same_epoch_produces_identical_output(self):
        items = _make_items("s1", "noul", 50) + _make_items("s2", "choice", 50)
        config = SamplerConfig(epoch_size=30)
        s1 = MultitaskSampler(items, config, seed=42)
        s2 = MultitaskSampler(items, config, seed=42)
        assert [it.id for it in s1.sample_epoch(0)] == [it.id for it in s2.sample_epoch(0)]

    def test_different_epochs_produce_different_output(self):
        items = _make_items("s1", "noul", 50) + _make_items("s2", "choice", 50)
        config = SamplerConfig(epoch_size=30)
        sampler = MultitaskSampler(items, config, seed=42)
        e0 = [it.id for it in sampler.sample_epoch(0)]
        e1 = [it.id for it in sampler.sample_epoch(1)]
        assert e0 != e1


class TestSamplerSourceWeighting:
    def test_higher_weight_increases_representation(self):
        items = _make_items("heavy", "noul", 100) + _make_items("light", "noul", 100)
        config = SamplerConfig(
            type_ratios={"noul": 1.0},
            source_weights={"heavy": 10.0, "light": 1.0},
            epoch_size=200,
        )
        sampler = MultitaskSampler(items, config, seed=42)
        epoch = sampler.sample_epoch(0)
        heavy_count = sum(1 for it in epoch if it.source == "heavy")
        assert heavy_count > 150, f"heavy should dominate: {heavy_count}/200"


class TestSamplerOversampling:
    def test_small_pool_oversampled(self):
        items = _make_items("tiny", "score", 5)
        config = SamplerConfig(type_ratios={"score": 1.0}, epoch_size=20)
        sampler = MultitaskSampler(items, config, seed=42)
        epoch = sampler.sample_epoch(0)
        assert len(epoch) == 20


class TestTemperatureSampling:
    def test_temperature_upweights_small_sources(self):
        items = _make_items("big", "noul", 500) + _make_items("small", "noul", 50)
        # Without temperature, equal manual weights → ~50/50 by weight.
        # With T=1.0 (proportional): big gets 500/550 ≈ 91%
        # With T=5.0 (near-uniform): weights are 500^0.2≈3.47 vs 50^0.2≈2.19
        # so small gets ~2.19/(3.47+2.19) ≈ 39% — much more than proportional 9%
        config_proportional = SamplerConfig(
            type_ratios={"noul": 1.0},
            sampling_temperature=1.0,
            epoch_size=1000,
        )
        config_balanced = SamplerConfig(
            type_ratios={"noul": 1.0},
            sampling_temperature=5.0,
            epoch_size=1000,
        )
        s_prop = MultitaskSampler(items, config_proportional, seed=42)
        s_bal = MultitaskSampler(items, config_balanced, seed=42)
        small_prop = sum(1 for it in s_prop.sample_epoch(0) if it.source == "small")
        small_bal = sum(1 for it in s_bal.sample_epoch(0) if it.source == "small")
        assert small_bal > small_prop, (
            f"higher T should upweight small sources: T=5.0 got {small_bal}, T=1.0 got {small_prop}"
        )

    def test_temperature_1_is_proportional(self):
        items = _make_items("big", "noul", 900) + _make_items("small", "noul", 100)
        config = SamplerConfig(
            type_ratios={"noul": 1.0},
            sampling_temperature=1.0,
            epoch_size=1000,
        )
        sampler = MultitaskSampler(items, config, seed=42)
        epoch = sampler.sample_epoch(0)
        big_count = sum(1 for it in epoch if it.source == "big")
        # T=1.0: weights proportional to size, so big≈900/1000=90%
        assert 850 <= big_count <= 950, f"T=1 should be ~proportional: big={big_count}/1000"

    def test_temperature_none_uses_manual_weights(self):
        items = _make_items("a", "noul", 100) + _make_items("b", "noul", 100)
        config = SamplerConfig(
            type_ratios={"noul": 1.0},
            source_weights={"a": 10.0, "b": 1.0},
            epoch_size=200,
        )
        sampler = MultitaskSampler(items, config, seed=42)
        epoch = sampler.sample_epoch(0)
        a_count = sum(1 for it in epoch if it.source == "a")
        assert a_count > 150, f"manual weights should apply: a={a_count}/200"

    def test_temperature_overrides_manual_weights(self):
        items = _make_items("big", "noul", 500) + _make_items("small", "noul", 500)
        config = SamplerConfig(
            type_ratios={"noul": 1.0},
            source_weights={"big": 10.0, "small": 1.0},
            sampling_temperature=1.0,
            epoch_size=200,
        )
        sampler = MultitaskSampler(items, config, seed=42)
        epoch = sampler.sample_epoch(0)
        big_count = sum(1 for it in epoch if it.source == "big")
        # T=1.0 with equal sizes: weights are equal (500^1 == 500^1), ~50/50
        # Manual weights (10:1) should be ignored
        assert 80 <= big_count <= 120, f"temperature should override manual weights: big={big_count}/200"

    def test_temperature_reproducible(self):
        items = _make_items("a", "noul", 100) + _make_items("b", "noul", 50)
        config = SamplerConfig(
            type_ratios={"noul": 1.0},
            sampling_temperature=0.5,
            epoch_size=100,
        )
        s1 = MultitaskSampler(items, config, seed=42)
        s2 = MultitaskSampler(items, config, seed=42)
        assert [it.id for it in s1.sample_epoch(0)] == [it.id for it in s2.sample_epoch(0)]
class TestDifficultyWeighting:
    def test_difficulty_weights_override_type_ratios(self):
        items = (
            _make_items("s1", "noul", 200)
            + _make_items("s2", "choice", 200)
            + _make_items("s3", "score", 200)
        )
        config = SamplerConfig(
            type_ratios={"noul": 1.0, "choice": 1.0, "score": 1.0},
            difficulty_weights={"noul": 0.5, "choice": 1.0, "score": 2.0},
            epoch_size=210,
        )
        sampler = MultitaskSampler(items, config, seed=42)
        epoch = sampler.sample_epoch(0)
        types: dict[str, int] = {}
        for it in epoch:
            t = it.question["type"]
            types[t] = types.get(t, 0) + 1
        assert len(epoch) == 210
        # score (weight 2.0) should get ~4x noul (weight 0.5)
        assert types["score"] > types["noul"] * 3, (
            f"score={types['score']} should be ~4x noul={types['noul']}"
        )
        # choice (weight 1.0) should be between noul and score
        assert types["choice"] > types["noul"], (
            f"choice={types['choice']} should exceed noul={types['noul']}"
        )

    def test_difficulty_weights_none_uses_type_ratios(self):
        items = (
            _make_items("s1", "noul", 200)
            + _make_items("s2", "choice", 200)
            + _make_items("s3", "score", 200)
        )
        config = SamplerConfig(
            type_ratios={"noul": 1.0, "choice": 1.0, "score": 1.0},
            difficulty_weights=None,
            epoch_size=90,
        )
        sampler = MultitaskSampler(items, config, seed=42)
        epoch = sampler.sample_epoch(0)
        types: dict[str, int] = {}
        for it in epoch:
            t = it.question["type"]
            types[t] = types.get(t, 0) + 1
        assert len(epoch) == 90
        # Equal type_ratios → roughly equal counts
        for t in ("noul", "choice", "score"):
            assert 25 <= types.get(t, 0) <= 35, (
                f"{t} count {types.get(t, 0)} out of expected range"
            )

    def test_difficulty_weights_with_missing_type(self):
        # Only noul and choice items, but difficulty_weights includes score
        items = (
            _make_items("s1", "noul", 100)
            + _make_items("s2", "choice", 100)
        )
        config = SamplerConfig(
            difficulty_weights={"noul": 0.5, "choice": 1.0, "score": 2.0},
            epoch_size=90,
        )
        sampler = MultitaskSampler(items, config, seed=42)
        epoch = sampler.sample_epoch(0)
        types: dict[str, int] = {}
        for it in epoch:
            t = it.question["type"]
            types[t] = types.get(t, 0) + 1
        assert len(epoch) == 90
        # No score items exist, so all items are noul or choice
        assert "score" not in types
        # choice (weight 1.0) should get ~2x noul (weight 0.5)
        assert types["choice"] > types["noul"] * 1.5, (
            f"choice={types['choice']} should be ~2x noul={types['noul']}"
        )


class TestSamplerEdgeCases:
    def test_empty_items(self):
        config = SamplerConfig(epoch_size=10)
        sampler = MultitaskSampler([], config, seed=42)
        assert sampler.sample_epoch(0) == []

    def test_single_type(self):
        items = _make_items("s1", "choice", 50)
        config = SamplerConfig(epoch_size=20)
        sampler = MultitaskSampler(items, config, seed=42)
        epoch = sampler.sample_epoch(0)
        assert len(epoch) == 20
        assert all(it.question["type"] == "choice" for it in epoch)
