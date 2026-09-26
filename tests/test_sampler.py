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



class TestSyntheticReplayRatio:
    """Tests for synthetic_ratio controlling benchmark/synthetic split."""

    def _build_mixed_items(self):
        """Build items from benchmark and synthetic sources."""
        return (
            _make_items("mnli", "noul", 200)
            + _make_items("sst2", "noul", 200)
            + _make_items("syn_domain1", "noul", 100)
            + _make_items("syn_domain2", "noul", 100)
            + _make_items("arc", "choice", 200)
            + _make_items("syn_domain3", "choice", 100)
        )

    def test_synthetic_ratio_controls_proportion(self):
        """With ratio=0.4, ~40% of items should come from synthetic sources."""
        items = self._build_mixed_items()
        syn_names = {"syn_domain1", "syn_domain2", "syn_domain3"}
        config = SamplerConfig(
            synthetic_sources=syn_names,
            synthetic_ratio=0.4,
            epoch_size=600,
        )
        sampler = MultitaskSampler(items, config, seed=42)
        epoch = sampler.sample_epoch(0)
        assert len(epoch) == 600

        syn_count = sum(1 for it in epoch if it.source in syn_names)
        ratio = syn_count / len(epoch)
        # Allow +-5% tolerance due to integer rounding per type
        assert 0.30 <= ratio <= 0.50, f"synthetic ratio {ratio:.2f} outside [0.30, 0.50]"

    def test_synthetic_ratio_none_no_effect(self):
        """Without ratio set, sampling works as before (no split)."""
        items = self._build_mixed_items()
        config_with = SamplerConfig(
            synthetic_sources={"syn_domain1", "syn_domain2", "syn_domain3"},
            synthetic_ratio=None,
            epoch_size=300,
        )
        config_without = SamplerConfig(epoch_size=300)
        s1 = MultitaskSampler(items, config_with, seed=42)
        s2 = MultitaskSampler(items, config_without, seed=42)
        e1 = [it.id for it in s1.sample_epoch(0)]
        e2 = [it.id for it in s2.sample_epoch(0)]
        assert e1 == e2

    def test_synthetic_ratio_zero_excludes_synthetic(self):
        """With ratio=0.0, no synthetic items should appear."""
        items = self._build_mixed_items()
        syn_names = {"syn_domain1", "syn_domain2", "syn_domain3"}
        config = SamplerConfig(
            synthetic_sources=syn_names,
            synthetic_ratio=0.0,
            epoch_size=300,
        )
        sampler = MultitaskSampler(items, config, seed=42)
        epoch = sampler.sample_epoch(0)
        assert len(epoch) == 300
        syn_count = sum(1 for it in epoch if it.source in syn_names)
        assert syn_count == 0, f"expected 0 synthetic items, got {syn_count}"

    def test_synthetic_ratio_one_all_synthetic(self):
        """With ratio=1.0, all items should come from synthetic sources."""
        items = self._build_mixed_items()
        syn_names = {"syn_domain1", "syn_domain2", "syn_domain3"}
        config = SamplerConfig(
            synthetic_sources=syn_names,
            synthetic_ratio=1.0,
            epoch_size=300,
        )
        sampler = MultitaskSampler(items, config, seed=42)
        epoch = sampler.sample_epoch(0)
        assert len(epoch) == 300
        syn_count = sum(1 for it in epoch if it.source in syn_names)
        assert syn_count == 300, f"expected 300 synthetic items, got {syn_count}"

    def test_synthetic_ratio_with_missing_synthetic_type(self):
        """If a type has no synthetic sources, all items come from benchmark."""
        items = (
            _make_items("mnli", "noul", 200)
            + _make_items("syn_domain1", "noul", 100)
            + _make_items("arc", "choice", 200)  # no synthetic choice sources
        )
        syn_names = {"syn_domain1"}
        config = SamplerConfig(
            synthetic_sources=syn_names,
            synthetic_ratio=0.4,
            epoch_size=300,
        )
        sampler = MultitaskSampler(items, config, seed=42)
        epoch = sampler.sample_epoch(0)
        assert len(epoch) == 300

        # Choice items should all be from benchmark (arc)
        choice_items = [it for it in epoch if it.question["type"] == "choice"]
        assert all(it.source == "arc" for it in choice_items)

        # Noul items should have ~40% synthetic
        noul_items = [it for it in epoch if it.question["type"] == "noul"]
        noul_syn = sum(1 for it in noul_items if it.source in syn_names)
        if noul_items:
            noul_ratio = noul_syn / len(noul_items)
            assert 0.30 <= noul_ratio <= 0.50, f"noul synthetic ratio {noul_ratio:.2f} outside [0.30, 0.50]"
