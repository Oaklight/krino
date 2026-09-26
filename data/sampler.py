"""Multi-task sampler for type-balanced, source-weighted training.

Produces epoch-level item sequences that balance across question types
(noul/choice/score) and apply per-source weights and caps. Pure Python
with no torch dependency.
"""

from __future__ import annotations

import random
from collections import defaultdict, deque
from dataclasses import dataclass, field

from .format import TypedQuestion


@dataclass
class SamplerConfig:
    """Configuration for multi-task sampling.

    Args:
        type_ratios: Relative sampling ratio per question type.
            E.g. {"noul": 1.0, "choice": 1.0, "score": 1.0} for equal balance.
        source_weights: Relative sampling weight per source within each type pool.
            Higher weight = more likely to be drawn. Sources not listed default to 1.0.
        source_caps: Maximum items per source before sampling. Applied once during
            init to prevent large datasets from dominating.
        epoch_size: Total items per epoch. None = sum of all capped source sizes.
        accumulation_steps: Window size for type interleaving. Each window of this
            size has slots allocated proportionally to type_ratios.
        synthetic_sources: Source names that are considered synthetic data.
        synthetic_ratio: Target proportion of synthetic data per type (0.0 to 1.0).
            None means no synthetic/benchmark split — all sources sampled together.
    """

    type_ratios: dict[str, float] = field(default_factory=lambda: {"noul": 1.0, "choice": 1.0, "score": 1.0})
    source_weights: dict[str, float] = field(default_factory=dict)
    source_caps: dict[str, int] = field(default_factory=dict)
    epoch_size: int | None = None
    accumulation_steps: int = 8
    synthetic_sources: set[str] = field(default_factory=set)
    synthetic_ratio: float | None = None


class MultitaskSampler:
    """Type-balanced, source-weighted sampler for multi-task training.

    Three levels of control:
    - Type ratios: each accumulation window has proportional noul:choice:score slots
    - Source weights: within each type pool, sources sampled with configurable probability
    - Source caps: max items per source (applied once at init)

    Small pools are oversampled with replacement when needed. Each epoch uses
    a different seed (base_seed + epoch) for reproducible but varied ordering.
    """

    def __init__(
        self,
        items: list[TypedQuestion],
        config: SamplerConfig,
        seed: int = 42,
    ) -> None:
        self._config = config
        self._seed = seed

        # Index items by type, then by source within each type
        by_type: dict[str, dict[str, list[TypedQuestion]]] = defaultdict(lambda: defaultdict(list))
        for item in items:
            q_type = item.question.get("type", "noul")
            by_type[q_type][item.source].append(item)

        # Apply source caps (downsample once, deterministically)
        cap_rng = random.Random(seed)
        self._pools: dict[str, dict[str, list[TypedQuestion]]] = {}
        total_capped = 0
        for q_type, sources in by_type.items():
            self._pools[q_type] = {}
            for source, source_items in sources.items():
                cap = config.source_caps.get(source)
                if cap is not None and len(source_items) > cap:
                    source_items = cap_rng.sample(source_items, cap)
                self._pools[q_type][source] = source_items
                total_capped += len(source_items)

        # Compute per-source sampling weights within each type pool
        self._source_weights: dict[str, list[tuple[str, float]]] = {}
        for q_type, sources in self._pools.items():
            weighted = []
            for source in sources:
                w = config.source_weights.get(source, 1.0)
                weighted.append((source, w))
            self._source_weights[q_type] = weighted

        # Determine epoch size
        self._epoch_size = config.epoch_size if config.epoch_size is not None else total_capped

        # Only keep types that actually have items
        self._active_types = [t for t in self._pools if self._pools[t]]

    @property
    def epoch_size(self) -> int:
        """Number of items per epoch."""
        return self._epoch_size

    def _weighted_sample(
        self,
        q_type: str,
        n: int,
        rng: random.Random,
    ) -> list[TypedQuestion]:
        """Draw n items from a type pool with source weighting.

        Uses weighted random selection across sources, then picks a random
        item from the selected source. Small pools are oversampled with
        replacement.
        """
        sources = self._source_weights.get(q_type, [])
        if not sources:
            return []

        pool = self._pools[q_type]
        source_names = [s for s, _ in sources]
        weights = [w for _, w in sources]

        # Track per-source indices for within-epoch variety
        source_indices: dict[str, list[int]] = {}
        for name in source_names:
            indices = list(range(len(pool[name])))
            rng.shuffle(indices)
            source_indices[name] = indices

        source_pos: dict[str, int] = {name: 0 for name in source_names}

        result: list[TypedQuestion] = []
        for _ in range(n):
            # Weighted source selection
            chosen_source = rng.choices(source_names, weights=weights, k=1)[0]
            items = pool[chosen_source]
            if not items:
                continue

            pos = source_pos[chosen_source]
            if pos >= len(source_indices[chosen_source]):
                # Reshuffle for oversampling (with replacement across passes)
                rng.shuffle(source_indices[chosen_source])
                pos = 0

            idx = source_indices[chosen_source][pos]
            source_pos[chosen_source] = pos + 1
            result.append(items[idx])

        return result

    def _split_sample(
        self,
        q_type: str,
        n: int,
        rng: random.Random,
    ) -> list[TypedQuestion]:
        """Draw n items split between synthetic and benchmark sources.

        Splits the type's source weights into synthetic and benchmark groups,
        then draws from each group proportionally to synthetic_ratio. Falls
        back to the available group when one side has no sources.

        Args:
            q_type: Question type to sample from.
            n: Total number of items to draw.
            rng: Random instance for reproducible sampling.

        Returns:
            Combined list of items from both synthetic and benchmark sources.
        """
        assert self._config.synthetic_ratio is not None
        ratio = self._config.synthetic_ratio
        syn_sources = self._config.synthetic_sources

        all_sources = self._source_weights.get(q_type, [])
        if not all_sources:
            return []

        syn_list = [(s, w) for s, w in all_sources if s in syn_sources]
        bench_list = [(s, w) for s, w in all_sources if s not in syn_sources]

        # If one side is empty, draw everything from the other
        if not syn_list:
            return self._weighted_sample(q_type, n, rng)
        if not bench_list:
            return self._weighted_sample(q_type, n, rng)

        n_synthetic = int(n * ratio)
        n_benchmark = n - n_synthetic

        # Temporarily swap source_weights to draw from each group
        orig = self._source_weights[q_type]

        self._source_weights[q_type] = syn_list
        syn_items = self._weighted_sample(q_type, n_synthetic, rng)

        self._source_weights[q_type] = bench_list
        bench_items = self._weighted_sample(q_type, n_benchmark, rng)

        self._source_weights[q_type] = orig
        return syn_items + bench_items

    def sample_epoch(self, epoch: int) -> list[TypedQuestion]:
        """Sample items for one epoch with type-balanced interleaving.

        Args:
            epoch: Epoch number (0-indexed). Combined with base seed for
                reproducible but epoch-varying randomness.

        Returns:
            Flat list of TypedQuestion items, interleaved by type within
            accumulation-step windows.
        """
        rng = random.Random(self._seed + epoch)

        if not self._active_types:
            return []

        # Compute items per type from type_ratios
        total_ratio = sum(self._config.type_ratios.get(t, 0.0) for t in self._active_types)
        if total_ratio <= 0:
            return []

        items_per_type: dict[str, int] = {}
        remaining = self._epoch_size
        sorted_types = sorted(self._active_types)
        for i, q_type in enumerate(sorted_types):
            ratio = self._config.type_ratios.get(q_type, 0.0)
            if ratio <= 0:
                items_per_type[q_type] = 0
                continue
            if i == len(sorted_types) - 1:
                # Last type gets remainder to avoid rounding loss
                items_per_type[q_type] = remaining
            else:
                count = int(self._epoch_size * ratio / total_ratio)
                items_per_type[q_type] = count
                remaining -= count

        # Draw items for each type with weighted sampling
        use_split = self._config.synthetic_ratio is not None
        type_items: dict[str, list[TypedQuestion]] = {}
        for q_type in sorted_types:
            n = items_per_type.get(q_type, 0)
            if n > 0:
                if use_split:
                    type_items[q_type] = self._split_sample(q_type, n, rng)
                else:
                    type_items[q_type] = self._weighted_sample(q_type, n, rng)
            else:
                type_items[q_type] = []

        # Interleave: chunk into windows of accumulation_steps,
        # allocate slots to types proportionally within each window
        acc = self._config.accumulation_steps
        result: list[TypedQuestion] = []

        # Build per-type deques (popleft is O(1) vs list.pop(0) O(n))
        type_queues: dict[str, deque[TypedQuestion]] = {
            t: deque(items) for t, items in type_items.items()
        }
        active = [t for t in sorted_types if type_queues.get(t)]

        while active:
            window: list[TypedQuestion] = []
            remaining_items = sum(len(type_queues[t]) for t in active)
            window_size = min(acc, remaining_items)

            # Allocate slots proportionally, guarding against over-allocation
            total_r = sum(self._config.type_ratios.get(t, 0.0) for t in active)
            if total_r <= 0:
                break

            slots: dict[str, int] = {}
            slot_remaining = window_size
            for i, t in enumerate(active):
                r = self._config.type_ratios.get(t, 0.0)
                if i == len(active) - 1:
                    slots[t] = min(slot_remaining, len(type_queues[t]))
                else:
                    s = int(window_size * r / total_r) if r > 0 else 0
                    s = min(s, len(type_queues[t]), slot_remaining)
                    slots[t] = s
                    slot_remaining -= s

            for t in active:
                n_take = slots.get(t, 0)
                for _ in range(n_take):
                    if type_queues[t]:
                        window.append(type_queues[t].popleft())

            rng.shuffle(window)
            result.extend(window)
            active = [t for t in sorted_types if type_queues.get(t)]

        return result
