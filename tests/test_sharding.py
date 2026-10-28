"""Sharding must give every rank the same number of optimizer steps.

A collective has no timeout, so two ranks that take different step counts hang
the group instead of failing. Step count is derived from shard length, and shard
length must therefore be a property of the data and the group size alone —
never of what a rank happens to skip. These tests pin that arithmetic; they are
torch-free so CI actually runs them.
"""

from __future__ import annotations

import pytest

from model.training.sharding import shard, shard_length, steps_for


def test_shard_length_rounds_up():
    assert shard_length(10, 2) == 5
    assert shard_length(11, 2) == 6
    assert shard_length(1, 8) == 1
    assert shard_length(0, 4) == 0


def test_shard_length_rejects_a_nonpositive_group():
    with pytest.raises(ValueError):
        shard_length(10, 0)


def test_steps_for_rounds_up():
    assert steps_for(16, 8) == 2
    assert steps_for(17, 8) == 3
    assert steps_for(8, 8) == 1
    assert steps_for(0, 8) == 0


def test_steps_for_rejects_zero_accumulation():
    with pytest.raises(ValueError):
        steps_for(16, 0)


@pytest.mark.parametrize("n_items", [12, 15000, 15001, 15009, 7, 1])
@pytest.mark.parametrize("world_size", [2, 3, 4, 8])
@pytest.mark.parametrize("accumulation_steps", [1, 8])
def test_every_rank_takes_the_same_number_of_steps(n_items, world_size, accumulation_steps):
    """The property the all-reduce sequence depends on."""
    if n_items < world_size:
        pytest.skip("group larger than the data")
    shards = [shard(range(n_items), rank, world_size) for rank in range(world_size)]
    lengths = {len(s) for s in shards}
    assert len(lengths) == 1, f"unequal shard lengths {lengths}"
    steps = {steps_for(len(s), accumulation_steps) for s in shards}
    assert len(steps) == 1, f"unequal step counts {steps}"


def test_shards_cover_the_input():
    items = list(range(10))
    for world_size in (2, 3, 4, 5):
        union: set[int] = set()
        for rank in range(world_size):
            union.update(shard(items, rank, world_size))
        assert union == set(items), f"world_size={world_size} lost items"


def test_padding_repeats_an_item_rather_than_dropping_one():
    # 10 items over 4 ranks: stride slices are 3,3,2,2; the short ones pad to 3.
    shards = [shard(range(10), rank, 4) for rank in range(4)]
    assert [len(s) for s in shards] == [3, 3, 3, 3]
    for s in shards:
        assert len(set(s)) <= 3
    assert shards[2][-1] == shards[2][0]


def test_world_size_one_is_a_passthrough():
    items = [1, 2, 3]
    assert shard(items, 0, 1) == items
    assert shard(items, 0, 1) is not items


def test_rank_outside_the_group_is_rejected():
    with pytest.raises(ValueError):
        shard([1, 2, 3], 4, 4)


def test_group_larger_than_the_data_is_rejected():
    with pytest.raises(ValueError):
        shard([1], 2, 4)
