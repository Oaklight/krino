"""Tests for the data-parallel helpers.

Runs single-process on CPU: the collectives are replaced by fakes that emulate
a second rank, so no GPU or ``torchrun`` launch is needed.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from model.training import distributed


class TestWorldInfo:
    def test_single_process_defaults(self, monkeypatch):
        for var in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
            monkeypatch.delenv(var, raising=False)
        assert distributed.world_info() == (0, 1, 0)

    def test_reads_launcher_env(self, monkeypatch):
        monkeypatch.setenv("RANK", "2")
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("LOCAL_RANK", "2")
        assert distributed.world_info() == (2, 4, 2)


class TestShard:
    def test_covers_every_item_exactly_once(self):
        items = list(range(10))
        shards = [distributed.shard(items, r, 3) for r in range(3)]
        assert sorted(item for s in shards for item in s) == items

    def test_sizes_differ_by_at_most_one(self):
        items = list(range(10))
        sizes = [len(distributed.shard(items, r, 3)) for r in range(3)]
        assert max(sizes) - min(sizes) <= 1

    def test_single_process_is_identity(self):
        items = list(range(5))
        assert distributed.shard(items, 0, 1) == items

    def test_empty_input(self):
        assert distributed.shard([], 1, 2) == []


class TestMergeEvalStats:
    def _stats(self, n_items, correct, mean_loss, by_type=None):
        return {
            "aggregate": {
                "mean_loss": mean_loss,
                "accuracy": correct / n_items,
                "n_items": n_items,
                "correct": correct,
            },
            "by_type": by_type or {},
            "by_source": {},
        }

    def test_single_stats_returned_unchanged(self):
        stats = self._stats(10, 7, 0.5)
        assert distributed.merge_eval_stats([stats]) is stats

    def test_empty_list(self):
        merged = distributed.merge_eval_stats([])
        assert merged["aggregate"]["n_items"] == 0

    def test_weights_by_item_count(self):
        """A 30-item rank must dominate a 10-item one, not be averaged equally."""
        left = self._stats(30, 30, 0.1)
        right = self._stats(10, 0, 0.9)
        merged = distributed.merge_eval_stats([left, right])["aggregate"]
        assert merged["n_items"] == 40
        assert merged["correct"] == 30
        assert merged["accuracy"] == pytest.approx(0.75)
        assert merged["mean_loss"] == pytest.approx((0.1 * 30 + 0.9 * 10) / 40)

    def test_merges_partial_type_coverage(self):
        left = self._stats(4, 2, 0.4, {"noul": {"mean_loss": 0.4, "accuracy": 0.5,
                                               "n_items": 4, "correct": 2}})
        right = self._stats(6, 3, 0.6, {"choice": {"mean_loss": 0.6, "accuracy": 0.5,
                                                   "n_items": 6, "correct": 3}})
        merged = distributed.merge_eval_stats([left, right])
        assert set(merged["by_type"]) == {"noul", "choice"}
        assert merged["by_type"]["noul"]["n_items"] == 4
        assert merged["by_type"]["choice"]["n_items"] == 6

    def test_merges_overlapping_types(self):
        left = self._stats(4, 4, 0.2, {"noul": {"mean_loss": 0.2, "accuracy": 1.0,
                                                "n_items": 4, "correct": 4}})
        right = self._stats(4, 0, 0.8, {"noul": {"mean_loss": 0.8, "accuracy": 0.0,
                                                 "n_items": 4, "correct": 0}})
        merged = distributed.merge_eval_stats([left, right])["by_type"]["noul"]
        assert merged["n_items"] == 8
        assert merged["accuracy"] == pytest.approx(0.5)
        assert merged["mean_loss"] == pytest.approx(0.5)


class TestAverageGradients:
    def _enable_fake_all_reduce(self, monkeypatch, factor=2):
        monkeypatch.setattr(distributed, "is_initialized", lambda: True)

        def fake_all_reduce(tensor, op=None):
            # Emulate the other rank contributing an identical tensor.
            tensor.mul_(factor)

        monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    def test_single_process_is_noop(self, monkeypatch):
        param = torch.nn.Parameter(torch.zeros(2))
        param.grad = torch.tensor([3.0, 4.0])
        monkeypatch.setattr(distributed, "is_initialized", lambda: False)
        distributed.average_gradients([param], world_size=1)
        assert torch.equal(param.grad, torch.tensor([3.0, 4.0]))

    def test_missing_gradients_are_zero_filled(self, monkeypatch):
        used = torch.nn.Parameter(torch.zeros(2))
        used.grad = torch.tensor([2.0, 4.0])
        unused = torch.nn.Parameter(torch.zeros(2))
        self._enable_fake_all_reduce(monkeypatch)

        distributed.average_gradients([used, unused], world_size=2)

        assert torch.equal(used.grad, torch.tensor([2.0, 4.0]))
        assert unused.grad is not None, "every rank must step the same parameter set"
        assert torch.equal(unused.grad, torch.zeros(2))

    def test_mixed_dtypes_are_reduced_separately(self, monkeypatch):
        fp32 = torch.nn.Parameter(torch.zeros(2))
        fp32.grad = torch.tensor([1.0, 1.0])
        bf16 = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
        bf16.grad = torch.tensor([2.0, 2.0], dtype=torch.bfloat16)
        self._enable_fake_all_reduce(monkeypatch)

        distributed.average_gradients([fp32, bf16], world_size=2)

        assert torch.equal(fp32.grad, torch.tensor([1.0, 1.0]))
        assert torch.equal(bf16.grad, torch.tensor([2.0, 2.0], dtype=torch.bfloat16))


class TestGatherTrainStats:
    def test_weighted_by_item_count(self, monkeypatch):
        monkeypatch.setattr(distributed, "is_initialized", lambda: True)
        other = {"mean_loss": 1.0, "n_items": 10, "n_skipped": 2}

        def fake_gather(out, obj):
            out[0] = obj
            out[1] = other

        monkeypatch.setattr(torch.distributed, "all_gather_object", fake_gather)
        merged = distributed.gather_train_stats(
            {"mean_loss": 0.0, "n_items": 30, "n_skipped": 0}, world_size=2
        )
        assert merged["n_items"] == 40
        assert merged["mean_loss"] == pytest.approx(0.25)
        assert merged["n_skipped"] == 2

    def test_single_process_passthrough(self, monkeypatch):
        monkeypatch.setattr(distributed, "is_initialized", lambda: False)
        stats = {"mean_loss": 0.5, "n_items": 3, "n_skipped": 0}
        assert distributed.gather_train_stats(stats, world_size=1) is stats
