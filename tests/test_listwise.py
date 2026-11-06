"""Listwise marker readout on a tiny hybrid model.

A tiny random Qwen3.5 (3 GDN layers + 1 full attention) runs on CPU in
float32, so the whole-set sequence, the cyclic-shift average and the CE
gradient can be checked without downloading a model.  The same tests cover
the fork path a hybrid cache needs: the recurrent state cannot be cropped,
so each call must run on a copy of the one-row prefix cache.
"""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
qwen35 = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")

from data.format import TypedQuestion  # noqa: E402
from model.scripts.train_listwise import item_loss, training_set  # noqa: E402
from model.src import listwise as listwise_module  # noqa: E402
from model.src.decision_model import DecisionModel  # noqa: E402
from model.src.gpu_config import GPUConfig  # noqa: E402
from model.src.listwise import ListwiseMarkerReadout  # noqa: E402

VOCAB = 128
MARKER_ID = 1


class _Encoding(dict):
    def to(self, device):
        return self


class _CharTokenizer:
    """One token per byte, with the marker as its own id."""

    eos_token_id = MARKER_ID
    pad_token_id = 0

    def __call__(
        self,
        text,
        return_tensors=None,
        add_special_tokens=True,
        truncation=False,
        max_length=None,
        padding=False,
        **kwargs,
    ):
        ids = [2 + (b % (VOCAB - 4)) for b in str(text).encode()]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        if return_tensors == "pt":
            return _Encoding(
                input_ids=torch.tensor([ids], dtype=torch.long),
                attention_mask=torch.ones(1, len(ids), dtype=torch.long),
            )
        return {"input_ids": ids}


def _tiny_model():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    config = Qwen3_5TextConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        max_position_embeddings=4096,
    )
    torch.manual_seed(0)
    return qwen35.Qwen3_5ForCausalLM(config).eval()


def _readout() -> ListwiseMarkerReadout:
    gpu = GPUConfig(
        flash_attention=False,
        bf16_compute=False,
        max_length=256,
        gradient_checkpointing=False,
    )
    model = DecisionModel(
        backbone=_tiny_model(),
        tokenizer=_CharTokenizer(),
        rank=8,
        gpu_config=gpu,
        option_marker=True,
        option_chunk_size=1,
    ).eval()
    return ListwiseMarkerReadout(model)


STATE = "a fairly long state " * 4
INSTRUCTIONS = "pick one"
OPTIONS = ["alpha", "beta", "gamma", "delta"]


@pytest.fixture(scope="module")
def readout() -> ListwiseMarkerReadout:
    return _readout()


def test_the_backbone_is_hybrid(readout):
    assert readout.model.is_hybrid


def test_score_is_one_logit_per_option(readout):
    with torch.no_grad():
        logits = readout.score(STATE, INSTRUCTIONS, OPTIONS)
    assert logits.shape == (1, len(OPTIONS))
    assert torch.isfinite(logits).all()


def test_probabilities_sum_to_one(readout):
    with torch.no_grad():
        probs = readout.probabilities(STATE, INSTRUCTIONS, OPTIONS)
    assert probs.shape == (1, len(OPTIONS))
    assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_later_options_see_earlier_ones(readout):
    """The set is mutually visible: earlier options change a later marker."""
    with torch.no_grad():
        first = readout.score(STATE, INSTRUCTIONS, ["alpha", "zeta"])
        second = readout.score(STATE, INSTRUCTIONS, ["omega", "zeta"])
    assert (first[0, 1] - second[0, 1]).abs().item() > 1e-6


def test_shift_average_uses_every_rotation(readout):
    with torch.no_grad():
        averaged = readout.score_shifted(STATE, INSTRUCTIONS, OPTIONS, shifts=4)
        single = readout.probabilities(STATE, INSTRUCTIONS, OPTIONS)
    assert averaged.shape == single.shape
    assert averaged.sum().item() == pytest.approx(1.0, abs=1e-5)
    # The identity shift is one of the four, so the average sits between the
    # identity distribution and the others rather than equal to it.
    with torch.no_grad():
        rotated = readout.probabilities(
            STATE, INSTRUCTIONS, OPTIONS[1:] + OPTIONS[:1]
        )
    unrotated = torch.cat([rotated[:, -1:], rotated[:, :-1]], dim=1)
    manual = (single + unrotated) / 2
    assert averaged[0].tolist() != pytest.approx(single[0].tolist(), abs=1e-6)
    assert (averaged - manual).abs().max().item() > 1e-6


def test_one_shift_equals_the_plain_distribution(readout):
    with torch.no_grad():
        shifted = readout.score_shifted(STATE, INSTRUCTIONS, OPTIONS, shifts=1)
        plain = readout.probabilities(STATE, INSTRUCTIONS, OPTIONS)
    assert torch.equal(shifted, plain)


def test_the_hybrid_path_forks_the_prefix_cache(readout, monkeypatch):
    calls = []
    original = listwise_module.fork_prefix_cache

    def spy(cache, rows):
        calls.append(rows)
        return original(cache, rows)

    monkeypatch.setattr(listwise_module, "fork_prefix_cache", spy)
    with torch.no_grad():
        readout.score(STATE, INSTRUCTIONS, OPTIONS)
    assert calls == [1]


def test_set_loss_backpropagates_to_the_head(readout):
    readout.model.zero_grad(set_to_none=True)
    loss = readout.set_loss(STATE, INSTRUCTIONS, OPTIONS, gold_index=2)
    loss.backward()
    grads = [
        parameter.grad
        for parameter in readout.model.marker_head.parameters()
        if parameter.grad is not None
    ]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)
    assert any(grad.abs().sum().item() > 0 for grad in grads)


def test_repeated_calls_do_not_drift(readout):
    with torch.no_grad():
        first = readout.score(STATE, INSTRUCTIONS, OPTIONS)
        second = readout.score(STATE, INSTRUCTIONS, OPTIONS)
    assert torch.equal(first, second)


class TestSubsetSampling:
    def test_the_gold_is_always_present(self):
        rng = random.Random(0)
        for _ in range(200):
            subset = ListwiseMarkerReadout.sample_subset(10, 3, rng)
            assert 3 in subset

    def test_sizes_cover_two_to_k(self):
        rng = random.Random(0)
        sizes = {
            len(ListwiseMarkerReadout.sample_subset(8, 0, rng)) for _ in range(200)
        }
        assert min(sizes) == 2
        assert max(sizes) == 8

    def test_the_gold_lands_in_every_position(self):
        rng = random.Random(0)
        positions = set()
        for _ in range(200):
            subset = ListwiseMarkerReadout.sample_subset(6, 2, rng)
            positions.add(subset.index(2))
        assert positions == set(range(6))

    def test_a_two_option_set_is_allowed(self):
        rng = random.Random(0)
        assert sorted(ListwiseMarkerReadout.sample_subset(2, 1, rng)) == [0, 1]

    def test_bad_gold_index_is_rejected(self):
        with pytest.raises(ValueError):
            ListwiseMarkerReadout.sample_subset(4, 4, random.Random(0))


def test_the_readout_rejects_a_non_marker_model():
    gpu = GPUConfig(
        flash_attention=False,
        bf16_compute=False,
        max_length=256,
        gradient_checkpointing=False,
    )
    model = DecisionModel(
        backbone=_tiny_model(),
        tokenizer=_CharTokenizer(),
        rank=8,
        gpu_config=gpu,
    )
    with pytest.raises(ValueError):
        ListwiseMarkerReadout(model)


def _choice_item(n_options: int = 4, gold: int = 2) -> TypedQuestion:
    criteria = {f"option_{i}": f"option text {i}" for i in range(n_options)}
    return TypedQuestion(
        id="listwise-test-1",
        state=STATE,
        question={
            "type": "choice",
            "instructions": INSTRUCTIONS,
            "criteria": criteria,
        },
        label=f"option_{gold}",
        source="listwise_test",
        split="train",
    )


def test_item_loss_backpropagates_through_a_sampled_subset(readout):
    readout.model.zero_grad(set_to_none=True)
    item = _choice_item()
    base = torch.full((4,), 0.25)
    # A zero alpha gates the head off by construction, so open the gate to
    # check that the listwise head trains.
    alpha = torch.nn.Parameter(torch.tensor(1.0))
    loss, kl, ce = item_loss(
        readout, item, base, alpha, random.Random(0), 1.0, 0.2, 2, False
    )
    loss.backward()
    assert kl >= 0 and ce >= 0
    assert alpha.grad is not None and torch.isfinite(alpha.grad).all()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in readout.model.marker_head.parameters()
    )


def test_item_loss_is_zero_for_an_item_without_gold(readout):
    item = _choice_item(gold=2)
    item.label = "missing_key"
    loss, kl, ce = item_loss(
        readout, item, torch.full((4,), 0.25), torch.zeros(()), random.Random(0),
        1.0, 0.2, 2, False,
    )
    assert float(loss) == 0.0 and kl == 0.0 and ce == 0.0


def _noul_item(label=True, answerable=None):
    return TypedQuestion(
        id="listwise-noul-1",
        state=STATE,
        question={"type": "noul", "instructions": INSTRUCTIONS, "criteria": None},
        label=label,
        source="listwise_test",
        split="train",
        answerable=answerable,
    )


def test_noul_item_trains_the_two_option_head(readout):
    readout.model.zero_grad(set_to_none=True)
    base = torch.full((2,), 0.5)
    alpha = torch.nn.Parameter(torch.tensor(1.0))
    loss, _kl, ce = item_loss(
        readout, _noul_item(), base, alpha, random.Random(0), 1.0, 0.2, 2, False
    )
    assert ce > 0.0
    loss.backward()
    # Identical scores for "no" and "yes" would cancel the alpha gradient to
    # exactly zero; this is the degeneracy check for the two-option softmax.
    assert alpha.grad is not None and torch.count_nonzero(alpha.grad) > 0
    before = alpha.item()
    torch.optim.SGD([alpha], lr=0.1).step()
    assert alpha.item() != before


def test_noul_abstention_row_is_masked(readout):
    loss, kl, ce = item_loss(
        readout,
        _noul_item(label=False, answerable=False),
        torch.full((2,), 0.5),
        torch.zeros(()),
        random.Random(0),
        1.0,
        0.2,
        2,
        False,
    )
    assert float(loss) == 0.0 and kl == 0.0 and ce == 0.0
    assert not loss.requires_grad


def test_training_set_includes_noul_and_excludes_abstention():
    from model.scripts.train_listwise import TRAINED_TYPES

    assert "noul" in TRAINED_TYPES
    kept, excluded = training_set(
        [
            _noul_item(),
            _noul_item(label=False, answerable=False),
            _choice_item(),
        ]
    )
    assert {item.question["type"] for item in kept} == {"noul", "choice"}
    assert len(excluded) == 1 and excluded[0].answerable is False


def test_provenance_records_overrides_and_hashes(tmp_path):
    import argparse
    import hashlib

    from model.scripts.train_listwise import run_provenance

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_bytes(b"corpus")
    base = tmp_path / "base.jsonl"
    base.write_bytes(b"base")
    args = argparse.Namespace(
        code_commit="abc123",
        corpus_sha256="def456",
        items=corpus,
        base_probs=base,
        holdout_ids=None,
    )
    provenance = run_provenance(args)
    assert provenance["code_commit"] == "abc123"
    assert provenance["corpus_sha256"] == "def456"
    assert provenance["base_probs_sha256"] == hashlib.sha256(b"base").hexdigest()
    assert "holdout_ids_sha256" not in provenance


def test_suffix_length_counts_every_marker(readout):
    ids = readout.model._tokenize_option_ids(OPTIONS)
    expected = sum(len(option_ids) + 1 for option_ids in ids)
    assert readout.suffix_length(OPTIONS) == expected


def test_fits_rejects_a_set_longer_than_max_length(readout):
    assert readout.fits(OPTIONS)
    # Options are capped at option_max_tokens, so length alone does not blow
    # the budget; the option count does.
    assert not readout.fits(["x"] * 300)


def test_sampling_respects_the_capacity_cap(readout):
    rng = random.Random(0)
    for _ in range(100):
        subset = ListwiseMarkerReadout.sample_subset(
            40, 7, rng, min_size=2, max_size=5
        )
        assert 2 <= len(subset) <= 5
        assert 7 in subset
