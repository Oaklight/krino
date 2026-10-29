"""Which choice head a checkpoint carries, read from its tensor names.

Torch-free on purpose. ``infer_choice_mode`` is pure key inspection, but its
tests used to live in ``test_cross_encoder_head.py``, which is
``pytest.importorskip("torch")`` — so CI skipped them along with the rest of
the file, and the checkpoint-rebuild path had no coverage on any run that
matters. A wrong answer here loads a tower checkpoint into a cross-encoder
model, leaves the head at initialisation, and reports a plausible number.
"""

from __future__ import annotations

from model.evaluation.checkpoint_readout import infer_choice_mode


class TestInferChoiceMode:
    def test_cross_encoder_tensors_are_recognised(self):
        keys = ["choice_head.norm.weight", "choice_head.score.weight"]
        assert infer_choice_mode(keys) == "cross_encoder"

    def test_attention_head_tensors_are_recognised(self):
        keys = ["choice_head.attention.query.weight", "noul_head.proj.0.weight"]
        assert infer_choice_mode(keys) == "tower"

    def test_a_checkpoint_without_a_choice_head_is_not_guessed(self):
        assert infer_choice_mode(["noul_head.proj.0.weight"]) is None

    def test_a_shared_attention_checkpoint_is_a_tower(self):
        # --shared-attention stores the module once, under choice_head.
        keys = ["choice_head.attention.key.weight", "score_head.attention.value.weight"]
        assert infer_choice_mode(keys) == "tower"

    def test_it_accepts_any_iterable(self):
        assert infer_choice_mode(iter(["choice_head.norm.bias"])) == "cross_encoder"
