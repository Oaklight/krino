"""Regression test for ``train()`` referencing a ``mlp_lr`` it never took.

The per-component LR branch in ``train()`` read ``mlp_lr`` while the
signature had no such parameter, so every call raised ``NameError`` before
reaching the first epoch.  ``model/scripts/train.py`` imports ``train``, so
that script could not run at all.
"""

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from model.training.supervised import train


class _Item:
    def __init__(self, label=True):
        self.state = "s"
        self.question = {"type": "noul", "instructions": "i"}
        self.label = label
        self.teacher_probs = None
        self.source = "stub"


class _TinyModel(nn.Module):
    """One linear head, enough for the noul path."""

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(1, 1)

    def forward_noul(self, state, instructions):
        return self.head(torch.zeros(1, 1))


class _ProjectorModel(_TinyModel):
    """A model that looks like it has an MLP projector."""

    def __init__(self):
        super().__init__()
        self.projector = nn.Linear(1, 1)


def test_train_takes_mlp_lr():
    """The parameter the optimizer branch reads must exist in the signature."""
    import inspect

    sig = inspect.signature(train)
    assert "mlp_lr" in sig.parameters
    assert sig.parameters["mlp_lr"].default is None


def test_train_runs_one_epoch():
    """End-to-end: the NameError fired before any epoch, so this is the guard."""
    model = _TinyModel()
    items = [_Item(True), _Item(False)]

    result = train(model, items, items, epochs=1, eval_every=1)

    assert result["history"]
    assert result["history"][0]["epoch"] == 1
    assert "eval" in result["history"][0]


def test_train_mlp_lr_builds_separate_group(capsys):
    """With a projector and mlp_lr set, the projector gets its own LR."""
    model = _ProjectorModel()

    train(model, [], [], epochs=1, eval_every=1, mlp_lr=3e-5)

    out = capsys.readouterr().out
    assert "Per-component LR: heads=0.001, mlp=3e-05" in out


def test_train_without_mlp_lr_uses_one_group(capsys):
    model = _ProjectorModel()

    train(model, [], [], epochs=1, eval_every=1)

    out = capsys.readouterr().out
    assert "Per-component LR" not in out
