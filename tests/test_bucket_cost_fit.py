"""The committed fit pairs and the script that refits them.

The pairs are provenance for the bucket cost model's constants: the model is
quoted in the PR body and in note 13, and the run that produced it lives on a
host the repository cannot reach, so the file is what makes the numbers
re-derivable. These tests pin the schema, the exclusion verdicts and the
reproduction of the constants, and that the script refuses a file whose rows
are not in the run's order rather than sorting it.
"""

import importlib

import pytest

FIT = importlib.import_module("model.scripts.fit_bucket_cost")
TRAINER = importlib.import_module("model.scripts.train_panel2_lora")


def test_the_fit_pairs_carry_their_provenance_columns():
    """Rank and step are what make the fold ordering reconstructible."""
    lines = FIT.DEFAULT_TSV.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    assert header == ["rank", "step", "width", "padded_tokens", "seconds",
                      "excluded_by_1_5x_rule"]
    records = FIT.load(FIT.DEFAULT_TSV)
    assert len(records) == 203
    assert sum(1 for record in records if record["excluded"]) == 4
    assert {record["width"] for record in records} == {512, 1024, 2048, 4096}
    # The file is in the run's order, which is what the folds are assigned
    # over.
    keys = [(record["rank"], record["step"], record["width"])
            for record in records]
    assert keys == sorted(keys)


def test_the_pairs_reproduce_the_trainers_constants():
    """The committed pairs, refitted, are the constants the trainer uses."""
    records = FIT.load(FIT.DEFAULT_TSV)
    kept = [record for record in records if not record["excluded"]]
    fitted = FIT.solve([FIT.features(record) for record in kept],
                       [record["seconds"] for record in kept])
    # The trainer stores four significant figures, so the comparison tolerates
    # that rounding rather than demanding the full fit's digits.
    assert fitted[0] == pytest.approx(TRAINER.BUCKET_CALL_SECONDS, rel=5e-3)
    assert fitted[1] == pytest.approx(TRAINER.BUCKET_TOKEN_SECONDS, rel=5e-3)
    assert fitted[2] == pytest.approx(TRAINER.BUCKET_TOKEN_SQUARED_SECONDS,
                                      rel=5e-3)


def test_a_file_out_of_run_order_is_refused_not_sorted(tmp_path):
    """Sorting would move the folds, so the order is checked instead."""
    lines = FIT.DEFAULT_TSV.read_text(encoding="utf-8").splitlines()
    swapped = lines[:]
    swapped[1], swapped[2] = swapped[2], swapped[1]
    path = tmp_path / "swapped.tsv"
    path.write_text("\n".join(swapped) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit) as failure:
        FIT.load(path)
    assert "run order" in str(failure.value)
    assert "line 3" in str(failure.value)
