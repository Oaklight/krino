"""The committed fit pairs and the script that refits them.

The pairs are provenance for the bucket cost model's constants: the model is
quoted in the PR body and in note 13, and the run that produced it lives on a
host the repository cannot reach, so the file is what makes the numbers
re-derivable. These tests pin the schema, the exclusion verdicts and the
reproduction of the constants, the per-fold figures the script's docstring
quotes, and that the script refuses a file whose rows are not in the run's
order rather than sorting it.
"""

import importlib

import pytest

# The fit script imports nothing heavier than the standard library, so it can
# be imported at collection time. The trainer pulls in torch, so it is imported
# inside the one test that reads its constants; importing it here would make
# collection fail on an interpreter without torch.
FIT = importlib.import_module("model.scripts.fit_bucket_cost")


def test_the_fit_pairs_carry_their_provenance_columns():
    """The four key columns are what make the fold ordering reconstructible."""
    lines = FIT.DEFAULT_TSV.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    assert header == ["rank", "step", "width", "presentation", "padded_tokens",
                      "seconds", "excluded_by_1_5x_rule"]
    records = FIT.load(FIT.DEFAULT_TSV)
    assert len(records) == 203
    assert sum(1 for record in records if record["excluded"]) == 4
    assert {record["width"] for record in records} == {512, 1024, 2048, 4096}
    # The file is in the run's order, which is what the folds are assigned
    # over. Strictly increasing, not merely non-decreasing: every row's
    # position is pinned, including the pairs that differ only in
    # presentation.
    keys = [(record["rank"], record["step"], record["width"],
             record["presentation"]) for record in records]
    assert keys == sorted(keys)
    assert len(set(keys)) == len(keys)
    # A bucket presented as two forwards gives two rows with the same rank,
    # step and width; without the presentation ordinal those rows are
    # unordered relative to each other.
    groups = {}
    for record in records:
        groups.setdefault((record["rank"], record["step"], record["width"]),
                          []).append(record["presentation"])
    doubled = [group for group, values in groups.items() if values == [0, 1]]
    assert len(doubled) == 43
    assert all(sorted(values) == list(range(len(values)))
               for values in groups.values())


def test_the_pairs_reproduce_the_trainers_constants():
    """The committed pairs, refitted, are the constants the trainer uses."""
    trainer = pytest.importorskip("model.scripts.train_panel2_lora")
    records = FIT.load(FIT.DEFAULT_TSV)
    kept = [record for record in records if not record["excluded"]]
    fitted = FIT.solve([FIT.features(record) for record in kept],
                       [record["seconds"] for record in kept])
    # The trainer stores four significant figures, so the comparison tolerates
    # that rounding rather than demanding the full fit's digits.
    assert fitted[0] == pytest.approx(trainer.BUCKET_CALL_SECONDS, rel=5e-3)
    assert fitted[1] == pytest.approx(trainer.BUCKET_TOKEN_SECONDS, rel=5e-3)
    assert fitted[2] == pytest.approx(trainer.BUCKET_TOKEN_SQUARED_SECONDS,
                                      rel=5e-3)


def test_the_per_fold_figures_the_docstring_quotes():
    """The docstring's fold description is the file's, not a plausible one."""
    report = FIT.per_fold_report(FIT.load(FIT.DEFAULT_TSV))
    folds = report["folds"]
    assert [fold["n"] for fold in folds] == [41, 41, 41, 40, 40]
    assert [fold["rms"] for fold in folds] == [
        pytest.approx(0.100576), pytest.approx(0.024083),
        pytest.approx(0.166413), pytest.approx(0.026625),
        pytest.approx(0.184910)]
    assert [fold["excluded"] for fold in folds] == [2, 0, 1, 0, 1]
    assert [fold["by_width"]["4096"] for fold in folds] == [3, 3, 2, 4, 2]
    assert report["pooled_rms"] == pytest.approx(0.120956)
    assert report["kept_only_rms"] == pytest.approx(0.026822)
    # The spread across folds is the excluded rows, not the fit.
    assert report["excluded_sse_share"] == pytest.approx(0.951797)


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


def test_a_swap_within_a_presentation_pair_is_refused(tmp_path):
    """The old key cannot see this swap, so the new column has to.

    Two rows of one bucket presented as two forwards share rank, step and
    width. Swapping them leaves a (rank, step, width) check green; only the
    presentation ordinal makes the transposition visible.
    """
    lines = FIT.DEFAULT_TSV.read_text(encoding="utf-8").splitlines()
    pair = next(index for index in range(1, len(lines) - 1)
                if (lines[index].split("\t")[:3]
                    == lines[index + 1].split("\t")[:3]))
    swapped = lines[:]
    swapped[pair], swapped[pair + 1] = swapped[pair + 1], swapped[pair]
    path = tmp_path / "pair-swapped.tsv"
    path.write_text("\n".join(swapped) + "\n", encoding="utf-8")
    # The three-column key is unchanged by this swap, which is why the check
    # could not have been left there.
    old_keys = [tuple(int(field) for field in line.split("\t")[:3])
                for line in swapped[1:]]
    assert old_keys == sorted(old_keys)
    with pytest.raises(SystemExit) as failure:
        FIT.load(path)
    assert "presentation" in str(failure.value)
    # The second row of the swapped pair is the one that follows a larger key.
    assert f"line {pair + 2}" in str(failure.value)
