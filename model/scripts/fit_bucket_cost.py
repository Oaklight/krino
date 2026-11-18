"""Fit and cross-validate the bucket cost model from the committed pairs.

`model/scripts/bucket_cost_fit.tsv` holds every measured padded forward of the
a20k twenty-step bucket run: one row per forward, in the order the run
recorded them (rank, then step, then bucket width, then the presentation when
one bucket is presented as two forwards), with the 1.5x exclusion rule's
verdict in the last column. The model's three coefficients are a least
squares fit to the rows that column does not exclude.

`(rank, step, width)` does not identify a forward: 43 rank-steps present one
bucket as two forwards, so 86 of the 203 rows sit in pairs whose relative
order that key cannot fix. A check on it alone would accept a transposition
inside each pair and still call the file "in run order". `presentation` is
each row's occurrence ordinal within its group; the check is on all four key
columns, so every row's position is pinned.

Prints one JSON block and nothing else, with sorted keys and fixed precision,
so two runs are byte-identical. The keys mirror the PR body: the three
constants as fitted and as written in the trainer, k and the per-fold counts,
the four in-sample and out-of-sample rms figures each naming its fit and
population, the coefficient range across folds, and the linear form's
out-of-sample bias for comparison.

The cross-validation uses `fold = index % k` over the file's order. That order
is rank-major, so the index cycles over steps within a rank rather than across
ranks. Every fold holds rows of every width, but not in equal numbers: the
4096-token rows fall 3, 3, 2, 4 and 2 across folds 0 to 4. The four excluded
rows land in folds 0, 2 and 4, so folds 1 and 3 carry none of them in test,
and those four rows carry 95% of the out-of-sample squared error: per-fold
rms is 0.1006, 0.0241, 0.1664, 0.0266 and 0.1849 against a pooled 0.120956,
while the 199 kept rows alone give 0.026822. The spread across folds is the
excluded rows, not the fit. The exclusion rule is applied inside each fold to
the training records only; held-out records are predicted unfiltered.
`--per-fold` prints the five folds' counts and rms.

Usage: fit_bucket_cost.py [--per-fold] [TSV]
"""

import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_TSV = HERE / "bucket_cost_fit.tsv"
TRAINER = HERE / "train_panel2_lora.py"
K = 5


def load(path):
    """Read the pairs in the file's order, and check that order.

    The rows are the run's: rank, then step, then bucket width, then the
    presentation of that width. The folds are assigned over that order, so a
    file in another order would produce a cross-validation of something the
    run did not record. This verifies the order instead of sorting it, because
    a sort would silently repair such a file and hide the fact. The key covers
    `presentation` because a bucket presented as two forwards gives two rows
    with the same rank, step and width, and a check that stopped at width
    would accept those two in either order.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    needed = ("rank", "step", "width", "presentation", "padded_tokens",
              "seconds", "excluded_by_1_5x_rule")
    missing = [name for name in needed if name not in header]
    if missing:
        raise SystemExit(f"{path.name} lacks {', '.join(missing)}")
    column = {name: header.index(name) for name in needed}
    records = []
    previous = None
    for number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        fields = line.split("\t")
        rank = int(fields[column["rank"]])
        step = int(fields[column["step"]])
        width = int(fields[column["width"]])
        presentation = int(fields[column["presentation"]])
        key = (rank, step, width, presentation)
        if previous is not None and key < previous:
            raise SystemExit(
                f"{path.name} line {number}: {key} follows {previous}; the "
                f"file must be in run order (rank, step, width, presentation) "
                f"and is not sorted here, because sorting would move the folds")
        previous = key
        records.append({"rank": rank, "step": step, "width": width,
                        "presentation": presentation,
                        "padded": int(fields[column["padded_tokens"]]),
                        "seconds": float(fields[column["seconds"]]),
                        "excluded": int(fields[column["excluded_by_1_5x_rule"]])})
    return records


def features(record, quadratic=True):
    padded = record["padded"]
    return [1.0, padded, padded * padded] if quadratic else [1.0, padded]


def solve(rows, targets):
    n = len(rows[0])
    ata = [[sum(row[i] * row[j] for row in rows) for j in range(n)]
           for i in range(n)]
    atb = [sum(row[i] * y for row, y in zip(rows, targets)) for i in range(n)]
    for i in range(n):
        pivot = ata[i][i]
        for j in range(i + 1, n):
            factor = ata[j][i] / pivot
            for k in range(i, n):
                ata[j][k] -= factor * ata[i][k]
            atb[j] -= factor * atb[i]
    solution = [0.0] * n
    for i in range(n - 1, -1, -1):
        solution[i] = (atb[i] - sum(ata[i][j] * solution[j]
                                    for j in range(i + 1, n))) / ata[i][i]
    return solution


def rms(values):
    return statistics.mean([value * value for value in values]) ** 0.5


def residual(record, coefficients, quadratic=True):
    predicted = sum(c * f for c, f
                    in zip(coefficients, features(record, quadratic)))
    return record["seconds"] - predicted


def written_constants(path):
    text = path.read_text(encoding="utf-8")
    values = {}
    for name, key in (("BUCKET_CALL_SECONDS", "call_seconds"),
                      ("BUCKET_TOKEN_SECONDS", "token_seconds"),
                      ("BUCKET_TOKEN_SQUARED_SECONDS", "token_squared_seconds")):
        match = re.search(rf"^{name} = ([\d.eE+-]+)$", text, re.MULTILINE)
        if match is None:
            # A partial report would diff clean while dropping the one check
            # that ties the artifact to the code, so this is fatal.
            raise SystemExit(
                f"cannot read {name} from {path.name}; the report would not "
                f"carry the constants the trainer uses")
        values[key] = float(match.group(1))
    return values


def fold_residuals(records, filtered_training, quadratic=True):
    """The held-out residuals, fold by fold, with each fold's own fit."""
    coefficients = []
    by_fold = []
    for fold in range(K):
        train = [record for index, record in enumerate(records)
                 if index % K != fold]
        test = [record for index, record in enumerate(records)
                if index % K == fold]
        if filtered_training:
            means = {}
            for record in train:
                means.setdefault(record["width"], []).append(record["seconds"])
            means = {width: statistics.mean(values)
                     for width, values in means.items()}
            train = [record for record in train
                     if record["seconds"] <= 1.5 * means[record["width"]]]
        solution = solve([features(record, quadratic) for record in train],
                         [record["seconds"] for record in train])
        coefficients.append(solution)
        by_fold.append([(record, residual(record, solution, quadratic))
                        for record in test])
    return coefficients, by_fold


def cross_validate(records, filtered_training, quadratic=True):
    coefficients, by_fold = fold_residuals(records, filtered_training,
                                           quadratic)
    return coefficients, [item for fold in by_fold for item in fold]


def per_fold_report(records):
    """What each fold holds and what it predicts, so the docstring's five rms
    figures have a command behind them rather than a recipe."""
    _, by_fold = fold_residuals(records, True)
    folds = []
    for fold, held in enumerate(by_fold):
        values = [value for _, value in held]
        folds.append({
            "fold": fold,
            "n": len(held),
            "excluded": sum(1 for record, _ in held if record["excluded"]),
            "by_width": {str(width): sum(1 for record, _ in held
                                         if record["width"] == width)
                         for width in sorted({record["width"]
                                              for record, _ in held})},
            "rms": round(rms(values), 6),
            "sse": round(sum(value * value for value in values), 6),
        })
    held = [item for fold in by_fold for item in fold]
    return {
        "folds": folds,
        "pooled_rms": round(rms([value for _, value in held]), 6),
        "kept_only_rms": round(
            rms([value for record, value in held if not record["excluded"]]), 6),
        "excluded_sse_share": round(
            sum(value * value for record, value in held if record["excluded"])
            / sum(value * value for _, value in held), 6),
    }


def main(path=None, per_fold=False):
    records = load(Path(path) if path else DEFAULT_TSV)
    if per_fold:
        print(json.dumps(per_fold_report(records), indent=2, sort_keys=True))
        return
    kept = [record for record in records if not record["excluded"]]
    fitted = solve([features(record) for record in kept],
                   [record["seconds"] for record in kept])
    coefficients, held = cross_validate(records, True)
    _, held_unfiltered = cross_validate(records, False)
    linear_coefficients, linear_held = cross_validate(records, True,
                                                      quadratic=False)

    by_width = {}
    for record, value in held:
        by_width.setdefault(record["width"], []).append(value)
    linear_by_width = {}
    for record, value in linear_held:
        linear_by_width.setdefault(record["width"], []).append(value)

    report = {
        "constants": {
            "fitted_on_kept": {
                "call_seconds": round(fitted[0], 6),
                "token_seconds": round(fitted[1], 10),
                "token_squared_seconds": round(fitted[2], 12),
            },
            "written_in_trainer": written_constants(TRAINER),
        },
        "population": {
            "records": len(records),
            "kept": len(kept),
            "excluded": len(records) - len(kept),
            "by_width": {str(width): sum(1 for record in records
                                         if record["width"] == width)
                         for width in sorted({r["width"] for r in records})},
        },
        "k": K,
        "fold_sizes": [sum(1 for index in range(len(records))
                           if index % K == fold) for fold in range(K)],
        "rms": {
            "shipped_fit_kept_in_sample": round(rms([residual(r, fitted)
                                                     for r in kept]), 6),
            "shipped_fit_all_records_in_sample": round(rms([residual(r, fitted)
                                                            for r in records]), 6),
            "refit_all_records_in_sample": round(rms(
                [residual(r, solve([features(x) for x in records],
                                   [x["seconds"] for x in records]))
                 for r in records]), 6),
            "five_fold_out_of_sample_filtered_training": round(
                rms([value for _, value in held]), 6),
            "five_fold_out_of_sample_unfiltered_training": round(
                rms([value for _, value in held_unfiltered]), 6),
        },
        "out_of_sample_by_width": {
            str(width): {"n": len(values), "rms": round(rms(values), 6),
                         "bias": round(statistics.mean(values), 6)}
            for width, values in sorted(by_width.items())},
        "coefficient_range_across_folds": {
            name: {"min": round(min(values), 12), "max": round(max(values), 12),
                   "mean": round(statistics.mean(values), 12),
                   "sd": round(statistics.stdev(values), 12)}
            for name, values in zip(
                ("call_seconds", "token_seconds", "token_squared_seconds"),
                zip(*coefficients))},
        "linear_form": {
            "coefficients": [round(value, 12)
                             for value in solve(
                                 [features(r, False) for r in kept],
                                 [r["seconds"] for r in kept])],
            "out_of_sample_rms": round(rms([value for _, value
                                            in linear_held]), 6),
            "out_of_sample_by_width": {
                str(width): {"n": len(values), "rms": round(rms(values), 6),
                             "bias": round(statistics.mean(values), 6)}
                for width, values in sorted(linear_by_width.items())},
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    # The path is read here, not at import: a test process's argv[1] is the
    # test file, and binding it at import would aim the loader at that.
    arguments = [argument for argument in sys.argv[1:] if argument != "--per-fold"]
    main(arguments[0] if arguments else None,
         per_fold="--per-fold" in sys.argv[1:])
