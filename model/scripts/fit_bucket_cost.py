"""Fit and cross-validate the bucket cost model from the committed pairs.

`model/scripts/bucket_cost_fit.tsv` holds every measured padded forward of the
a20k twenty-step bucket run: one row per forward, in the order the run
recorded them (rank, then step, then bucket width), with the 1.5x exclusion
rule's verdict in the last column. The model's three coefficients are a least
squares fit to the rows that column does not exclude.

Prints one JSON block and nothing else, with sorted keys and fixed precision,
so two runs are byte-identical. The keys mirror the PR body: the three
constants as fitted and as written in the trainer, k and the per-fold counts,
the four in-sample and out-of-sample rms figures each naming its fit and
population, the coefficient range across folds, and the linear form's
out-of-sample bias for comparison.

The cross-validation uses `fold = index % k` over the file's order: within a
step that cycles across widths, so every fold holds every width; across steps
it interleaves, so drift over the run is spread over the folds. The exclusion
rule is applied inside each fold to the training records only; held-out
records are predicted unfiltered.

Usage: fit_bucket_cost.py [TSV]
"""

import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TSV = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "bucket_cost_fit.tsv"
TRAINER = HERE / "train_panel2_lora.py"
K = 5


def load(path):
    records = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if index == 0 or not line.strip():
            continue
        width, padded, seconds, excluded = line.split("\t")
        records.append({"width": int(width),
                        "padded": int(padded),
                        "seconds": float(seconds),
                        "excluded": int(excluded)})
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


def cross_validate(records, filtered_training, quadratic=True):
    held = []
    coefficients = []
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
        held.extend((record, residual(record, solution, quadratic))
                    for record in test)
    return coefficients, held


def main():
    records = load(TSV)
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
    main()
