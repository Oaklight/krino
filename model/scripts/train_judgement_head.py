#!/usr/bin/env python3
"""Train the two-logit judgement head on cached read-position states (#324).

Every arm this feeds trains only the head; the backbone is frozen and its
hidden states were cached by ``precompute_judgement_states.py``.  The loss is
row-normalised (``#299``): the per-option loss of one row is averaged within the
row, then across rows.

The rerun of arm B uses three pre-declared variants (``--variant``): ``plain``
(BCE, zero bias, lr 1e-3, five epochs, no early stop), ``calibrated`` (plain
plus a bias-only pass fitted on the train split), and ``weighted`` (BCE with
positive weight ``1 / base_rate``).  ``--gate`` evaluates the validity checks on
the eval split and writes them beside the head; the selection is made on those,
never on JevBench.

Usage:
    python model/scripts/train_judgement_head.py \\
        --train data/benchmarks/judgement_states/causal_shard*of8.npz \\
        --state-key h_answer --variant plain --gate \\
        --output runs/arm_b_plain/head.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.src.judgement_objective import TwoWayHead, row_normalised_loss  # noqa: E402

#: Declared variants of the arm-B rerun.  ``None`` keeps the historical recipe.
VARIANTS = {
    "plain": {"loss": "bce", "yes_bias": 0.0, "lr": 1e-3, "epochs": 5, "patience": 0},
    "calibrated": {"loss": "bce", "yes_bias": 0.0, "lr": 1e-3, "epochs": 5, "patience": 0},
    "weighted": {"loss": "bce", "yes_bias": 0.0, "lr": 1e-3, "epochs": 5, "patience": 0},
}


def load_shards(paths: list[Path], state_key: str) -> dict:
    hidden, labels, row_ids, splits = [], [], [], []
    for path in paths:
        payload = np.load(path, allow_pickle=False)
        if state_key not in payload:
            raise SystemExit(f"{path} has no {state_key}; keys {sorted(payload.files)}")
        hidden.append(payload[state_key])
        labels.append(payload["labels"])
        row_ids.append(payload["row_ids"])
        splits.append(payload["split"])
    return {
        "hidden": np.concatenate(hidden),
        "labels": np.concatenate(labels).astype(np.int64),
        "row_ids": np.concatenate(row_ids),
        "splits": np.concatenate(splits),
    }


def rows_by_id(row_ids: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index in np.nonzero(mask)[0]:
        groups[str(row_ids[index])].append(int(index))
    return [np.array(indices, dtype=np.int64) for indices in groups.values()]


def option_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    rows: list[np.ndarray],
    *,
    loss_kind: str,
    pos_weight: float | None,
) -> torch.Tensor:
    if loss_kind == "ce":
        return row_normalised_loss(logits, labels, rows)
    weight = None if pos_weight is None else torch.tensor(pos_weight)
    per_option = F.binary_cross_entropy_with_logits(
        logits[:, 1], labels.float(), reduction="none", pos_weight=weight
    )
    offset = 0
    per_row = []
    for row in rows:
        per_row.append(per_option[offset : offset + len(row)].mean())
        offset += len(row)
    return torch.stack(per_row).mean()


def batch_loss(head, data: dict, rows: list[np.ndarray], **kwargs) -> torch.Tensor:
    index = np.concatenate(rows)
    hidden = torch.from_numpy(data["hidden"][index]).float()
    labels = torch.from_numpy(data["labels"][index])
    return option_loss(head(hidden), labels, rows, **kwargs)


def yes_probabilities(head, data: dict, rows: list[np.ndarray], batch_rows: int = 256) -> np.ndarray:
    out = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_rows):
            index = np.concatenate(rows[start : start + batch_rows])
            logits = head(torch.from_numpy(data["hidden"][index]).float())
            out.append(torch.softmax(logits, dim=-1)[:, 1].numpy())
    return np.concatenate(out) if out else np.zeros(0)


def evaluate(head, data: dict, rows: list[np.ndarray], batch_rows: int = 64, **kwargs) -> dict:
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(rows), batch_rows):
            chunk = rows[start : start + batch_rows]
            total += batch_loss(head, data, chunk, **kwargs).item() * len(chunk)
    probabilities = yes_probabilities(head, data, rows)
    labels = np.concatenate([data["labels"][row] for row in rows])
    gold = labels == 1
    return {
        "loss": total / max(len(rows), 1),
        "eval_rows": len(rows),
        "gold_options": int(gold.sum()),
        "gold_with_positive_yes": int((probabilities[gold] > 0.5).sum()),
        "mean_yes_all": float(probabilities.mean()) if len(probabilities) else None,
        "mean_yes_gold": float(probabilities[gold].mean()) if gold.any() else None,
        "mean_yes_non_gold": float(probabilities[~gold].mean()) if (~gold).any() else None,
    }


def paired_row_test(probabilities: np.ndarray, labels: np.ndarray, rows: list[np.ndarray]) -> dict:
    """Per-row gold-minus-non-gold difference, then a paired t-test.

    ``probabilities`` and ``labels`` are concatenated in ``rows`` order; the
    offset walk keeps the row grouping without assuming contiguous indices.
    """
    differences = []
    offset = 0
    for row in rows:
        count = len(row)
        row_probabilities = probabilities[offset : offset + count]
        row_labels = labels[offset : offset + count]
        offset += count
        gold = row_labels == 1
        if not gold.any() or gold.all():
            continue
        differences.append(
            float(row_probabilities[gold].mean() - row_probabilities[~gold].mean())
        )
    if len(differences) < 2:
        return {"rows": len(differences), "mean_diff": None, "p": None}
    array = np.asarray(differences, dtype=np.float64)
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    if sd == 0.0:
        return {"rows": len(differences), "mean_diff": mean, "p": 0.0 if mean > 0 else 1.0}
    t = mean / (sd / math.sqrt(len(array)))
    # Normal approximation; the row count is four figures.
    p = math.erfc(abs(t) / math.sqrt(2.0))
    return {"rows": len(differences), "mean_diff": mean, "p": float(p)}


def validity_gate(
    head,
    data: dict,
    rows: list[np.ndarray],
    base_rate: float,
) -> dict:
    probabilities = yes_probabilities(head, data, rows)
    labels = np.concatenate([data["labels"][row] for row in rows])
    brier = float(np.mean((probabilities - labels) ** 2))
    brier_constant = float(np.mean((base_rate - labels) ** 2))
    gold = labels == 1
    positive_rate = float((probabilities[gold] > 0.5).mean()) if gold.any() else 0.0
    paired = paired_row_test(probabilities, labels, rows)
    checks = {
        "brier_beats_constant": brier < brier_constant,
        "positive_rate_at_least_half_gold_fraction": positive_rate >= base_rate / 2,
        "gold_minus_non_gold_positive": (
            paired["mean_diff"] is not None
            and paired["mean_diff"] > 0
            and paired["p"] is not None
            and paired["p"] < 0.01
        ),
    }
    return {
        "base_rate_train": base_rate,
        "brier": brier,
        "brier_constant": brier_constant,
        "gold_positive_rate": positive_rate,
        "positive_rate_threshold": base_rate / 2,
        "paired": paired,
        "checks": checks,
        "passes": all(checks.values()),
    }


def fit_bias(head, data: dict, rows: list[np.ndarray], target: float) -> float:
    """Find the yes-logit offset whose mean P(yes) on ``rows`` equals ``target``."""
    probabilities = yes_probabilities(head, data, rows)
    labels = np.concatenate([data["labels"][row] for row in rows])
    logits_yes = np.log(np.clip(probabilities, 1e-9, 1 - 1e-9)) - np.log(
        np.clip(1 - probabilities, 1e-9, 1 - 1e-9)
    )
    low, high = -40.0, 40.0
    for _ in range(80):
        mid = (low + high) / 2
        mean = float(np.mean(1.0 / (1.0 + np.exp(-(logits_yes + mid)))))
        if mean > target:
            high = mid
        else:
            low = mid
    return (low + high) / 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", nargs="+", type=Path, required=True)
    parser.add_argument("--eval", nargs="+", type=Path, default=None)
    parser.add_argument("--state-key", default="h_answer", choices=["h_answer", "h_option"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default=None)
    parser.add_argument("--loss", choices=["ce", "bce"], default="ce")
    parser.add_argument("--bias-init", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rows-per-step", type=int, default=64)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--yes-bias", type=float, default=None)
    parser.add_argument("--gate", action="store_true", help="run the validity gate on the eval split")
    parser.add_argument(
        "--allow-collapse",
        action="store_true",
        help="write the head even if no gold option reaches P(yes) > 0.5",
    )
    args = parser.parse_args()

    declared = VARIANTS.get(args.variant or "", {})
    loss_kind = args.loss if args.loss != "ce" else declared.get("loss", "ce")
    yes_bias = (
        args.yes_bias
        if args.yes_bias is not None
        else args.bias_init
        if args.bias_init is not None
        else declared.get("yes_bias", 2.0)
    )
    lr = args.lr if args.lr is not None else declared.get("lr", 3e-3)
    epochs = args.epochs if args.epochs is not None else declared.get("epochs", 20)
    patience = args.patience if args.patience is not None else declared.get("patience", 2)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train = load_shards(args.train, args.state_key)
    train_rows = rows_by_id(train["row_ids"], train["splits"] == "train")
    if not train_rows:
        raise SystemExit("no train-split rows in the cached shards")
    train_labels = np.concatenate([train["labels"][row] for row in train_rows])
    base_rate = float(train_labels.mean())

    if args.eval:
        eval_data = load_shards(args.eval, args.state_key)
        eval_rows = rows_by_id(eval_data["row_ids"], eval_data["splits"] != "train")
    else:
        eval_data = train
        eval_rows = rows_by_id(train["row_ids"], train["splits"] != "train")
    if not eval_rows:
        raise SystemExit("no eval-split rows; the gate and early stopping would be blind")

    pos_weight = 1.0 / base_rate if args.variant == "weighted" else None
    loss_kwargs = {"loss_kind": loss_kind, "pos_weight": pos_weight}

    hidden_size = train["hidden"].shape[1]
    head = TwoWayHead(hidden_size, yes_bias=yes_bias)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=args.weight_decay)

    best = {"loss": float("inf"), "epoch": 0}
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(train_rows))
        head.train()
        running = 0.0
        steps = 0
        for start in range(0, len(order), args.rows_per_step):
            chunk = [train_rows[i] for i in order[start : start + args.rows_per_step]]
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(head, train, chunk, **loss_kwargs)
            loss.backward()
            optimizer.step()
            running += loss.item()
            steps += 1
        head.eval()
        metrics = evaluate(head, eval_data, eval_rows, **loss_kwargs)
        train_probe = yes_probabilities(
            head, train, train_rows[: min(2000, len(train_rows))]
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / max(steps, 1),
                "mean_yes_train_probe": float(train_probe.mean()),
                **metrics,
            }
        )
        print(
            f"epoch {epoch:2d}  train {running / max(steps, 1):.4f}  "
            f"eval {metrics['loss']:.4f}  P(yes) train {train_probe.mean():.4f} "
            f"eval {metrics['mean_yes_all']:.4f}  gold>0.5 {metrics['gold_with_positive_yes']}/"
            f"{metrics['gold_options']}",
            flush=True,
        )
        if metrics["loss"] < best["loss"] - 1e-5:
            best = {"loss": metrics["loss"], "epoch": epoch}
            stale = 0
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
            if patience > 0 and stale >= patience:
                print(f"early stop after {epoch} epochs (best {best['epoch']})")
                break

    head.load_state_dict(best_state)
    head.eval()

    calibration = None
    if args.variant == "calibrated":
        offset = fit_bias(head, train, train_rows, base_rate)
        with torch.no_grad():
            head.proj.bias[1] += float(offset)
        calibration = {"bias_offset": float(offset), "target_mean_yes": base_rate}
        print(f"calibrated: yes bias offset {offset:+.4f}")

    final = evaluate(head, eval_data, eval_rows, **loss_kwargs)
    gate = validity_gate(head, eval_data, eval_rows, base_rate) if args.gate else None
    if final["gold_with_positive_yes"] == 0 and not args.allow_collapse:
        raise SystemExit(
            "the head never emits P(yes) > 0.5 on a gold option; the run has "
            "collapsed and is not readable"
        )

    digest = hashlib.sha256(b"".join(path.read_bytes() for path in args.train)).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "hidden_size": hidden_size,
            "yes_bias": yes_bias,
            "state_key": args.state_key,
            "variant": args.variant,
            "loss": loss_kind,
            "pos_weight": pos_weight,
            "train_shards_sha256": digest,
            "epochs_run": len(history),
            "best_epoch": best["epoch"],
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "base_rate_train": base_rate,
            "calibration": calibration,
            "eval_final": final,
            "gate": gate,
            "history": history,
            "seed": args.seed,
            "lr": lr,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        args.output,
    )
    print(f"wrote {args.output}")
    print(json.dumps({"final": final, "gate": gate}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
