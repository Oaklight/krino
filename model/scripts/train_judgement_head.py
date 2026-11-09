#!/usr/bin/env python3
"""Train the two-logit judgement head on cached read-position states (#324).

Every arm this feeds trains only the head; the backbone is frozen and its
hidden states were cached by ``precompute_judgement_states.py``.  The loss is
row-normalised (``#299``): the cross-entropy of one row's options is averaged
within the row, then across rows.

The positive-rate check is mandatory: if no gold option reaches P(yes) > 0.5 on
the validation split, the head has collapsed to the 1-in-K prior and the run is
not readable.

Usage:
    python model/scripts/train_judgement_head.py \\
        --train data/benchmarks/judgement_states/causal_shard*of8.npz \\
        --eval  data/benchmarks/judgement_states/causal_shard*of8.npz \\
        --state-key h_answer --output runs/arm_b/head.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.src.judgement_objective import TwoWayHead, row_normalised_loss  # noqa: E402


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


def batch_loss(head, data: dict, rows: list[np.ndarray]) -> torch.Tensor:
    index = np.concatenate(rows)
    hidden = torch.from_numpy(data["hidden"][index]).float()
    labels = torch.from_numpy(data["labels"][index])
    logits = head(hidden)
    offset = 0
    groups = []
    for row in rows:
        groups.append(torch.arange(offset, offset + len(row)))
        offset += len(row)
    return row_normalised_loss(logits, labels, groups)


def evaluate(head, data: dict, rows: list[np.ndarray], batch_rows: int = 64) -> dict:
    total = 0.0
    positive = 0
    gold = 0
    with torch.no_grad():
        for start in range(0, len(rows), batch_rows):
            chunk = rows[start : start + batch_rows]
            total += batch_loss(head, data, chunk).item() * len(chunk)
            index = np.concatenate(chunk)
            logits = head(torch.from_numpy(data["hidden"][index]).float())
            gold_mask = data["labels"][index] == 1
            gold += int(gold_mask.sum())
            if gold_mask.any():
                positive += int((logits[gold_mask, 1] > 0).sum())
    return {
        "loss": total / max(len(rows), 1),
        "eval_rows": len(rows),
        "gold_options": gold,
        "gold_with_positive_yes_logit": positive,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", nargs="+", type=Path, required=True)
    parser.add_argument("--eval", nargs="+", type=Path, default=None)
    parser.add_argument("--state-key", default="h_answer", choices=["h_answer", "h_option"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rows-per-step", type=int, default=64)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--yes-bias", type=float, default=2.0)
    parser.add_argument(
        "--allow-collapse",
        action="store_true",
        help="write the head even if no gold option reaches P(yes) > 0.5",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train = load_shards(args.train, args.state_key)
    train_rows = rows_by_id(train["row_ids"], train["splits"] == "train")
    if not train_rows:
        raise SystemExit("no train-split rows in the cached shards")

    if args.eval:
        eval_data = load_shards(args.eval, args.state_key)
        eval_rows = rows_by_id(eval_data["row_ids"], eval_data["splits"] != "train")
    else:
        eval_data = train
        eval_rows = rows_by_id(train["row_ids"], train["splits"] != "train")
    if not eval_rows:
        raise SystemExit("no eval-split rows; early stopping would be blind")

    hidden_size = train["hidden"].shape[1]
    head = TwoWayHead(hidden_size, yes_bias=args.yes_bias)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best = {"loss": float("inf"), "epoch": 0}
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(train_rows))
        head.train()
        running = 0.0
        steps = 0
        for start in range(0, len(order), args.rows_per_step):
            chunk = [train_rows[i] for i in order[start : start + args.rows_per_step]]
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(head, train, chunk)
            loss.backward()
            optimizer.step()
            running += loss.item()
            steps += 1
        head.eval()
        metrics = evaluate(head, eval_data, eval_rows)
        history.append({"epoch": epoch, "train_loss": running / max(steps, 1), **metrics})
        print(
            f"epoch {epoch:2d}  train {running / max(steps, 1):.4f}  "
            f"eval {metrics['loss']:.4f}  gold>0.5 {metrics['gold_with_positive_yes_logit']}/"
            f"{metrics['gold_options']}",
            flush=True,
        )
        if metrics["loss"] < best["loss"] - 1e-5:
            best = {"loss": metrics["loss"], "epoch": epoch}
            stale = 0
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop after {epoch} epochs (best {best['epoch']})")
                break

    head.load_state_dict(best_state)
    head.eval()
    final = evaluate(head, eval_data, eval_rows)
    if final["gold_with_positive_yes_logit"] == 0 and not args.allow_collapse:
        raise SystemExit(
            "the head never emits P(yes) > 0.5 on a gold option; the run has "
            "collapsed to the 1-in-K prior and is not readable"
        )

    digest = hashlib.sha256(
        b"".join(path.read_bytes() for path in args.train)
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "hidden_size": hidden_size,
            "yes_bias": args.yes_bias,
            "state_key": args.state_key,
            "train_shards_sha256": digest,
            "epochs_run": len(history),
            "best_epoch": best["epoch"],
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "eval_final": final,
            "history": history,
            "seed": args.seed,
            "lr": args.lr,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        args.output,
    )
    print(f"wrote {args.output}")
    print(json.dumps(final))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
