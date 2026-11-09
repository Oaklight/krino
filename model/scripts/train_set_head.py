"""Train the set-function head on cached option states (#340, arm D).

The backbone is frozen; the head sees only the cached ``(K, H)`` option states
of each item.  The loss is a softmax cross-entropy over the K options, so a
noul item is a two-way choice and a choice item is a K-way one, under one
objective.

Selection is on the corpus test split's NLL, and the validity gate runs on the
same split.  JevBench and the development set are read once, after selection.

Usage::

    python model/scripts/train_set_head.py --cache <natural-cache> \\
        --output runs/set_head/head.pt --gate
    python model/scripts/train_set_head.py --print-config-hash --cache <natural-cache>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.src.set_head import (  # noqa: E402
    DEFAULT_FFN,
    DEFAULT_HEADS,
    DEFAULT_LAYERS,
    DEFAULT_WIDTH,
    SetHead,
)

SEED = 42
EPOCHS = 5
BATCH_SIZE = 64
EVAL_BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 0.01


def load_cache(directory: Path, split: str, order: str = "natural") -> dict[str, dict]:
    """All ``(split, K)`` files for one order as ``K -> {states, gold, ids}``.

    States are kept as bf16 on the CPU: a reload of the stored bit pattern is
    exact, and the batch is promoted to float32 for the head.
    """
    groups: dict[str, dict] = {}
    files = sorted(directory.glob(f"states_{split}_K*_{order}*.npz"))
    if not files:
        raise SystemExit(
            f"no cache files for split={split} order={order} in {directory}"
        )
    for path in files:
        with np.load(path, allow_pickle=False) as payload:
            states = torch.from_numpy(payload["states"]).view(torch.bfloat16)
            gold = torch.from_numpy(payload["gold"]).to(torch.long)
            ids = [str(value) for value in payload["ids"]]
        k = int(path.stem.split("_K")[1].split("_")[0])
        if states.shape[1] != k:
            raise SystemExit(f"{path}: K={k} but states have {states.shape[1]} options")
        if k in groups:
            groups[k]["states"].append(states)
            groups[k]["gold"].append(gold)
            groups[k]["ids"].extend(ids)
        else:
            groups[k] = {"states": [states], "gold": [gold], "ids": ids}
    for k, entry in groups.items():
        states = torch.cat(entry["states"], dim=0)
        gold = torch.cat(entry["gold"], dim=0)
        if len(entry["ids"]) != states.shape[0]:
            raise SystemExit(
                f"K={k}: {len(entry['ids'])} ids for {states.shape[0]} states"
            )
        entry["states"] = states
        entry["gold"] = gold
        entry["K"] = k
    return groups


def batch_from(
    states: torch.Tensor, gold: torch.Tensor, indices: torch.Tensor, device: str
):
    """``(K, H)`` rows to a padded batch; K is fixed within a group."""
    selected = states[indices].to(device=device, dtype=torch.float32)
    valid = torch.ones(selected.shape[:2], dtype=torch.bool, device=device)
    return selected, valid, gold[indices].to(device)


def epoch_batches(sizes: dict[int, int], generator: torch.Generator, batch_size: int):
    """Uniform item order, batches kept single-K."""
    pairs = [(k, index) for k, size in sizes.items() for index in range(size)]
    order = torch.randperm(len(pairs), generator=generator).tolist()
    open_batches: dict[int, list[int]] = defaultdict(list)
    for position in order:
        k, index = pairs[position]
        open_batches[k].append(index)
        if len(open_batches[k]) == batch_size:
            yield k, open_batches[k]
            open_batches[k] = []
    for k, leftover in open_batches.items():
        if leftover:
            yield k, leftover


@torch.inference_mode()
def evaluate(head: SetHead, groups: dict[str, dict], device: str) -> dict:
    """NLL, accuracy and the per-row probabilities on a split."""
    head.eval()
    total_nll = 0.0
    correct = 0
    items = 0
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    gold_probabilities: list[np.ndarray] = []
    uniform: list[np.ndarray] = []
    row_lengths: list[int] = []
    for k in sorted(groups):
        entry = groups[k]
        states, gold = entry["states"], entry["gold"]
        for start in range(0, states.shape[0], EVAL_BATCH):
            indices = torch.arange(start, min(start + EVAL_BATCH, states.shape[0]))
            selected, valid, target = batch_from(states, gold, indices, device)
            logits = head(selected, valid)
            loss = F.cross_entropy(logits, target, reduction="sum")
            probs = torch.softmax(logits, dim=-1)
            chosen = probs.argmax(dim=-1)
            total_nll += float(loss)
            correct += int((chosen == target).sum())
            items += len(target)
            probabilities.append(probs.cpu().numpy().reshape(-1))
            one_hot = np.zeros((len(target), k), dtype=np.float64)
            for row, option in enumerate(target.cpu().tolist()):
                one_hot[row, option] = 1.0
            labels.append(one_hot.reshape(-1))
            row_lengths.extend([k] * len(target))
            gold_probabilities.append(
                probs.gather(1, target.unsqueeze(1)).squeeze(1).cpu().numpy()
            )
            uniform.append(np.full(len(target), 1.0 / k))
    return {
        "items": items,
        "nll": total_nll / max(items, 1),
        "accuracy": correct / max(items, 1),
        "probabilities": np.concatenate(probabilities),
        "labels": np.concatenate(labels),
        "gold_probabilities": np.concatenate(gold_probabilities),
        "uniform": np.concatenate(uniform),
        "row_lengths": row_lengths,
    }


def validity_gate(evaluation: dict) -> dict:
    """The issue's three checks, on the corpus eval split only.

    The row is a K-way distribution, so each ``(row, option)`` is one binary
    outcome with the head's probability for that option against the gold
    indicator.  The constant predictor uses the pooled gold fraction,
    ``base_rate = mean(1/K)``.  ``positive_rate`` counts gold options whose
    probability exceeds the uniform ``1/K``; for a two-option item that is
    exactly B's ``P(yes) > 0.5``, and the threshold stays ``base_rate / 2``.
    """
    from train_judgement_head import paired_row_test  # same directory, #324

    probabilities = evaluation["probabilities"]
    labels = evaluation["labels"]
    rows: list[np.ndarray] = []
    offset = 0
    for length in evaluation["row_lengths"]:
        rows.append(np.arange(offset, offset + length))
        offset += length
    base_rate = float(labels.mean())
    brier = float(np.mean((probabilities - labels) ** 2))
    brier_constant = float(np.mean((base_rate - labels) ** 2))
    positive_rate = float(
        (evaluation["gold_probabilities"] > evaluation["uniform"]).mean()
    )
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
        "base_rate": base_rate,
        "brier": brier,
        "brier_constant": brier_constant,
        "gold_positive_rate": positive_rate,
        "positive_rate_threshold": base_rate / 2,
        "paired": paired,
        "checks": checks,
        "passes": all(checks.values()),
    }


def config_for(args: argparse.Namespace, hidden_size: int) -> dict:
    """The pre-registered configuration; its hash goes into #340 before the run.

    ``data_digest`` is the rows file the states were extracted from, so the hash
    is fixed before the cache is built.  The stored-state digest and the elapsed
    time are recorded beside the config, not inside it.
    """
    return {
        "arm": "D",
        "head": "set transformer over option states",
        "backbone": "Qwen/Qwen3-Reranker-4B",
        "hidden_size": hidden_size,
        "width": DEFAULT_WIDTH,
        "layers": DEFAULT_LAYERS,
        "heads": DEFAULT_HEADS,
        "ffn": DEFAULT_FFN,
        "dropout": 0.0,
        "pooling": "pma_learned_query",
        "readout": "dot product of each option with the pooled set vector",
        "objective": "softmax cross-entropy over the K options",
        "train_split": "corpus train",
        "eval_split": "corpus test",
        "epochs": args.epochs,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "selection": "minimum eval-split NLL over the five epochs",
        "data_digest": args.data_digest,
        "extraction": json.loads(args.extraction),
        "io_dtype": "bfloat16 storage, float32 head",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config-out", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--print-config-hash", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--data-digest", required=True, help="sha256 of the rows file")
    parser.add_argument(
        "--extraction",
        default="{}",
        help=(
            "JSON object describing how the states were extracted (batch size, "
            "token caps); it enters the config hash because a different batching "
            "changes the stored bits"
        ),
    )
    args = parser.parse_args()

    train_groups = load_cache(args.cache, "train", "natural")
    eval_groups = load_cache(args.cache, "test", "natural")
    hidden_size = int(train_groups[sorted(train_groups)[0]]["states"].shape[-1])
    digest = hashlib.sha256()
    for k in sorted(train_groups):
        digest.update(train_groups[k]["states"].view(torch.int16).numpy().tobytes())
    cache_digest = digest.hexdigest()
    config = config_for(args, hidden_size)
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    config["config_hash"] = config_hash

    print(f"config_hash {config_hash}", flush=True)
    print(json.dumps(config, sort_keys=True), flush=True)
    if args.config_out:
        args.config_out.parent.mkdir(parents=True, exist_ok=True)
        args.config_out.write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
    if args.print_config_hash:
        return 0
    if args.output is None:
        raise SystemExit("--output is required unless --print-config-hash")

    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = SetHead(hidden_size).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sizes = {k: int(entry["states"].shape[0]) for k, entry in train_groups.items()}
    print(f"train items {sum(sizes.values())} by K {sizes}", flush=True)

    best = None
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        head.train()
        running = 0.0
        seen = 0
        for k, indices in epoch_batches(sizes, generator, BATCH_SIZE):
            entry = train_groups[k]
            selected, valid, target = batch_from(
                entry["states"], entry["gold"], torch.as_tensor(indices), device
            )
            logits = head(selected, valid)
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(target)
            seen += len(target)
        evaluation = evaluate(head, eval_groups, device)
        history.append(
            {
                "epoch": epoch,
                "train_nll": running / max(seen, 1),
                "eval_nll": evaluation["nll"],
                "eval_accuracy": evaluation["accuracy"],
            }
        )
        print(
            f"epoch {epoch} {history[-1]} {time.perf_counter() - started:.0f}s",
            flush=True,
        )
        if best is None or evaluation["nll"] < best["eval_nll"]:
            best = {
                "epoch": epoch,
                "eval_nll": evaluation["nll"],
                "state": {
                    key: value.detach().cpu()
                    for key, value in head.state_dict().items()
                },
            }

    head.load_state_dict(best["state"])
    final = evaluate(head, eval_groups, device)
    payload = {
        "config": config,
        "cache_digest": cache_digest,
        "history": history,
        "selected_epoch": best["epoch"],
        "selected_eval_nll": best["eval_nll"],
        "eval": {
            "items": final["items"],
            "nll": final["nll"],
            "accuracy": final["accuracy"],
        },
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    if args.gate:
        payload["validity"] = validity_gate(final)
        print(f"validity {json.dumps(payload['validity'], sort_keys=True)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config, "state_dict": best["state"]}, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"written {args.output} (epoch {best['epoch']})", flush=True)
    if args.gate and not payload["validity"]["passes"]:
        print(
            "VALIDITY GATE FAILED: the run is invalid, not to be reported", flush=True
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
