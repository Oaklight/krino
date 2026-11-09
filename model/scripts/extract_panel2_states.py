"""Extract and cache the K option states of a panel-2 prompt (#340).

For each item the prompt is exactly :func:`render_panel2`'s, tokenised with
``add_special_tokens=False`` -- the same input panel 2 reads.  One forward
through the frozen tower gives a hidden state at the last token of every
option's text; those K states are what the set head consumes.

The cache is verifiable on its own:

* the key includes the model revision, the tokenizer digest, the prompt
  template digest and the extraction order (:func:`cache_key`);
* ``--verify N`` re-runs the first N batches twice and compares the states
  bit for bit before the main pass;
* the manifest records the byte size, the elapsed seconds and the
  verification result.

States are stored as the bf16 bit pattern in an ``int16`` array, so a reload is
exact.  A reversed-order run stores states in **natural** option order: display
position ``p`` holds natural option ``K-1-p``, and that is the order written.

Usage::

    python model/scripts/extract_panel2_states.py \\
        --snapshot <snapshot> --rows rows.jsonl --order natural \\
        --out <cache-dir> --verify 8 --shard 0/4
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.evaluation.panel2 import PANEL2_TEMPLATE  # noqa: E402
from model.evaluation.panel2_states import (  # noqa: E402
    build_batches,
    cache_key,
    data_digest,
    option_token_indices,
)
from model.src.backbone import load_causal_lm  # noqa: E402
from model.src.judgement_objective import run_backbone  # noqa: E402

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)


def tokenizer_digest(tokenizer) -> str:
    """sha256 over the tokenizer's own files, in a fixed order.

    ``tokenizer.init_kwargs`` records where the files came from; hashing the
    bytes is what makes two checkouts with the same name distinguishable.
    """
    name_or_path = Path(tokenizer.name_or_path)
    digest = hashlib.sha256()
    found = False
    for filename in TOKENIZER_FILES:
        path = name_or_path / filename
        if path.is_file():
            found = True
            digest.update(filename.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    if not found:
        raise SystemExit(
            f"no tokenizer files under {name_or_path}; cannot key the cache"
        )
    return digest.hexdigest()


def model_revision(snapshot: Path, override: str | None) -> str:
    """The model revision the cache is keyed on, or a hard failure."""
    if override:
        return override
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    if config.get("_commit_hash"):
        return str(config["_commit_hash"])
    for ref in (
        snapshot.parent / "refs" / "main",
        snapshot.parent.parent / "refs" / "main",
    ):
        if ref.is_file():
            return ref.read_text(encoding="utf-8").strip()
    raise SystemExit(
        "no model revision found (config.json has no _commit_hash and no refs/main); "
        "pass --revision"
    )


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_batch(model, tokenizer, device: str, batch: list[dict]) -> torch.Tensor:
    """One padded forward; returns ``(B, K, H)`` states in natural option order."""
    prompts = [item["prompt"] for item in batch]
    encoded = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        truncation=False,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    offsets = encoded.pop("offset_mapping")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        hidden = run_backbone(
            model, encoded["input_ids"], encoded["attention_mask"], bidirectional=False
        )
    batch_states = []
    for position, item in enumerate(batch):
        length = int(encoded["attention_mask"][position].sum())
        indices = option_token_indices(
            offsets[position][:length].tolist(), item["spans"]
        )
        states = hidden[position, indices, :].to(torch.bfloat16).cpu()
        if item["order"] == "reversed":
            states = torch.flip(states, dims=(0,))
        batch_states.append(states)
    return torch.stack(batch_states)


def save_group(
    out: Path,
    order: str,
    split: str,
    n_options: int,
    items: list[dict],
    states: torch.Tensor,
    shard: str,
) -> dict:
    """Write one ``(split, K)`` array file and return its manifest entry."""
    bits = states.view(torch.int16).numpy()
    path = out / f"states_{split}_K{n_options}_{order}{shard}.npz"
    np.savez(
        path,
        states=bits,
        rows_index=np.asarray([item["index"] for item in items], dtype=np.int64),
        ids=np.asarray([item["id"] for item in items]),
        gold=np.asarray([item["gold"] for item in items], dtype=np.int64),
    )
    return {
        "path": path.name,
        "split": split,
        "order": order,
        "n_options": n_options,
        "items": len(items),
        "states": int(states.numel() // states.shape[-1]),
        "hidden_size": int(states.shape[-1]),
        "bytes": path.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--order", choices=("natural", "reversed"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=65536,
        help="cap on the summed prompt tokens of one batch",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=8192,
        help="items with a longer prompt are skipped and counted",
    )
    parser.add_argument(
        "--verify",
        type=int,
        default=0,
        help="re-run this many of the leading batches twice and compare bits",
    )
    parser.add_argument("--shard", default="0/1", help="i/N, applied over the row list")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    args = parser.parse_args()

    shard_index, shard_count = (int(part) for part in args.shard.split("/"))
    if not 0 <= shard_index < shard_count:
        raise SystemExit(f"bad shard {args.shard}")
    shard_suffix = f"_s{shard_index}of{shard_count}" if shard_count > 1 else ""

    rows = load_rows(args.rows)
    for row in rows:
        row["order"] = args.order
    if shard_count > 1:
        rows = [
            row for index, row in enumerate(rows) if index % shard_count == shard_index
        ]
    print(f"{len(rows)} rows for order={args.order} shard={args.shard}", flush=True)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model, tokenizer = load_causal_lm(str(args.snapshot), dtype=dtype)
    tokenizer.padding_side = "right"
    device = str(next(model.parameters()).device)
    revision = model_revision(args.snapshot, args.revision)
    key = cache_key(revision, tokenizer_digest(tokenizer), args.order)
    print(f"model_revision {revision} cache_key {key}", flush=True)

    started = time.perf_counter()
    batches, skipped = build_batches(
        rows, tokenizer, args.batch_size, args.max_batch_tokens, args.max_prompt_tokens
    )
    if skipped:
        by_split: dict[str, int] = defaultdict(int)
        for item in skipped:
            by_split[item["split"]] += 1
        print(
            f"skipped {len(skipped)} items over {args.max_prompt_tokens} tokens "
            f"{dict(by_split)} longest {max(item['tokens'] for item in skipped)}",
            flush=True,
        )
    print(
        f"{len(batches)} batches prepared in {time.perf_counter() - started:.1f}s",
        flush=True,
    )

    verification = {
        "requested": args.verify,
        "batches": 0,
        "states": 0,
        "identical": None,
    }
    if args.verify:
        checked = batches[: args.verify]
        identical = True
        max_diff = 0.0
        states_checked = 0
        for batch in checked:
            first_states = extract_batch(model, tokenizer, device, batch)
            second_states = extract_batch(model, tokenizer, device, batch)
            identical = identical and torch.equal(first_states, second_states)
            max_diff = max(
                max_diff,
                float((first_states.float() - second_states.float()).abs().max()),
            )
            states_checked += int(first_states.shape[0] * first_states.shape[1])
        verification = {
            "requested": args.verify,
            "batches": len(checked),
            "states": states_checked,
            "batch_size": len(checked[0]),
            "n_option_counts": sorted({len(batch[0]["spans"]) for batch in checked}),
            "identical": bool(identical),
            "max_abs_diff": max_diff,
        }
        print(f"verification {verification}", flush=True)
        if not identical:
            raise SystemExit("re-extraction is not bit-identical; refusing to cache")

    args.out.mkdir(parents=True, exist_ok=True)
    collected: dict[tuple[str, int], dict] = {}
    entries: list[dict] = []
    done = 0
    for batch in batches:
        states = extract_batch(model, tokenizer, device, batch)
        split = rows[batch[0]["index"]]["split"]
        n_options = len(batch[0]["spans"])
        entry = collected.setdefault((split, n_options), {"items": [], "states": []})
        entry["items"].extend(batch)
        entry["states"].append(states)
        done += 1
        if done % 25 == 0 or done == len(batches):
            print(
                f"  {done}/{len(batches)} batches, {time.perf_counter() - started:.0f}s",
                flush=True,
            )

    for (split, n_options), entry in sorted(collected.items()):
        stacked = torch.cat(entry["states"], dim=0)
        entries.append(
            save_group(
                args.out,
                args.order,
                split,
                n_options,
                entry["items"],
                stacked,
                shard_suffix,
            )
        )
        print(f"wrote {split} K={n_options} {entries[-1]['items']} items", flush=True)

    elapsed = time.perf_counter() - started
    manifest_path = (
        args.out / f"manifest_{args.rows.stem}_{args.order}{shard_suffix}.json"
    )
    manifest = {
        "model": str(args.snapshot),
        "model_revision": revision,
        "tokenizer_digest": tokenizer_digest(tokenizer),
        "cache_key": key,
        "template_sha256": hashlib.sha256(PANEL2_TEMPLATE.encode("utf-8")).hexdigest(),
        "order": args.order,
        "dtype_on_disk": "bfloat16-bits-in-int16",
        "rows_file": str(args.rows),
        "rows_file_sha256": hashlib.sha256(args.rows.read_bytes()).hexdigest(),
        "rows": len(rows),
        "shard": args.shard,
        "batch_size": args.batch_size,
        "max_batch_tokens": args.max_batch_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "skipped_over_cap": len(skipped),
        "skipped_over_cap_items": skipped,
        "data_digest": data_digest(rows),
        "files": entries,
        "items": sum(entry["items"] for entry in entries),
        "states": sum(entry["states"] for entry in entries),
        "bytes": sum(entry["bytes"] for entry in entries),
        "elapsed_s": round(elapsed, 1),
        "verification": verification,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"manifest {manifest_path}: {manifest['items']} items, {manifest['states']} states, "
        f"{manifest['bytes'] / 1e9:.2f} GB, {elapsed:.0f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
