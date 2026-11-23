"""Report the nine 20K scan checkpoints on the frozen canonical held-out.

Every arm was trained on the legacy 20K subset, while the held-out was drawn
from the canonical corpus.  A low number is evidence about transfer, not about
the arm having been trained badly.  The base column is the frozen backbone's
precomputed base distribution, not an arm.
"""

import os
import json
from pathlib import Path

ROOT = Path(os.path.expandvars("${KRINO_WORKTREE}"))
TAGS = [
    "lr1e-3_kl0", "lr1e-3_kl0.5", "lr1e-3_kl1.0",
    "lr3e-4_kl0", "lr3e-4_kl0.5", "lr3e-4_kl1.0",
    "lr1e-4_kl0", "lr1e-4_kl0.5", "lr1e-4_kl1.0",
]


def corrected(row):
    k = row["n_options"]
    return (float(bool(row["correct"])) - 1.0 / k) / (1.0 - 1.0 / k)


def summarize(rows):
    usable = [row for row in rows if row.get("n_options", 0) > 1]
    if not usable:
        return {"n": 0}
    values = [corrected(row) for row in usable]
    per_type = {}
    for q_type in ("noul", "choice", "score"):
        subset = [row for row in usable if row.get("q_type") == q_type]
        if subset:
            per_type[q_type] = {
                "n": len(subset),
                "cc": sum(corrected(row) for row in subset) / len(subset),
            }
    return {
        "n": len(usable),
        "cc": sum(values) / len(values),
        "accuracy": sum(bool(row["correct"]) for row in usable) / len(usable),
        "by_type": per_type,
    }


def main():
    base_rows = [
        json.loads(line)
        for line in (ROOT / "fullarm/heldout_base.pred.jsonl").read_text().splitlines()
        if line.strip()
    ]
    base = summarize(base_rows)
    # Arm dumps carry the 170 abstention rows with no label index and a
    # forced-wrong ``correct``; both sides must be the same 6,254 items or the
    # delta compares different populations.
    base_ids = {row["id"] for row in base_rows}
    report = {
        "base": base,
        "confound": (
            "All nine arms were trained on the legacy 20K subset; the frozen "
            "held-out was drawn from the canonical corpus. A low number is "
            "evidence about transfer to the canonical distribution, not that "
            "the arm was trained badly on its own corpus."
        ),
        "arms": {},
    }
    for tag in TAGS:
        path = ROOT / f"scan/heldout/{tag}.pred.jsonl"
        if not path.exists():
            report["arms"][tag] = {"status": "missing"}
            continue
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        dropped = sum(1 for row in rows if row["id"] not in base_ids)
        rows = [row for row in rows if row["id"] in base_ids]
        metrics = summarize(rows)
        metrics["unscorable_rows_dropped"] = dropped
        metrics["delta_cc"] = metrics["cc"] - base["cc"] if metrics.get("n") else None
        report["arms"][tag] = metrics

    out = ROOT / "scan/heldout/report.json"
    out.write_text(json.dumps(report, indent=1))

    print(f"base cc {base['cc']:.4f} n {base['n']}")
    print("tag                 n     cc      delta   noul    choice  score")
    for tag in TAGS:
        record = report["arms"][tag]
        if record.get("status") == "missing":
            print(f"{tag:18s} MISSING")
            continue
        by_type = record["by_type"]
        print(
            f"{tag:18s} {record['n']:5d} {record['cc']:.4f} "
            f"{record['delta_cc']:+.4f} "
            f"{by_type.get('noul', {}).get('cc', float('nan')):.4f}  "
            f"{by_type.get('choice', {}).get('cc', float('nan')):.4f}  "
            f"{by_type.get('score', {}).get('cc', float('nan')):.4f}"
        )


if __name__ == "__main__":
    main()
