#!/usr/bin/env python3
"""Summarise training logs into a comparison table.

Reads the per-epoch lines that ``train_multitask.py`` prints and renders them
as a markdown table: one row per run, one column per eval epoch.  Runs are
grouped by backbone so a 0.6B row is never read as a delta against an 8B one,
and the first frozen ``last``-readout run in each group is used as the
baseline for the Δ column.

Logs are the only artefact the architecture sweep keeps — the runs are
launched without ``--checkpoint-dir``, so there is no ``history.json`` to
parse instead.

Usage:
    python model/scripts/sweep_report.py model/experiments/arch_*.log
    python model/scripts/sweep_report.py --by-type model/experiments/arch_*.log
    python model/scripts/sweep_report.py --json model/experiments/*.log
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# "  epoch 3/5: train_loss=1.3325 eval_loss=1.5292 eval_acc=0.4108 (773.8s)"
# Epochs without an eval checkpoint print only the train loss:
# "  epoch 1/20: train_loss=1.3510 (787.3s)"
_EPOCH_RE = re.compile(
    r"^\s*epoch\s+(?P<epoch>\d+)/(?P<total>\d+):\s+"
    r"train_loss=(?P<train>[\d.]+)"
    r"(?:\s+eval_loss=(?P<eval_loss>[\d.]+)\s+eval_acc=(?P<acc>[\d.]+))?"
    r"\s+\((?P<elapsed>[\d.]+)s\)"
)
_BY_TYPE_RE = re.compile(r"^\s+by_type:\s+(?P<body>.+)$")
_TYPE_ENTRY_RE = re.compile(r"(?P<type>\w+)=(?P<acc>[\d.]+)\((?P<n>\d+)\)")
_BACKBONE_RE = re.compile(r"^Backbone:\s+(?P<model>\S+)\s+\(")
_READOUT_RE = re.compile(r"^\s+Readout:\s+(?P<mode>\w+)")
_LORA_RE = re.compile(r"^\s+LoRA:\s+(?P<layers>\d+) layers \((?P<scope>.+?), r=")


@dataclass
class Epoch:
    """One eval checkpoint (or a bare train-loss epoch, when eval_every > 1)."""

    train_loss: float
    eval_loss: float | None = None
    accuracy: float | None = None
    elapsed_s: float | None = None
    by_type: dict[str, float] = field(default_factory=dict)


@dataclass
class Run:
    """One training log."""

    name: str
    path: Path
    backbone: str = "?"
    readout: str = "last"
    lora: str = "none"
    total_epochs: int = 0
    epochs: dict[int, Epoch] = field(default_factory=dict)

    @property
    def best(self) -> tuple[int, float] | None:
        """(epoch, accuracy) of the best eval checkpoint, or None."""
        scored = [(e, ep.accuracy) for e, ep in self.epochs.items() if ep.accuracy is not None]
        if not scored:
            return None
        return max(scored, key=lambda pair: pair[1])

    @property
    def config(self) -> str:
        """Short human label for the architecture knobs this run varied."""
        return f"readout={self.readout} lora={self.lora}"


def parse_log(path: Path) -> Run:
    """Parse one training log into a :class:`Run`."""
    run = Run(name=path.stem, path=path)
    last_epoch: int | None = None

    for line in path.read_text(errors="replace").splitlines():
        if match := _BACKBONE_RE.match(line):
            run.backbone = match["model"]
        elif match := _READOUT_RE.match(line):
            run.readout = match["mode"]
        elif match := _LORA_RE.match(line):
            run.lora = match["scope"]
        elif match := _EPOCH_RE.match(line):
            last_epoch = int(match["epoch"])
            run.total_epochs = int(match["total"])
            run.epochs[last_epoch] = Epoch(
                train_loss=float(match["train"]),
                eval_loss=float(match["eval_loss"]) if match["eval_loss"] else None,
                accuracy=float(match["acc"]) if match["acc"] else None,
                elapsed_s=float(match["elapsed"]),
            )
        elif (match := _BY_TYPE_RE.match(line)) and last_epoch is not None:
            run.epochs[last_epoch].by_type = {
                entry["type"]: float(entry["acc"])
                for entry in _TYPE_ENTRY_RE.finditer(match["body"])
            }

    return run


def _pct(value: float | None) -> str:
    return "·" if value is None else f"{value * 100:.2f}"


def _baseline_for(runs: list[Run]) -> Run | None:
    """The frozen last-readout run a group's deltas are measured against."""
    for run in runs:
        if run.readout == "last" and run.lora == "none":
            return run
    return None


def _sort_key(run: Run) -> tuple:
    """Frozen runs first, then by readout, then by name."""
    return (run.lora != "none", run.readout, run.name)


def render(runs: list[Run], by_type: bool = False) -> str:
    """Render runs as markdown, grouped by backbone."""
    out: list[str] = []
    groups: dict[str, list[Run]] = {}
    for run in runs:
        groups.setdefault(run.backbone, []).append(run)

    for backbone in sorted(groups):
        members = sorted(groups[backbone], key=_sort_key)
        n_epochs = max((max(r.epochs, default=0) for r in members), default=0)
        baseline = _baseline_for(members)

        out.append(f"### {backbone} ({len(members)} runs)")
        out.append("")
        header = ["run", "readout", "lora"] + [f"ep{e}" for e in range(1, n_epochs + 1)]
        header += ["best", "Δ vs baseline"]
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "|".join(["---"] * len(header)) + "|")

        for run in members:
            row = [run.name, run.readout, run.lora]
            row += [_pct(run.epochs[e].accuracy) if e in run.epochs else "" for e in range(1, n_epochs + 1)]
            best = run.best
            if best is None:
                row += ["", ""]
            else:
                row.append(f"{best[1] * 100:.2f} (ep{best[0]})")
                # A different epoch budget means a different LR schedule
                # (warmup and cosine both scale with `epochs`), so the two
                # bests are not on the same curve and a delta would mislead.
                if baseline is None or run is baseline or baseline.best is None:
                    row.append("—")
                elif run.total_epochs != baseline.total_epochs:
                    row.append(f"— ({run.total_epochs} vs {baseline.total_epochs} ep)")
                else:
                    delta = (best[1] - baseline.best[1]) * 100
                    row.append(f"{delta:+.2f} pp")
            out.append("| " + " | ".join(row) + " |")
        out.append("")

        if by_type:
            out.append("Per-type accuracy at each run's best epoch:")
            out.append("")
            out.append(_render_by_type(members))
            out.append("")

    return "\n".join(out).rstrip() + "\n"


def _render_by_type(runs: list[Run]) -> str:
    """Per-type accuracy at each run's best epoch."""
    types = sorted({t for run in runs for ep in run.epochs.values() for t in ep.by_type})
    if not types:
        return "_no per-type breakdown in these logs_"

    header = ["run", "best ep"] + types
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for run in runs:
        best = run.best
        if best is None:
            continue
        epoch = run.epochs[best[0]]
        row = [run.name, str(best[0])]
        row += [f"{epoch.by_type[t] * 100:.2f}" if t in epoch.by_type else "·" for t in types]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarise training logs")
    parser.add_argument("logs", nargs="+", type=Path, help="Training log files")
    parser.add_argument("--by-type", action="store_true", help="Add a per-type table")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of markdown")
    args = parser.parse_args()

    missing = [str(p) for p in args.logs if not p.is_file()]
    if missing:
        print(f"not a file: {', '.join(missing)}", file=sys.stderr)
        return 1

    runs = [parse_log(path) for path in args.logs]

    if args.json:
        payload = [
            {
                "name": run.name,
                "backbone": run.backbone,
                "readout": run.readout,
                "lora": run.lora,
                "epochs": {
                    epoch: {
                        "train_loss": ep.train_loss,
                        "eval_loss": ep.eval_loss,
                        "accuracy": ep.accuracy,
                        "by_type": ep.by_type,
                    }
                    for epoch, ep in sorted(run.epochs.items())
                },
            }
            for run in runs
        ]
        print(json.dumps(payload, indent=2))
        return 0

    print(render(runs, by_type=args.by_type))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
