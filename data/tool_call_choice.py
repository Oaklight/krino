"""Derive tool-selection choice items from open tool-calling datasets.

Workstream D of epic #207.  The raw dataset schema is a source, not the model
input: this module rewrites one row into a typed decision.  The first slice is
next-tool selection from APIGen-MT-5k:

- state: system policy plus every conversation turn before the first call;
- instructions: which tool should be called next;
- criteria: the row's tool names, with their descriptions as values;
- label: the tool named by the first ``function_call``.

K is the row's tool count (APIGen-MT rows carry 14); rows outside 5-20, rows
without a call, and calls to ``think`` are skipped.  Argument extraction is a
later slice: it needs per-argument closed sets or a candidate generator.

Usage:
    python -m data.tool_call_choice \
        --source apigen --split train --limit 2000 \
        --out data/benchmarks/apigen_tool_choice.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.format import TypedQuestion  # noqa: E402
from data.pipeline import save_jsonl  # noqa: E402

INSTRUCTIONS = (
    "Which tool should be called next to make progress on the user's request? "
    "Choose exactly one tool."
)

ACTION_INSTRUCTIONS = (
    "What should the agent do next? Choose exactly one action. Calling a tool, "
    "answering directly, asking for missing information, and refusing or "
    "escalating are different judgments."
)
POLICY_INSTRUCTIONS = (
    "Does the requested action comply with the policy in the system message? "
    "Answer yes if the agent may proceed, no if the request must be refused."
)
ACTION_CRITERIA = {
    "call_tool": "Call a tool to make progress on the request.",
    "answer": "Answer the user directly without calling a tool.",
    "clarify": "Ask the user for information that is missing before acting.",
    "refuse": "Refuse or escalate because the request violates policy or is out of scope.",
}
REFUSE_CUES = (
    "cannot", "can't", "unable", "not able", "apolog", "not allowed",
    "can only", "must not", "won't be able", "against policy",
)
CLARIFY_CUES = (
    "could you", "can you provide", "please provide", "please share",
    "i need", "to assist you", "what is your", "could you please",
)
ACK_CUES = (
    "let me", "i'll", "i will", "sure", "okay", "one moment", "checking",
    "i can help", "happy to help",
)


def _json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def tool_map(tools: Any) -> dict[str, str] | None:
    """Tool name -> description, or None when the schema is unusable."""
    parsed = _json(tools)
    if not isinstance(parsed, list):
        return None
    mapping: dict[str, str] = {}
    for tool in parsed:
        if not isinstance(tool, dict):
            continue
        name = str(tool.get("name") or "").strip()
        if not name or name in mapping:
            continue
        description = str(tool.get("description") or "").strip()
        mapping[name] = description or name
    return mapping or None


def first_call(conversations: Any) -> tuple[str, int] | None:
    """Return ``(tool_name, index)`` for the first non-think function call."""
    if not isinstance(conversations, list):
        return None
    for index, turn in enumerate(conversations):
        if not isinstance(turn, dict) or turn.get("from") != "function_call":
            continue
        payload = _json(turn.get("value"))
        if not isinstance(payload, dict):
            return None
        name = str(payload.get("name") or "").strip()
        if not name or name == "think":
            # A think turn is not a tool action.  Dropping the row keeps the
            # thought out of the state, where it would name the next tool.
            return None
        return name, index
    return None


def classify_assistant(text: str) -> str:
    """Classify a direct assistant turn as answer / clarify / refuse."""
    lowered = text.lower()
    if any(cue in lowered for cue in REFUSE_CUES):
        return "refuse"
    if text.strip().endswith("?") or any(cue in lowered for cue in CLARIFY_CUES):
        return "clarify"
    return "answer"


def derive_action(conversations: Any) -> tuple[str, int, str | None] | None:
    """First agent action: ``(action, index, tool_name)`` or None.

    ``action`` is one of call_tool / answer / clarify / refuse.  A ``think``
    turn before the first action drops the row: its text often names the
    action and would leak the label.
    """
    if not isinstance(conversations, list):
        return None
    fallback: tuple[str, int, str | None] | None = None
    for index, turn in enumerate(conversations):
        if not isinstance(turn, dict):
            continue
        role = turn.get("from")
        if role in ("human", "user", "observation", "tool"):
            continue
        if role == "function_call":
            payload = _json(turn.get("value"))
            if not isinstance(payload, dict):
                return None
            name = str(payload.get("name") or "").strip()
            if not name or name == "think":
                return None
            return "call_tool", index, name
        if role in ("gpt", "assistant"):
            text = str(turn.get("value") or "").strip()
            if not text:
                return None
            lowered = text.lower()
            if len(text) <= 160 and any(cue in lowered for cue in ACK_CUES):
                # "Let me check" is a preamble, not the decision; keep it as a
                # fallback in case no tool call follows.
                if fallback is None:
                    fallback = ("answer", index, None)
                continue
            return classify_assistant(text), index, None
        return None
    return fallback


def build_action_item(
    row: dict,
    item_id: str,
    max_chars: int = 8000,
) -> TypedQuestion | None:
    """Decision point 1: act or not, and which kind of non-call action."""
    action = derive_action(row.get("conversations"))
    if action is None:
        return None
    label, index, _tool = action
    state = render_state(row.get("system"), row.get("conversations"), index)
    if not state.strip():
        return None
    if len(state) > max_chars:
        state = state[:max_chars].rsplit("\n", 1)[0]
    return TypedQuestion(
        id=f"apigen-action-{item_id}",
        state=state,
        question={
            "type": "choice",
            "instructions": ACTION_INSTRUCTIONS,
            "criteria": ACTION_CRITERIA,
        },
        label=label,
        source="apigen_tool_action",
        split="train",
        group="act_or_not",
    )


def build_policy_item(
    row: dict,
    item_id: str,
    max_chars: int = 8000,
) -> TypedQuestion | None:
    """Decision point 5: does the request comply with the policy?

    Uses the first agent action as the label: a tool call is compliant, a
    refusal is non-compliant.  Clarifications and direct answers are skipped
    because they do not decide compliance.
    """
    action = derive_action(row.get("conversations"))
    if action is None:
        return None
    label, index, _tool = action
    if label not in ("call_tool", "refuse"):
        return None
    state = render_state(row.get("system"), row.get("conversations"), index)
    if not state.strip():
        return None
    if len(state) > max_chars:
        state = state[:max_chars].rsplit("\n", 1)[0]
    return TypedQuestion(
        id=f"apigen-policy-{item_id}",
        state=state,
        question={"type": "noul", "instructions": POLICY_INSTRUCTIONS},
        label=label == "call_tool",
        source="apigen_policy_compliance",
        split="train",
        group="policy",
    )


def render_state(system: Any, conversations: Any, stop: int) -> str:
    """System policy plus every turn before ``stop``."""
    lines: list[str] = []
    system_text = str(system or "").strip()
    if system_text:
        lines.append(f"System: {system_text}")
    for turn in (conversations or [])[:stop]:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("from") or "turn")
        value = str(turn.get("value") or "").strip()
        if value:
            lines.append(f"{role}: {value}")
    return "\n\n".join(lines)


def build_item(
    row: dict,
    item_id: str,
    min_k: int = 5,
    max_k: int = 20,
    max_chars: int = 8000,
) -> TypedQuestion | None:
    tools = tool_map(row.get("tools"))
    if tools is None or not min_k <= len(tools) <= max_k:
        return None
    call = first_call(row.get("conversations"))
    if call is None:
        return None
    gold, index = call
    if gold not in tools:
        return None
    state = render_state(row.get("system"), row.get("conversations"), index)
    if not state.strip():
        return None
    if len(state) > max_chars:
        state = state[:max_chars].rsplit("\n", 1)[0]
    return TypedQuestion(
        id=f"apigen-tool-{item_id}",
        state=state,
        question={
            "type": "choice",
            "instructions": INSTRUCTIONS,
            "criteria": tools,
        },
        label=gold,
        source="apigen_tool_choice",
        split="train",
        group=f"k{len(tools)}",
    )


def load_apigen(split: str, limit: int) -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset("Salesforce/APIGen-MT-5k", split=split)
    rows = []
    for row in dataset:
        rows.append(row)
        if limit and len(rows) >= limit:
            break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="apigen")
    parser.add_argument(
        "--decision",
        type=int,
        choices=(1, 2, 5),
        default=2,
        help="1 = act-or-not action; 2 = which tool (default); 5 = policy compliance",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--min-k", type=int, default=5)
    parser.add_argument("--max-k", type=int, default=20)
    parser.add_argument("--max-chars", type=int, default=8000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stats", type=Path, default=None)
    args = parser.parse_args()

    if args.source != "apigen":
        raise SystemExit(f"unknown --source {args.source!r}; only 'apigen' is supported")

    rows = load_apigen(args.split, args.limit)
    items: list[TypedQuestion] = []
    dropped: Counter = Counter()
    k_counts: Counter = Counter()
    action_counts: Counter = Counter()
    for index, row in enumerate(rows):
        if args.decision == 1:
            item = build_action_item(row, f"{index:05d}", args.max_chars)
        elif args.decision == 5:
            item = build_policy_item(row, f"{index:05d}", args.max_chars)
        else:
            item = build_item(row, f"{index:05d}", args.min_k, args.max_k, args.max_chars)
        if item is None:
            dropped["unusable_row"] += 1
            continue
        items.append(item)
        if item.question.get("type") == "choice":
            k_counts[len(item.question["criteria"])] += 1
        action_counts[item.label] += 1

    save_jsonl(items, args.out)
    lengths = sorted(len(item.state) for item in items)
    stats = {
        "source": args.source,
        "split": args.split,
        "inspected": len(rows),
        "selected": len(items),
        "dropped": dict(dropped),
        "decision": args.decision,
        "k_counts": dict(sorted(k_counts.items())),
        "label_counts": dict(action_counts),
        "state_chars": {
            "min": lengths[0] if lengths else None,
            "median": lengths[len(lengths) // 2] if lengths else None,
            "max": lengths[-1] if lengths else None,
        },
        "gold_tools": dict(Counter(item.label for item in items).most_common(20)),
        "out": str(args.out),
    }
    stats_path = args.stats or args.out.with_suffix(".stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=1, sort_keys=True) + "\n")
    print(json.dumps(stats, indent=1, sort_keys=True), flush=True)
    print(f"wrote {len(items)} items to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
