"""Derive tool-selection choice items from open tool-calling datasets.

Workstream D of epic #207.  The raw dataset schema is a source, not the model
input: this module rewrites one row into a typed decision.  Every item is a
``TypedQuestion`` whose state is the system policy plus the conversation turns
before the decision point.

Decision points (CLI ``--decision``):

1. act or not, including the kind of non-call action: ``build_action_item``;
2. which tool to call next: ``build_item``;
3. per-argument closed set: ``build_argument_items``;
4. after-result next action: ``build_next_action_item``;
5. policy compliance: ``build_policy_item``.

Decision 2 uses the row's tool names and descriptions as criteria.  K is the
tool count; rows outside ``--min-k``..``--max-k``, rows without a call, and
rows whose first call is ``think`` are skipped.

Decision 3 takes the first non-``think`` function call.  For each argument it
mines candidates from every call to the same tool and argument name in the
inspected split, plus the tool schema's ``parameters.properties[arg].enum``
values.  The gold value is always the first criterion.  Strings become Choice
items; booleans become Noul items; numbers become Choice items over pooled
numeric values.  List and dict values are skipped as ``non_scalar``.  An
argument with fewer than ``--min-arg-candidates`` distinct values (and at
least two) is skipped as ``too_few_candidates``.  ``group`` is the argument
name, not the tool name, so group-level splits keep one argument together.

Decision 4 starts at the first ``observation``/``tool`` turn that directly
follows a function call.  The next non-empty turn supplies the label: a
function call becomes a tool Choice item that reuses the decision-2 criteria,
and a ``gpt``/``assistant`` turn becomes the four-way action Choice.  A row is
skipped when no agent action follows, or when a ``think`` turn appears before
the label (including before the observation), because the thought text can
name the label.

Source probes (2026-11-04):

- APIGen-MT-5k: public ``apigen-mt_5k.json``; 5000 rows, 14-16 tools, one
  observation after every call, 4898 rows with more than one call.  It is the
  only probed source that carries rendered observation turns, so decisions 3
  and 4 use it.
- BFCL: the Gorilla ``main`` branch now ships ``BFCL_v4_*`` JSONL, not v3.
  ``BFCL_v4_irrelevance.json`` (240 rows) has question turns plus a function
  list and no gold label; its schema can express a "no applicable tool" Noul
  item.  ``BFCL_v4_multi_turn_base.json`` (200 rows) has only user turns plus
  ``initial_config``/``path``, with no per-turn observations, so it needs the
  executor.  No adapter was added.
- tau-bench / tau2-bench: reachable.  tau-bench tasks are Python data in
  ``tau_bench/envs/*/tasks.py`` (user instruction plus gold actions);
  tau2-bench ``data/tau2/domains/*/tasks.json`` has user scenarios plus
  evaluation criteria.  Neither ships a rendered agent/observation
  transcript, so no decision-4 adapter was added.
- When2Call: public HF JSONL under ``nvidia/When2Call``; the 3652-row MCQ test
  split labels ``correct_answer`` as tool_call / request_for_info /
  cannot_answer.  It maps to decision 1, not 3 or 4, and no adapter was added.

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
from dataclasses import dataclass, field
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
ARGUMENT_INSTRUCTIONS = (
    "What value should the agent pass for the '{arg}' argument of the "
    "'{tool}' tool? Choose exactly one value."
)
BOOL_ARGUMENT_INSTRUCTIONS = (
    "Should the agent set the '{arg}' argument of the '{tool}' tool to true or false?"
)
ACTION_CRITERIA = {
    "call_tool": "Call a tool to make progress on the request.",
    "answer": "Answer the user directly without calling a tool.",
    "clarify": "Ask the user for information that is missing before acting.",
    "refuse": "Refuse or escalate because the request violates policy or is out of scope.",
}
REFUSE_CUES = (
    "cannot",
    "can't",
    "unable",
    "not able",
    "apolog",
    "not allowed",
    "can only",
    "must not",
    "won't be able",
    "against policy",
)
CLARIFY_CUES = (
    "could you",
    "can you provide",
    "please provide",
    "please share",
    "i need",
    "to assist you",
    "what is your",
    "could you please",
)
ACK_CUES = (
    "let me",
    "i'll",
    "i will",
    "sure",
    "okay",
    "one moment",
    "checking",
    "i can help",
    "happy to help",
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


@dataclass
class ArgumentPool:
    """Candidate values seen for one (tool, argument) pair."""

    strings: Counter[str] = field(default_factory=Counter)
    numbers: Counter[float] = field(default_factory=Counter)
    booleans: Counter[bool] = field(default_factory=Counter)
    enums: list[str] = field(default_factory=list)


def first_call_payload(
    conversations: Any,
) -> tuple[str, dict[str, Any], int] | None:
    """First non-think call: ``(tool_name, arguments, index)``.

    Returns None when the first call is ``think`` for the same reason as
    ``first_call``: the thought turn can name the tool or the arguments.
    """
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
            return None
        arguments = payload.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        return name, arguments, index
    return None


def tool_enums(tools: Any) -> dict[str, dict[str, list[str]]]:
    """Tool -> argument -> enum values declared by the tool schema."""
    parsed = _json(tools)
    if not isinstance(parsed, list):
        return {}
    result: dict[str, dict[str, list[str]]] = {}
    for tool in parsed:
        if not isinstance(tool, dict):
            continue
        name = str(tool.get("name") or "").strip()
        if not name:
            continue
        parameters = tool.get("parameters")
        if not isinstance(parameters, dict):
            continue
        properties = parameters.get("properties")
        if not isinstance(properties, dict):
            continue
        for arg, spec in properties.items():
            if not isinstance(spec, dict):
                continue
            raw = spec.get("enum")
            if not isinstance(raw, list):
                continue
            values: list[str] = []
            for item in raw:
                if isinstance(item, bool):
                    continue
                if isinstance(item, (str, int, float)):
                    text = str(item)
                    if text and text not in values:
                        values.append(text)
            if values:
                result.setdefault(name, {})[str(arg)] = values
    return result


def collect_argument_candidates(
    rows: list[dict],
) -> dict[tuple[str, str], ArgumentPool]:
    """Mine scalar argument values and schema enums across the split.

    Every call is inspected, not only the first call, so the candidate pool
    for a tool/argument pair reflects the whole inspected split.
    """
    pool: dict[tuple[str, str], ArgumentPool] = {}
    for row in rows:
        for tool, args in tool_enums(row.get("tools")).items():
            for arg, values in args.items():
                entry = pool.setdefault((tool, arg), ArgumentPool())
                for value in values:
                    if value not in entry.enums:
                        entry.enums.append(value)
        conversations = _json(row.get("conversations"))
        if not isinstance(conversations, list):
            continue
        for turn in conversations:
            if not isinstance(turn, dict) or turn.get("from") != "function_call":
                continue
            payload = _json(turn.get("value"))
            if not isinstance(payload, dict):
                continue
            tool = str(payload.get("name") or "").strip()
            if not tool or tool == "think":
                continue
            arguments = payload.get("arguments")
            if not isinstance(arguments, dict):
                continue
            for arg, value in arguments.items():
                key = (tool, str(arg))
                if isinstance(value, bool):
                    pool.setdefault(key, ArgumentPool()).booleans[value] += 1
                elif isinstance(value, (int, float)):
                    pool.setdefault(key, ArgumentPool()).numbers[float(value)] += 1
                elif isinstance(value, str) and value:
                    pool.setdefault(key, ArgumentPool()).strings[value] += 1
    return pool


def _number_key(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return repr(float(value))


def _string_candidates(pool: ArgumentPool, gold: str, k: int) -> list[str]:
    """Gold first, then enum values, then mined values by frequency."""
    ordered: list[str] = []
    seen = {gold}
    for value in pool.enums:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    for value, _count in sorted(
        pool.strings.items(), key=lambda item: (-item[1], item[0])
    ):
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return [gold] + ordered[: max(0, k - 1)]


def _numeric_candidates(pool: ArgumentPool, gold: float, k: int) -> list[str]:
    """Gold key first, then pooled numeric values by frequency."""
    gold_key = _number_key(gold)
    others: list[tuple[str, float, int]] = []
    seen = {gold_key}
    for value, count in pool.numbers.items():
        key = _number_key(value)
        if key in seen:
            continue
        seen.add(key)
        others.append((key, value, count))
    others.sort(key=lambda item: (-item[2], item[1]))
    return [gold_key] + [key for key, _value, _count in others[: max(0, k - 1)]]


def build_argument_items(
    row: dict,
    item_id: str,
    candidates: dict[tuple[str, str], ArgumentPool] | None = None,
    arg_k: int = 5,
    min_candidates: int = 2,
    max_chars: int = 8000,
) -> tuple[list[TypedQuestion], Counter]:
    """Decision point 3: one closed-set item per scalar argument.

    Returns the items and a Counter of skipped reasons.  ``group`` is the
    argument name.  A Choice item always has at least two criteria, even when
    ``min_candidates`` is set below two.
    """
    skipped: Counter = Counter()
    conversations = _json(row.get("conversations"))
    call = first_call_payload(conversations)
    if call is None:
        skipped["no_call"] += 1
        return [], skipped
    tool, arguments, index = call
    state = render_state(row.get("system"), conversations, index)
    if not state.strip():
        skipped["empty_state"] += 1
        return [], skipped
    if len(state) > max_chars:
        state = state[:max_chars].rsplit("\n", 1)[0]

    pool_map = candidates or {}
    required = max(2, min_candidates)
    items: list[TypedQuestion] = []
    for arg, gold in arguments.items():
        arg_name = str(arg)
        pool = pool_map.get((tool, arg_name), ArgumentPool())
        if isinstance(gold, bool):
            items.append(
                TypedQuestion(
                    id=f"apigen-arg-{item_id}-{arg_name}",
                    state=state,
                    question={
                        "type": "noul",
                        "instructions": BOOL_ARGUMENT_INSTRUCTIONS.format(
                            tool=tool, arg=arg_name
                        ),
                    },
                    label=bool(gold),
                    source="apigen_tool_argument",
                    split="train",
                    group=arg_name,
                )
            )
            continue
        if isinstance(gold, (int, float)):
            keys = _numeric_candidates(pool, float(gold), arg_k)
            if len(keys) < required:
                skipped["too_few_candidates"] += 1
                continue
            items.append(
                TypedQuestion(
                    id=f"apigen-arg-{item_id}-{arg_name}",
                    state=state,
                    question={
                        "type": "choice",
                        "instructions": ARGUMENT_INSTRUCTIONS.format(
                            tool=tool, arg=arg_name
                        ),
                        "criteria": {key: f"{arg_name} = {key}" for key in keys},
                    },
                    label=_number_key(float(gold)),
                    source="apigen_tool_argument",
                    split="train",
                    group=arg_name,
                )
            )
            continue
        if isinstance(gold, str) and gold:
            keys = _string_candidates(pool, gold, arg_k)
            if len(keys) < required:
                skipped["too_few_candidates"] += 1
                continue
            items.append(
                TypedQuestion(
                    id=f"apigen-arg-{item_id}-{arg_name}",
                    state=state,
                    question={
                        "type": "choice",
                        "instructions": ARGUMENT_INSTRUCTIONS.format(
                            tool=tool, arg=arg_name
                        ),
                        "criteria": {key: f"{arg_name} = {key}" for key in keys},
                    },
                    label=gold,
                    source="apigen_tool_argument",
                    split="train",
                    group=arg_name,
                )
            )
            continue
        skipped["non_scalar"] += 1

    if not items and not skipped:
        skipped["no_arguments"] += 1
    return items, skipped


def _is_think_turn(turn: Any) -> bool:
    if not isinstance(turn, dict) or turn.get("from") != "function_call":
        return False
    payload = _json(turn.get("value"))
    return (
        isinstance(payload, dict) and str(payload.get("name") or "").strip() == "think"
    )


def _next_nonempty(conversations: list, start: int) -> tuple[int, dict] | None:
    for index in range(start, len(conversations)):
        turn = conversations[index]
        if not isinstance(turn, dict):
            continue
        if str(turn.get("value") or "").strip():
            return index, turn
    return None


def build_next_action_item(
    row: dict,
    item_id: str,
    max_chars: int = 8000,
    skipped: Counter | None = None,
) -> TypedQuestion | None:
    """Decision point 4: after a tool result, what does the agent do next?

    Uses the first observation that directly follows a function call.  A
    ``think`` turn anywhere before the label causes a skip: its text can name
    the next tool or action.  Pass a Counter as ``skipped`` to collect
    per-reason counts.
    """

    def _skip(reason: str) -> TypedQuestion | None:
        if skipped is not None:
            skipped[reason] += 1
        return None

    conversations = _json(row.get("conversations"))
    if not isinstance(conversations, list):
        return _skip("no_conversations")

    observation: int | None = None
    for index, turn in enumerate(conversations):
        if not isinstance(turn, dict):
            continue
        if turn.get("from") not in ("observation", "tool"):
            continue
        previous = None
        for back in range(index - 1, -1, -1):
            candidate = conversations[back]
            if not isinstance(candidate, dict):
                continue
            if str(candidate.get("value") or "").strip():
                previous = candidate
                break
        if previous is not None and previous.get("from") == "function_call":
            observation = index
            break
    if observation is None:
        return _skip("no_observation")
    if any(_is_think_turn(turn) for turn in conversations[:observation]):
        return _skip("think_before_observation")

    nxt = _next_nonempty(conversations, observation + 1)
    if nxt is None:
        return _skip("no_later_action")
    _next_index, turn = nxt
    if _is_think_turn(turn):
        return _skip("think_before_label")

    state = render_state(row.get("system"), conversations, observation + 1)
    if not state.strip():
        return _skip("empty_state")
    if len(state) > max_chars:
        state = state[:max_chars].rsplit("\n", 1)[0]

    role = turn.get("from")
    if role == "function_call":
        payload = _json(turn.get("value"))
        if not isinstance(payload, dict):
            return _skip("bad_call_payload")
        name = str(payload.get("name") or "").strip()
        if not name or name == "think":
            return _skip("bad_call_payload")
        tools = tool_map(row.get("tools"))
        if tools is None or name not in tools:
            return _skip("call_not_in_tools")
        return TypedQuestion(
            id=f"apigen-next-{item_id}",
            state=state,
            question={
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": tools,
            },
            label=name,
            source="apigen_tool_next_action",
            split="train",
            group="tool",
        )
    if role in ("gpt", "assistant"):
        text = str(turn.get("value") or "").strip()
        if not text:
            return _skip("empty_assistant")
        return TypedQuestion(
            id=f"apigen-next-{item_id}",
            state=state,
            question={
                "type": "choice",
                "instructions": ACTION_INSTRUCTIONS,
                "criteria": ACTION_CRITERIA,
            },
            label=classify_assistant(text),
            source="apigen_tool_next_action",
            split="train",
            group="action",
        )
    return _skip("no_agent_action")


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
        choices=(1, 2, 3, 4, 5),
        default=2,
        help=(
            "1 = act-or-not action; 2 = which tool (default); "
            "3 = per-argument closed set; 4 = after-result next action; "
            "5 = policy compliance"
        ),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--min-k", type=int, default=5)
    parser.add_argument("--max-k", type=int, default=20)
    parser.add_argument("--arg-k", type=int, default=5)
    parser.add_argument("--min-arg-candidates", type=int, default=2)
    parser.add_argument("--max-chars", type=int, default=8000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stats", type=Path, default=None)
    args = parser.parse_args()

    if args.source != "apigen":
        raise SystemExit(
            f"unknown --source {args.source!r}; only 'apigen' is supported"
        )

    rows = load_apigen(args.split, args.limit)
    items: list[TypedQuestion] = []
    dropped: Counter = Counter()
    k_counts: Counter = Counter()
    action_counts: Counter = Counter()
    by_tool: Counter = Counter()
    by_argument: Counter = Counter()
    next_action_counts: Counter = Counter()
    argument_pool = collect_argument_candidates(rows) if args.decision == 3 else {}

    for index, row in enumerate(rows):
        item_id = f"{index:05d}"
        if args.decision == 3:
            new_items, skipped = build_argument_items(
                row,
                item_id,
                argument_pool,
                args.arg_k,
                args.min_arg_candidates,
                args.max_chars,
            )
            dropped.update(skipped)
            call = first_call_payload(_json(row.get("conversations")))
            tool = call[0] if call else None
            if not new_items:
                dropped["row_no_items"] += 1
            for item in new_items:
                items.append(item)
                if item.question.get("type") == "choice":
                    k_counts[len(item.question["criteria"])] += 1
                if tool:
                    by_tool[tool] += 1
                if item.group:
                    by_argument[item.group] += 1
            continue
        if args.decision == 4:
            item = build_next_action_item(row, item_id, args.max_chars, dropped)
            if item is None:
                continue
            items.append(item)
            next_action_counts[item.group or "unknown"] += 1
            k_counts[len(item.question["criteria"])] += 1
            action_counts[str(item.label)] += 1
            continue

        if args.decision == 1:
            item = build_action_item(row, item_id, args.max_chars)
        elif args.decision == 5:
            item = build_policy_item(row, item_id, args.max_chars)
        else:
            item = build_item(row, item_id, args.min_k, args.max_k, args.max_chars)
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
        "state_chars": {
            "min": lengths[0] if lengths else None,
            "median": lengths[len(lengths) // 2] if lengths else None,
            "max": lengths[-1] if lengths else None,
        },
        "out": str(args.out),
    }
    if args.decision == 3:
        stats["by_tool"] = dict(by_tool.most_common())
        stats["by_argument"] = dict(by_argument.most_common())
        stats["arg_k"] = args.arg_k
        stats["min_arg_candidates"] = args.min_arg_candidates
    elif args.decision == 4:
        stats["next_action_counts"] = dict(next_action_counts)
        stats["label_counts"] = dict(action_counts)
    else:
        stats["label_counts"] = dict(action_counts)
        stats["gold_tools"] = dict(
            Counter(item.label for item in items).most_common(20)
        )
    stats_path = args.stats or args.out.with_suffix(".stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=1, sort_keys=True) + "\n")
    print(json.dumps(stats, indent=1, sort_keys=True), flush=True)
    print(f"wrote {len(items)} items to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
