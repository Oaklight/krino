"""Tests for tool-selection construction from open tool-call datasets."""

from __future__ import annotations

import json
from collections import Counter

from data.format import TypedQuestion
from data.tool_call_choice import (
    build_argument_items,
    build_item,
    build_next_action_item,
    collect_argument_candidates,
    first_call,
    first_call_payload,
    render_state,
    tool_enums,
    tool_map,
)


def _tools(n: int = 6) -> list[dict]:
    return [{"name": f"tool_{i}", "description": f"does thing {i}"} for i in range(n)]


def _row(n_tools: int = 6, call_index: int = 2, call_name: str = "tool_3") -> dict:
    conversations = [
        {"from": "human", "value": "please help"},
        {"from": "gpt", "value": "let me check"},
    ]
    while len(conversations) < call_index:
        conversations.append({"from": "gpt", "value": "more context"})
    conversations.append(
        {
            "from": "function_call",
            "value": json.dumps({"name": call_name, "arguments": {}}),
        }
    )
    return {
        "system": "You are an agent.",
        "tools": json.dumps(_tools(n_tools)),
        "conversations": conversations,
    }


def test_tool_map_parses_json_and_deduplicates():
    mapping = tool_map(json.dumps(_tools(3)))
    assert mapping == {f"tool_{i}": f"does thing {i}" for i in range(3)}

    duplicated = json.dumps(_tools(2) + [{"name": "tool_0", "description": "dup"}])
    duplicated_map = tool_map(duplicated)
    assert duplicated_map is not None
    assert len(duplicated_map) == 2
    assert tool_map("not json") is None
    assert tool_map(json.dumps([])) is None


def test_first_call_rejects_a_think_turn_as_the_next_action():
    conversations = [
        {
            "from": "function_call",
            "value": json.dumps({"name": "think", "arguments": {}}),
        },
        {"from": "observation", "value": "thought"},
        {
            "from": "function_call",
            "value": json.dumps({"name": "tool_1", "arguments": {}}),
        },
    ]
    assert first_call(conversations) is None


def test_render_state_stops_before_the_call():
    row = _row()
    state = render_state(row["system"], row["conversations"], 2)
    assert state.startswith("System: You are an agent.")
    assert "human: please help" in state
    assert "gpt: let me check" in state
    assert "function_call" not in state


def test_build_item_uses_tool_names_as_criteria():
    row = _row(n_tools=6, call_index=2, call_name="tool_3")
    item = build_item(row, "00007", min_k=5, max_k=20)
    assert isinstance(item, TypedQuestion)
    assert item.id == "apigen-tool-00007"
    assert item.question["type"] == "choice"
    assert set(item.question["criteria"]) == {f"tool_{i}" for i in range(6)}
    assert item.label == "tool_3"
    assert item.question["criteria"][item.label] == "does thing 3"
    assert "human: please help" in item.state
    assert item.source == "apigen_tool_choice"


def test_build_item_rejects_bad_k_and_missing_calls():
    assert build_item(_row(n_tools=4), "1", min_k=5, max_k=20) is None
    assert build_item(_row(n_tools=21), "1", min_k=5, max_k=20) is None

    no_call = _row()
    no_call["conversations"] = [{"from": "human", "value": "hi"}]
    assert build_item(no_call, "1", min_k=5, max_k=20) is None

    wrong_tool = _row(call_name="not_a_tool")
    assert build_item(wrong_tool, "1", min_k=5, max_k=20) is None


def test_build_item_truncates_long_state():
    row = _row()
    row["system"] = "policy " * 2000
    item = build_item(row, "1", min_k=5, max_k=20, max_chars=500)
    assert item is not None
    assert len(item.state) <= 500


def test_classify_assistant_distinguishes_refuse_clarify_answer():
    from data.tool_call_choice import classify_assistant

    assert (
        classify_assistant("I'm sorry, but I cannot process that request.") == "refuse"
    )
    assert classify_assistant("Could you provide your user ID?") == "clarify"
    assert classify_assistant("I need the reservation ID to continue.") == "clarify"
    assert classify_assistant("Here are the details you asked for.") == "answer"


def test_derive_action_reads_the_first_agent_turn():
    from data.tool_call_choice import derive_action

    call = [
        {"from": "human", "value": "help"},
        {"from": "function_call", "value": json.dumps({"name": "tool_1"})},
    ]
    assert derive_action(call) == ("call_tool", 1, "tool_1")

    clarify = [
        {"from": "human", "value": "help"},
        {"from": "gpt", "value": "Could you provide your ID?"},
    ]
    assert derive_action(clarify) == ("clarify", 1, None)

    think = [
        {"from": "function_call", "value": json.dumps({"name": "think"})},
        {"from": "gpt", "value": "answer"},
    ]
    assert derive_action(think) is None


def test_build_action_item_uses_the_four_action_criteria():
    from data.tool_call_choice import ACTION_CRITERIA, build_action_item

    row = _row()
    item = build_action_item(row, "00003")
    assert item is not None
    assert item.id == "apigen-action-00003"
    assert set(item.question["criteria"]) == set(ACTION_CRITERIA)
    assert item.label == "call_tool"
    assert "human: please help" in item.state
    assert "function_call" not in item.state


def test_policy_item_labels_calls_true_and_refusals_false():
    from data.tool_call_choice import build_policy_item

    call_row = _row(call_index=2, call_name="tool_3")
    item = build_policy_item(call_row, "00001")
    assert item is not None
    assert item.question["type"] == "noul"
    assert item.label is True
    assert item.source == "apigen_policy_compliance"

    refusal = {
        "system": "Policy: flights already flown cannot be cancelled.",
        "tools": json.dumps(_tools(6)),
        "conversations": [
            {"from": "human", "value": "cancel my flown flight"},
            {"from": "gpt", "value": "I cannot process that cancellation."},
        ],
    }
    item = build_policy_item(refusal, "00002")
    assert item is not None
    assert item.label is False

    clarify = {
        "system": "Policy.",
        "tools": json.dumps(_tools(6)),
        "conversations": [
            {"from": "human", "value": "cancel"},
            {"from": "gpt", "value": "Could you provide your user ID?"},
        ],
    }
    assert build_policy_item(clarify, "00003") is None


def _arg_tools() -> list[dict]:
    return [
        {
            "name": "book",
            "description": "book an item",
            "parameters": {
                "type": "object",
                "properties": {
                    "cabin": {"type": "string", "enum": ["economy", "business"]},
                    "count": {"type": "integer"},
                    "insured": {"type": "boolean"},
                    "note": {"type": "string"},
                    "items": {"type": "array"},
                },
            },
        },
        {
            "name": "lookup",
            "description": "look up an item",
            "parameters": {"type": "object", "properties": {}},
        },
    ]


def _arg_row(
    cabin: str = "economy",
    count: int = 2,
    insured: bool = True,
    note: str = "x",
    items: tuple = (1, 2),
) -> dict:
    arguments = {
        "cabin": cabin,
        "count": count,
        "insured": insured,
        "note": note,
        "items": list(items),
    }
    return {
        "system": "You are an agent.",
        "tools": json.dumps(_arg_tools()),
        "conversations": [
            {"from": "human", "value": "please book"},
            {
                "from": "function_call",
                "value": json.dumps({"name": "book", "arguments": arguments}),
            },
        ],
    }


def _next_row(next_turn: dict, tools: list[dict] | None = None) -> dict:
    return {
        "system": "You are an agent.",
        "tools": json.dumps(tools or _tools(6)),
        "conversations": [
            {"from": "human", "value": "please help"},
            {
                "from": "function_call",
                "value": json.dumps({"name": "tool_3", "arguments": {}}),
            },
            {"from": "observation", "value": "result ok"},
            next_turn,
        ],
    }


def test_tool_enums_reads_schema_enums_only():
    assert tool_enums(json.dumps(_arg_tools())) == {
        "book": {"cabin": ["economy", "business"]}
    }
    assert tool_enums("not json") == {}
    assert tool_enums(json.dumps([{"name": "x", "parameters": {}}])) == {}


def test_collect_argument_candidates_mines_values_and_enums():
    rows = [
        _arg_row(cabin="economy", count=2, insured=True),
        _arg_row(cabin="business", count=3, insured=False),
        _arg_row(cabin="economy", count=1, insured=True),
    ]
    pool = collect_argument_candidates(rows)
    cabin = pool[("book", "cabin")]
    assert cabin.enums == ["economy", "business"]
    assert cabin.strings["economy"] == 2
    assert cabin.strings["business"] == 1
    assert pool[("book", "count")].numbers[2.0] == 1
    assert pool[("book", "insured")].booleans[True] == 2
    assert pool[("book", "note")].strings["x"] == 3
    # List-valued arguments are not mined into a scalar pool.
    assert ("book", "items") not in pool


def test_first_call_payload_returns_arguments_and_skips_think():
    row = _arg_row()
    payload = first_call_payload(row["conversations"])
    assert payload is not None
    tool, arguments, index = payload
    assert tool == "book"
    assert arguments["cabin"] == "economy"
    assert index == 1

    think = [
        {
            "from": "function_call",
            "value": json.dumps({"name": "think", "arguments": {}}),
        },
        {
            "from": "function_call",
            "value": json.dumps({"name": "book", "arguments": {}}),
        },
    ]
    assert first_call_payload(think) is None


def test_build_argument_items_builds_string_numeric_and_bool_items():
    rows = [
        _arg_row(cabin="economy", count=2, insured=True),
        _arg_row(cabin="business", count=3, insured=False),
        _arg_row(cabin="economy", count=1, insured=True),
    ]
    pool = collect_argument_candidates(rows)
    items, skipped = build_argument_items(rows[0], "00007", pool, arg_k=3)
    by_id = {item.id: item for item in items}

    cabin = by_id["apigen-arg-00007-cabin"]
    assert cabin.question["type"] == "choice"
    assert cabin.label == "economy"
    assert list(cabin.question["criteria"])[0] == "economy"
    assert set(cabin.question["criteria"]) == {"economy", "business"}
    assert cabin.source == "apigen_tool_argument"
    assert cabin.group == "cabin"
    assert "human: please book" in cabin.state
    assert "function_call" not in cabin.state

    count = by_id["apigen-arg-00007-count"]
    assert count.question["type"] == "choice"
    assert count.label == "2"
    assert set(count.question["criteria"]) == {"2", "1", "3"}
    assert count.question["criteria"]["2"] == "count = 2"

    insured = by_id["apigen-arg-00007-insured"]
    assert insured.question["type"] == "noul"
    assert insured.label is True
    assert insured.group == "insured"

    # "note" has one distinct value; "items" is a list.
    assert skipped["too_few_candidates"] == 1
    assert skipped["non_scalar"] == 1


def test_build_argument_items_respects_arg_k_and_min_candidates():
    rows = [_arg_row(count=n) for n in range(1, 7)]
    pool = collect_argument_candidates(rows)
    items, _skipped = build_argument_items(rows[0], "00009", pool, arg_k=4)
    count = next(item for item in items if item.group == "count")
    assert len(count.question["criteria"]) == 4
    assert count.label == "1"

    items, skipped = build_argument_items(
        rows[0], "00009", pool, arg_k=4, min_candidates=9
    )
    # Only the boolean argument survives: it does not need a candidate set.
    assert [item.group for item in items] == ["insured"]
    assert skipped["too_few_candidates"] == 3  # cabin, count, note
    assert skipped["non_scalar"] == 1


def test_build_argument_items_skips_rows_without_a_call():
    row = _arg_row()
    row["conversations"] = [{"from": "human", "value": "hi"}]
    items, skipped = build_argument_items(row, "00001", {})
    assert items == []
    assert skipped["no_call"] == 1


def test_build_next_action_item_uses_tool_choice_after_observation():
    next_call = {
        "from": "function_call",
        "value": json.dumps({"name": "tool_4", "arguments": {}}),
    }
    item = build_next_action_item(_next_row(next_call), "00011")
    assert isinstance(item, TypedQuestion)
    assert item.id == "apigen-next-00011"
    assert item.question["type"] == "choice"
    assert item.label == "tool_4"
    assert set(item.question["criteria"]) == {f"tool_{i}" for i in range(6)}
    assert item.group == "tool"
    assert item.source == "apigen_tool_next_action"
    assert "observation: result ok" in item.state
    assert "tool_4" not in item.state


def test_build_next_action_item_uses_action_choice_after_observation():
    item = build_next_action_item(
        _next_row({"from": "gpt", "value": "Here are the details."}),
        "00012",
    )
    assert item is not None
    assert item.group == "action"
    assert item.label == "answer"
    assert set(item.question["criteria"]) == {
        "call_tool",
        "answer",
        "clarify",
        "refuse",
    }
    assert "Here are the details." not in item.state


def test_build_next_action_item_skips_think_and_missing_actions():
    counter: Counter = Counter()
    think = {
        "from": "function_call",
        "value": json.dumps({"name": "think", "arguments": {"thought": "call tool_4"}}),
    }
    assert build_next_action_item(_next_row(think), "1", skipped=counter) is None
    assert counter["think_before_label"] == 1

    row = _next_row({"from": "gpt", "value": "answer"})
    row["conversations"] = row["conversations"][:-1]
    counter = Counter()
    assert build_next_action_item(row, "2", skipped=counter) is None
    assert counter["no_later_action"] == 1

    row = _next_row({"from": "human", "value": "thanks"})
    counter = Counter()
    assert build_next_action_item(row, "3", skipped=counter) is None
    assert counter["no_agent_action"] == 1

    no_obs = _next_row({"from": "gpt", "value": "answer"})
    no_obs["conversations"] = [no_obs["conversations"][0]]
    counter = Counter()
    assert build_next_action_item(no_obs, "4", skipped=counter) is None
    assert counter["no_observation"] == 1


def test_build_next_action_item_skips_a_think_turn_before_the_observation():
    row = _next_row({"from": "gpt", "value": "answer"})
    think = {
        "from": "function_call",
        "value": json.dumps({"name": "think", "arguments": {"thought": "next"}}),
    }
    row["conversations"].insert(1, think)
    counter = Counter()
    assert build_next_action_item(row, "5", skipped=counter) is None
    assert counter["think_before_observation"] == 1
