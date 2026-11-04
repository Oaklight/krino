"""Tests for tool-selection construction from open tool-call datasets."""

from __future__ import annotations

import json

from data.format import TypedQuestion
from data.tool_call_choice import (
    build_item,
    first_call,
    render_state,
    tool_map,
)


def _tools(n: int = 6) -> list[dict]:
    return [
        {"name": f"tool_{i}", "description": f"does thing {i}"}
        for i in range(n)
    ]


def _row(n_tools: int = 6, call_index: int = 2, call_name: str = "tool_3") -> dict:
    conversations = [
        {"from": "human", "value": "please help"},
        {"from": "gpt", "value": "let me check"},
    ]
    while len(conversations) < call_index:
        conversations.append({"from": "gpt", "value": "more context"})
    conversations.append(
        {"from": "function_call", "value": json.dumps({"name": call_name, "arguments": {}})}
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
    assert len(tool_map(duplicated)) == 2
    assert tool_map("not json") is None
    assert tool_map(json.dumps([])) is None


def test_first_call_rejects_a_think_turn_as_the_next_action():
    conversations = [
        {"from": "function_call", "value": json.dumps({"name": "think", "arguments": {}})},
        {"from": "observation", "value": "thought"},
        {"from": "function_call", "value": json.dumps({"name": "tool_1", "arguments": {}})},
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

    assert classify_assistant("I'm sorry, but I cannot process that request.") == "refuse"
    assert classify_assistant("Could you provide your user ID?") == "clarify"
    assert classify_assistant("I need the reservation ID to continue.") == "clarify"
    assert classify_assistant("Here are the details you asked for.") == "answer"


def test_derive_action_reads_the_first_agent_turn():
    from data.tool_call_choice import derive_action

    call = [{"from": "human", "value": "help"}, {"from": "function_call", "value": json.dumps({"name": "tool_1"})}]
    assert derive_action(call) == ("call_tool", 1, "tool_1")

    clarify = [{"from": "human", "value": "help"}, {"from": "gpt", "value": "Could you provide your ID?"}]
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
