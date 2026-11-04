"""Tests for rendering a JevBench ``state`` as model-readable text.

35 of the 111 hard-tier JevBench items carry a structured state rather than a
string. The loader used to pass that object straight through, and because both
scorers build their prompt with an f-string, the model was fed Python's ``repr``
of the dict -- single-quoted keys, escaped unicode, no structure -- for a third
of the hardest tier. These tests pin the rendering that replaces it.
"""

from __future__ import annotations

import json

import pytest

from data.pipeline import render_jevbench_state


class TestStringsPassThrough:
    def test_plain_state_is_unchanged(self):
        assert render_jevbench_state("send it to Alex") == "send it to Alex"

    def test_empty_string_is_unchanged(self):
        assert render_jevbench_state("") == ""

    def test_surrounding_whitespace_is_not_rewritten(self):
        # The loader must not silently normalise text the author wrote.
        assert render_jevbench_state("  spaced  ") == "  spaced  "


class TestStructuredStates:
    def test_dict_becomes_indented_json(self):
        state = {"agent_policy": ["ask which one", "do not guess"]}
        rendered = render_jevbench_state(state)
        assert json.loads(rendered) == state
        assert "\n" in rendered, "the rendering must be multi-line to stay readable"

    def test_keys_use_double_quotes_not_python_repr(self):
        # The bug: f-string interpolation of a dict produces "{'key': ...}",
        # which no author wrote and which a model has to re-parse.
        rendered = render_jevbench_state({"policy": "x"})
        assert '"policy"' in rendered
        assert "'policy'" not in rendered

    def test_insertion_order_is_preserved(self):
        state = {"zeta": 1, "alpha": 2, "mid": 3}
        rendered = render_jevbench_state(state)
        assert rendered.index('"zeta"') < rendered.index('"alpha"') < rendered.index('"mid"')

    def test_nesting_survives(self):
        state = {"conversation": [{"role": "user", "text": "hello"}]}
        rendered = render_jevbench_state(state)
        assert json.loads(rendered) == state

    def test_unicode_is_not_escaped(self):
        # The items contain em dashes and curly quotes; escaping them costs
        # tokens and makes the text harder to read.
        rendered = render_jevbench_state({"rule": "ask — unless earlier messages"})
        assert "—" in rendered
        assert "\\u2014" not in rendered

    def test_list_state_is_rendered(self):
        rendered = render_jevbench_state(["first", "second"])
        assert json.loads(rendered) == ["first", "second"]


class TestOtherTypes:
    def test_none_becomes_a_string(self):
        # A missing state must not reach a prompt builder as None, which would
        # interpolate as "None" next to the instructions.
        assert render_jevbench_state(None) == "None"

    def test_number_becomes_a_string(self):
        assert render_jevbench_state(7) == "7"

    def test_round_trip_is_deterministic(self):
        state = {"a": [1, 2], "b": {"c": "d"}}
        assert render_jevbench_state(state) == render_jevbench_state(state)


class TestAgainstTheRealCorpus:
    def test_every_public_hard_state_renders_to_a_string(self):
        from pathlib import Path

        data_dir = Path(__file__).resolve().parents[1] / "data" / "benchmarks" / "jevbench"
        path = data_dir / "hard.jsonl"
        if not path.exists():
            pytest.skip("JevBench corpus is not present in this checkout")
        structured = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                rendered = render_jevbench_state(row["state"])
                assert isinstance(rendered, str)
                if isinstance(row["state"], dict):
                    structured += 1
                    assert json.loads(rendered) == row["state"]
        assert structured > 0, "this test is meant to cover the structured hard-tier items"
