"""Tests for run-setting resolution and the startup banner (issue #125)."""

from __future__ import annotations

from model.settings import format_settings, resolve_setting


class TestResolveSetting:
    def test_cli_wins_over_config(self) -> None:
        value, source = resolve_setting(3e-4, {"lr": 1e-3}, "lr", 1e-2)
        assert value == 3e-4
        assert source == "cli"

    def test_config_wins_over_default(self) -> None:
        value, source = resolve_setting(None, {"lr": 1e-3}, "lr", 1e-2)
        assert value == 1e-3
        assert source == "config"

    def test_default_when_neither_sets_it(self) -> None:
        value, source = resolve_setting(None, {}, "lr", 1e-2)
        assert value == 1e-2
        assert source == "default"

    def test_explicit_none_in_config_is_still_config(self) -> None:
        # A config that says "mlp_lr: null" chose that; it is not a default.
        value, source = resolve_setting(None, {"mlp_lr": None}, "mlp_lr", 3e-5)
        assert value is None
        assert source == "config"

    def test_falsy_cli_values_count_as_cli(self) -> None:
        # The resolution must test `is not None`, not truthiness: 0 and False
        # are meaningful settings (e.g. --mlp-layers 0 turns the projector off).
        for cli_value in (0, False, 0.0, ""):
            value, source = resolve_setting(cli_value, {"key": "config"}, "key", "default")
            assert value == cli_value
            assert source == "cli"

    def test_4b_sweep_confound_reports_cli(self) -> None:
        # The exact shape of the bug this issue exists for: the config says
        # 1e-3 and the run was meant to pass 3e-4 on the command line.
        value, source = resolve_setting(3e-4, {"lr": 0.001}, "lr", 1e-3)
        assert value == 3e-4
        assert source == "cli"
        # ... and when the flag was forgotten, the banner must say "config"
        # so a 1e-3 run cannot look like an intended 3e-4 run.
        value, source = resolve_setting(None, {"lr": 0.001}, "lr", 3e-4)
        assert value == 0.001
        assert source == "config"


class TestFormatSettings:
    def test_includes_value_and_source(self) -> None:
        line = format_settings([("lr", 3e-4, "cli"), ("epochs", 5, "config")])
        assert line == "Effective: lr=0.0003 (cli)  epochs=5 (config)"

    def test_single_setting(self) -> None:
        assert format_settings([("seed", 42, "default")]) == "Effective: seed=42 (default)"

    def test_none_value_renders(self) -> None:
        assert format_settings([("mlp_lr", None, "config")]) == "Effective: mlp_lr=None (config)"
