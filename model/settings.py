"""Resolution of run settings from CLI flags, config files and defaults.

Kept free of torch and transformers so it can be unit-tested on the CI hosts,
which install neither.
"""

from __future__ import annotations

from typing import Any


def resolve_setting(
    cli_value: Any, cfg: dict, key: str, default: Any
) -> tuple[Any, str]:
    """Resolve a setting from the CLI, then the config, then a default.

    Returns the value together with which source supplied it.  Recording that
    source at startup is what was missing when six Reranker-4B sweep runs
    trained at the config's 1e-3 instead of the intended CLI 3e-4 and looked
    completely normal in their logs (issue #125).

    Args:
        cli_value: Value from argparse, or None when the flag was not given.
        cfg: Parsed config mapping.
        key: Config key to fall back to.
        default: Value to use when neither the CLI nor the config set it.

    Returns:
        Tuple of (value, source) where source is "cli", "config" or "default".
    """
    if cli_value is not None:
        return cli_value, "cli"
    if key in cfg:
        return cfg[key], "config"
    return default, "default"


def describe_effective_length(
    requested: int | None, effective: int, requested_source: str
) -> str:
    """Describe how the effective sequence length was chosen.

    ``max_length`` is resolved in two stages: the CLI/config value, then a
    clamp to the GPU's safe length and the backbone's native maximum inside
    ``DecisionModel``.  Only the second value is what training uses, so the
    banner has to report it.

    Args:
        requested: Value from the CLI or config, or None for auto-detect.
        effective: Length training will actually use.
        requested_source: Source returned by :func:`resolve_setting`.

    Returns:
        A source label: ``"auto"``, ``"<source>, clamped from <requested>"``,
        or the requested source unchanged.
    """
    if requested is None:
        return "auto"
    if effective != requested:
        return f"{requested_source}, clamped from {requested}"
    return requested_source


def format_settings(settings: list[tuple[str, Any, str]]) -> str:
    """Format resolved settings as a single startup banner line.

    Args:
        settings: Triples of (name, value, source).

    Returns:
        A line such as ``lr=0.0003 (cli)  epochs=5 (config)``.
    """
    return "Effective: " + "  ".join(
        f"{name}={value} ({source})" for name, value, source in settings
    )
