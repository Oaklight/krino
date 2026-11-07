"""Shared pytest configuration.

The ``feature_guard`` marker turns a test of a mask, an exclusion, or a guard
into a mutation check.  The test runs once with the feature on.  The hook then
re-runs the same test in a subprocess with the named environment kill switch
set.  A re-run that *passes* means the test does not depend on the feature and
asserts nothing about it, so the hook fails the test with that explanation.

The child process sets ``KRINO_FEATURE_GUARD_CHILD`` so the hook does not
recurse.  Only exit code 1 (tests failed) counts as the expected outcome; a
collection error, an internal error, or a timeout is reported as a broken
check rather than accepted as a failure.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_CHILD_ENV = "KRINO_FEATURE_GUARD_CHILD"
_TIMEOUT_SECONDS = 120


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "feature_guard(name): re-run this test with the named kill switch set; "
        "the test must fail when the guarded feature is disabled",
    )


def _switches(item: pytest.Item) -> list[str]:
    names = []
    for marker in item.iter_markers("feature_guard"):
        if not marker.args or not isinstance(marker.args[0], str):
            raise pytest.UsageError(
                f"{item.nodeid}: feature_guard needs one environment-variable name"
            )
        names.append(marker.args[0])
    return names


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item):
    # A new-style wrapper: the call phase has already run and raised if it
    # failed, so reaching this point means the test passed.
    result = yield
    if os.environ.get(_CHILD_ENV):
        return result
    switches = _switches(item)
    if not switches:
        return result

    env = dict(os.environ)
    env.update({name: "1" for name in switches})
    env[_CHILD_ENV] = "1"
    command = [
        sys.executable,
        "-m",
        "pytest",
        item.nodeid,
        "-q",
        "--no-header",
        "-p",
        "no:cacheprovider",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(item.config.rootpath),
            env=env,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            "feature_guard: the disabled-feature re-run of "
            f"{item.nodeid} timed out after {_TIMEOUT_SECONDS}s"
        )
    if completed.returncode != 1:
        pytest.fail(
            "feature_guard: the re-run with "
            f"{', '.join(switches)} set did not fail "
            f"(pytest exit code {completed.returncode}).  The test passes with "
            "its feature disabled, so it asserts nothing about that feature.\n"
            f"stdout:\n{completed.stdout[-4000:]}\n"
            f"stderr:\n{completed.stderr[-2000:]}"
        )
    return result
