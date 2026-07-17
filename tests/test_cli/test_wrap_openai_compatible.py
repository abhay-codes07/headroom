"""Tests for the generic `headroom wrap openai-compatible` command."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli import wrap as wrap_mod
from headroom.cli.main import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_wrap_openai_compatible_points_base_url_at_proxy(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The chosen CLI is launched with OPENAI_BASE_URL pointed at the proxy and
    the proxy forwarding to the default OpenAI upstream."""
    monkeypatch.chdir(tmp_path)

    captured: dict[str, Any] = {}

    def fake_launch_tool(**kwargs: Any) -> None:  # noqa: ANN003
        captured.update(kwargs)

    with patch.object(wrap_mod.shutil, "which", return_value="/usr/local/bin/devin"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            result = runner.invoke(
                main,
                ["wrap", "openai-compatible", "--port", "9000", "--bin", "devin", "--", "run"],
            )

    assert result.exit_code == 0, result.output
    assert captured["binary"] == "/usr/local/bin/devin"
    assert captured["agent_type"] == "openai-compatible"
    assert captured["tool_label"] == "OPENAI-COMPAT"
    assert captured["args"] == ("run",)
    assert captured["openai_api_url"] == "https://api.openai.com/v1"
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:9000/v1"


def test_wrap_openai_compatible_custom_env_var_and_upstream(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--base-url-env and --upstream are honored."""
    monkeypatch.chdir(tmp_path)

    captured: dict[str, Any] = {}

    def fake_launch_tool(**kwargs: Any) -> None:  # noqa: ANN003
        captured.update(kwargs)

    with patch.object(wrap_mod.shutil, "which", return_value="/opt/mycli"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            result = runner.invoke(
                main,
                [
                    "wrap",
                    "openai-compatible",
                    "--port",
                    "8100",
                    "--bin",
                    "mycli",
                    "--base-url-env",
                    "OPENAI_API_BASE",
                    "--upstream",
                    "https://api.example.com/v1",
                ],
            )

    assert result.exit_code == 0, result.output
    assert captured["openai_api_url"] == "https://api.example.com/v1"
    env = captured["env"]
    assert env["OPENAI_API_BASE"] == "http://127.0.0.1:8100/v1"
    # A custom base-url env var must not also rewrite OPENAI_BASE_URL to the proxy.
    assert env.get("OPENAI_BASE_URL") != "http://127.0.0.1:8100/v1"


def test_wrap_openai_compatible_missing_binary_errors(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    with patch.object(wrap_mod.shutil, "which", return_value=None):
        result = runner.invoke(main, ["wrap", "openai-compatible", "--bin", "nope-xyz"])

    assert result.exit_code == 1
    assert "'nope-xyz' not found in PATH" in result.output


def test_wrap_openai_compatible_requires_bin(runner: CliRunner) -> None:
    result = runner.invoke(main, ["wrap", "openai-compatible"])
    assert result.exit_code != 0
    assert "--bin" in result.output
