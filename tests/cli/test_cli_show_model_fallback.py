"""Regression tests for `/model` display after provider fallback (issue #7385).

`_try_activate_fallback()` in run_agent.py mutates agent.model / agent.provider
in place when the primary model fails.  The CLI's own self.model / self.provider
still reflect the originally configured primary, so reading them for the
"Current:" indicator shows a stale model name after fallback has taken over.
"""
from types import SimpleNamespace
from unittest.mock import patch

from cli import HermesCLI


def _make_cli(
    *,
    configured_model: str,
    configured_provider: str,
    agent_model: str | None,
    agent_provider: str | None,
):
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.model = configured_model
    cli_obj.provider = configured_provider
    cli_obj.requested_provider = configured_provider
    cli_obj._explicit_api_key = None
    cli_obj._explicit_base_url = None
    if agent_model is None and agent_provider is None:
        cli_obj.agent = None
    else:
        cli_obj.agent = SimpleNamespace(
            model=agent_model,
            provider=agent_provider,
        )
    return cli_obj


def _patch_model_modules():
    return patch.multiple(
        "hermes_cli.models",
        normalize_provider=lambda p: (p or "").lower(),
        list_available_providers=lambda: [],
        curated_models_for_provider=lambda p: [],
        _PROVIDER_LABELS={"anthropic": "Anthropic", "openrouter": "OpenRouter"},
        get_pricing_for_provider=lambda p: {},
        format_model_pricing_table=lambda *a, **kw: [],
    )


def test_current_reflects_fallback_model_when_agent_switched(capsys):
    cli_obj = _make_cli(
        configured_model="glm-5.1",
        configured_provider="anthropic",
        agent_model="qwen/qwen3.6-plus",
        agent_provider="openrouter",
    )

    with _patch_model_modules():
        cli_obj._show_model_and_providers()

    out = capsys.readouterr().out
    assert "Current: qwen/qwen3.6-plus via OpenRouter" in out
    assert "configured primary: glm-5.1" in out
    assert "fallback active" in out


def test_current_shows_configured_model_when_no_fallback(capsys):
    cli_obj = _make_cli(
        configured_model="glm-5.1",
        configured_provider="anthropic",
        agent_model="glm-5.1",
        agent_provider="anthropic",
    )

    with _patch_model_modules():
        cli_obj._show_model_and_providers()

    out = capsys.readouterr().out
    assert "Current: glm-5.1 via Anthropic" in out
    assert "fallback active" not in out
    assert "configured primary" not in out


def test_current_falls_back_to_self_before_agent_ready(capsys):
    cli_obj = _make_cli(
        configured_model="glm-5.1",
        configured_provider="anthropic",
        agent_model=None,
        agent_provider=None,
    )

    with _patch_model_modules():
        cli_obj._show_model_and_providers()

    out = capsys.readouterr().out
    assert "Current: glm-5.1 via Anthropic" in out
    assert "fallback active" not in out
