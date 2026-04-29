"""Tests for agent.auxiliary_client._try_custom_endpoint's anthropic_messages branch.

When a user configures a custom endpoint with ``api_mode: anthropic_messages``
(e.g. MiniMax, Zhipu GLM, LiteLLM in Anthropic-proxy mode), auxiliary tasks
(compression, web_extract, session_search, title generation) must use the
native Anthropic transport rather than being silently downgraded to an
OpenAI-wire client that speaks the wrong protocol.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in (
        "OPENAI_API_KEY", "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)


def _install_anthropic_adapter_mocks():
    """Patch build_anthropic_client so the test doesn't need the SDK."""
    fake_client = MagicMock(name="anthropic_client")
    return patch(
        "agent.anthropic_adapter.build_anthropic_client",
        return_value=fake_client,
    ), fake_client


def test_custom_endpoint_anthropic_messages_builds_anthropic_wrapper():
    """api_mode=anthropic_messages → returns AnthropicAuxiliaryClient, not OpenAI."""
    from agent.auxiliary_client import _try_custom_endpoint, AnthropicAuxiliaryClient

    with patch(
        "agent.auxiliary_client._resolve_custom_runtime",
        return_value=(
            "https://api.minimax.io/anthropic",
            "minimax-key",
            "anthropic_messages",
        ),
    ), patch(
        "agent.auxiliary_client._read_main_model",
        return_value="claude-sonnet-4-6",
    ):
        adapter_patch, fake_client = _install_anthropic_adapter_mocks()
        with adapter_patch:
            client, model = _try_custom_endpoint()

    assert isinstance(client, AnthropicAuxiliaryClient), (
        "Custom endpoint with api_mode=anthropic_messages must return the "
        f"native Anthropic wrapper, got {type(client).__name__}"
    )
    assert model == "claude-sonnet-4-6"
    # Wrapper should NOT be marked as OAuth — third-party endpoints are
    # always API-key authenticated.
    assert client.api_key == "minimax-key"
    assert client.base_url == "https://api.minimax.io/anthropic"


def test_custom_endpoint_anthropic_messages_falls_back_when_sdk_missing():
    """Graceful degradation when anthropic SDK is unavailable."""
    from agent.auxiliary_client import _try_custom_endpoint

    import_error = ImportError("anthropic package not installed")

    with patch(
        "agent.auxiliary_client._resolve_custom_runtime",
        return_value=("https://api.minimax.io/anthropic", "k", "anthropic_messages"),
    ), patch(
        "agent.auxiliary_client._read_main_model",
        return_value="claude-sonnet-4-6",
    ), patch(
        "agent.anthropic_adapter.build_anthropic_client",
        side_effect=import_error,
    ):
        client, model = _try_custom_endpoint()

    # Should fall back to an OpenAI-wire client rather than returning
    # (None, None) — the tool still needs to do *something*.
    assert client is not None
    assert model == "claude-sonnet-4-6"
    # OpenAI client, not AnthropicAuxiliaryClient.
    from agent.auxiliary_client import AnthropicAuxiliaryClient
    assert not isinstance(client, AnthropicAuxiliaryClient)


def test_custom_endpoint_chat_completions_still_uses_openai_wire():
    """Regression: default path (no api_mode) must remain OpenAI client."""
    from agent.auxiliary_client import _try_custom_endpoint, AnthropicAuxiliaryClient

    with patch(
        "agent.auxiliary_client._resolve_custom_runtime",
        return_value=("https://api.example.com/v1", "key", None),
    ), patch(
        "agent.auxiliary_client._read_main_model",
        return_value="my-model",
    ):
        client, model = _try_custom_endpoint()

    assert client is not None
    assert model == "my-model"
    assert not isinstance(client, AnthropicAuxiliaryClient)


# ── explicit_base_url + api_mode=anthropic_messages (path 3) ────────────────


def test_explicit_base_url_anthropic_messages_builds_anthropic_wrapper():
    """explicit_base_url + api_mode=anthropic_messages → AnthropicAuxiliaryClient."""
    from agent.auxiliary_client import resolve_provider_client, AnthropicAuxiliaryClient

    fake_anthropic = MagicMock(name="anthropic_client")
    with patch(
        "agent.anthropic_adapter.build_anthropic_client",
        return_value=fake_anthropic,
    ):
        client, model = resolve_provider_client(
            "custom",
            model="claude-sonnet-4-6",
            explicit_base_url="https://api.minimax.io/anthropic",
            explicit_api_key="minimax-key",
            api_mode="anthropic_messages",
        )

    assert isinstance(client, AnthropicAuxiliaryClient), (
        f"Expected AnthropicAuxiliaryClient, got {type(client).__name__}"
    )
    assert model == "claude-sonnet-4-6"
    assert client.api_key == "minimax-key"
    # URL must NOT be rewritten from /anthropic to /v1
    assert client.base_url == "https://api.minimax.io/anthropic"


def test_explicit_base_url_anthropic_messages_url_not_mangled():
    """/anthropic suffix must be preserved, not rewritten to /v1."""
    from agent.auxiliary_client import resolve_provider_client, AnthropicAuxiliaryClient

    captured = {}

    def fake_build(api_key, base_url, **_):
        captured["base_url"] = base_url
        return MagicMock(name="anthropic_client")

    with patch("agent.anthropic_adapter.build_anthropic_client", side_effect=fake_build):
        client, _ = resolve_provider_client(
            "custom",
            explicit_base_url="https://api.minimax.io/anthropic",
            explicit_api_key="k",
            api_mode="anthropic_messages",
        )

    assert captured.get("base_url") == "https://api.minimax.io/anthropic", (
        f"build_anthropic_client received mangled URL: {captured.get('base_url')!r}"
    )


def test_explicit_base_url_anthropic_messages_falls_back_when_sdk_missing():
    """Graceful degradation when anthropic SDK is absent."""
    from agent.auxiliary_client import resolve_provider_client, AnthropicAuxiliaryClient

    with patch(
        "agent.anthropic_adapter.build_anthropic_client",
        side_effect=ImportError("no anthropic"),
    ), patch("agent.auxiliary_client.OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        client, model = resolve_provider_client(
            "custom",
            model="some-model",
            explicit_base_url="https://api.minimax.io/anthropic",
            explicit_api_key="k",
            api_mode="anthropic_messages",
        )

    assert client is not None
    assert not isinstance(client, AnthropicAuxiliaryClient)
    # Fallback must use the /v1-rewritten URL for OpenAI-wire
    call_kwargs = mock_openai.call_args
    used_base = call_kwargs.kwargs.get("base_url") or (
        call_kwargs.args[1] if len(call_kwargs.args) > 1 else None
    )
    assert used_base is not None
    assert "/anthropic" not in str(used_base), (
        f"Fallback OpenAI client should use rewritten URL, got {used_base!r}"
    )


def test_explicit_base_url_chat_completions_still_uses_openai_wire():
    """Regression: explicit_base_url without api_mode stays on OpenAI-wire."""
    from agent.auxiliary_client import resolve_provider_client, AnthropicAuxiliaryClient

    with patch("agent.auxiliary_client.OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        client, model = resolve_provider_client(
            "custom",
            model="gpt-4o-mini",
            explicit_base_url="https://api.example.com/v1",
            explicit_api_key="k",
        )

    assert client is not None
    assert not isinstance(client, AnthropicAuxiliaryClient)
