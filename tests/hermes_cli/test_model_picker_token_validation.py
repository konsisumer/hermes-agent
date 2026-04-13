"""Test that /model picker validates token formats before listing providers.

Covers: #8826 — providers with invalid token formats should not appear as
authenticated in the picker.
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli.model_switch import list_authenticated_providers

_COPILOT_ENV_CLEARED = {
    "COPILOT_GITHUB_TOKEN": "",
    "GH_TOKEN": "",
    "GITHUB_TOKEN": "",
}


def _find_provider(providers, slug):
    return next((p for p in providers if p["slug"] == slug), None)


def _no_auth_store():
    return {}


def _no_pool(slug):
    return SimpleNamespace(has_credentials=lambda: False)


def _isolate_copilot_creds(*extra_patches):
    """Stack patches that disable non-env-var credential sources for copilot."""
    return [
        patch("hermes_cli.auth._load_auth_store", _no_auth_store),
        patch("agent.credential_pool.load_pool", _no_pool),
    ]


@patch.dict(os.environ, {**_COPILOT_ENV_CLEARED, "GITHUB_TOKEN": "ghp_classic_pat_not_supported"}, clear=False)
@patch("hermes_cli.auth._load_auth_store", _no_auth_store)
@patch("agent.credential_pool.load_pool", _no_pool)
def test_copilot_hidden_with_classic_pat():
    """A classic PAT (ghp_*) should NOT make copilot appear as authenticated."""
    providers = list_authenticated_providers()
    copilot = _find_provider(providers, "copilot")
    assert copilot is None, (
        "copilot should not appear when only a classic PAT (ghp_*) is available"
    )


@patch.dict(os.environ, {**_COPILOT_ENV_CLEARED, "GITHUB_TOKEN": "gho_valid_oauth_token"}, clear=False)
def test_copilot_shown_with_oauth_token():
    """An OAuth token (gho_*) should make copilot appear as authenticated."""
    providers = list_authenticated_providers()
    copilot = _find_provider(providers, "copilot")
    assert copilot is not None, (
        "copilot should appear when a valid OAuth token (gho_*) is available"
    )


@patch.dict(os.environ, {**_COPILOT_ENV_CLEARED, "GITHUB_TOKEN": "github_pat_fine_grained"}, clear=False)
def test_copilot_shown_with_fine_grained_pat():
    """A fine-grained PAT (github_pat_*) should make copilot appear."""
    providers = list_authenticated_providers()
    copilot = _find_provider(providers, "copilot")
    assert copilot is not None, (
        "copilot should appear when a fine-grained PAT (github_pat_*) is available"
    )


@patch.dict(
    os.environ,
    {**_COPILOT_ENV_CLEARED, "GITHUB_TOKEN": "ghp_bad", "COPILOT_GITHUB_TOKEN": "gho_good"},
    clear=False,
)
def test_copilot_shown_when_valid_token_in_higher_priority_var():
    """If one env var has an invalid token but a higher-priority one is valid, copilot appears."""
    providers = list_authenticated_providers()
    copilot = _find_provider(providers, "copilot")
    assert copilot is not None, (
        "copilot should appear when COPILOT_GITHUB_TOKEN has a valid token"
    )


@patch.dict(
    os.environ,
    {**_COPILOT_ENV_CLEARED, "COPILOT_GITHUB_TOKEN": "ghp_bad", "GH_TOKEN": "ghp_also_bad", "GITHUB_TOKEN": "ghp_still_bad"},
    clear=False,
)
@patch("hermes_cli.auth._load_auth_store", _no_auth_store)
@patch("agent.credential_pool.load_pool", _no_pool)
def test_copilot_hidden_when_all_tokens_invalid():
    """If all env vars have classic PATs, copilot should not appear."""
    providers = list_authenticated_providers()
    copilot = _find_provider(providers, "copilot")
    assert copilot is None, (
        "copilot should not appear when all tokens are classic PATs (ghp_*)"
    )
