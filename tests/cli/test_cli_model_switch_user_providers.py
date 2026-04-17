"""Regression test for /model --provider <user-defined> in the CLI.

Issue #6945: user-defined providers from `providers:` config could not be
resolved via `--provider <slug>` because `_handle_model_switch` only loaded
the config inside the no-args picker branch, leaving `user_providers=None`
on the explicit-switch path.
"""

from types import SimpleNamespace
from unittest.mock import patch

import yaml


def _make_cli():
    """Build a bare HermesCLI instance without running its real __init__."""
    import cli as cli_mod

    inst = cli_mod.HermesCLI.__new__(cli_mod.HermesCLI)
    inst.model = "gpt-4o"
    inst.provider = "openai"
    inst.base_url = ""
    inst.api_key = ""
    inst.requested_provider = ""
    inst.api_mode = ""
    inst._explicit_api_key = ""
    inst._explicit_base_url = ""
    inst.agent = None
    return inst


def test_handle_model_switch_passes_user_providers_for_explicit_provider(
    monkeypatch, tmp_path
):
    """`/model <name> --provider custom` must forward user_providers to switch_model."""
    user_providers = {
        "custom": {
            "name": "My Custom Endpoint",
            "api": "https://my-endpoint.example.com/v1",
            "api_key": "sk-xxx",
            "default_model": "my-model",
        }
    }

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"providers": user_providers}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    captured = {}

    def fake_switch_model(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            success=False,
            error_message="stop here",
            new_model="",
            target_provider="",
            api_key="",
            base_url="",
            api_mode="",
        )

    cli = _make_cli()

    with patch("hermes_cli.model_switch.switch_model", side_effect=fake_switch_model):
        cli._handle_model_switch("/model my-model --provider custom")

    assert captured.get("explicit_provider") == "custom"
    assert captured.get("user_providers") == user_providers, (
        "user_providers from config.yaml must be forwarded to switch_model so "
        "user-defined providers resolve via the --provider flag (issue #6945)."
    )
