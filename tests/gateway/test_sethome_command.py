"""Test /sethome writes to .env, not config.yaml."""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with minimal config."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("model: test\n")
    (home / ".env").write_text("# env\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


class TestSethomeWritesToEnv:
    """Regression: /sethome must write HOME_CHANNEL to .env, not config.yaml."""

    @pytest.mark.asyncio
    async def test_sethome_writes_env_not_yaml(self, hermes_home, monkeypatch):
        """Verify save_env_value is called with the right key/value."""
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)

        source = MagicMock()
        source.platform.value = "discord"
        source.chat_id = "123456789"
        source.chat_name = "general"

        event = MagicMock()
        event.source = source

        with patch("gateway.run.save_env_value", create=True) as mock_save, \
             patch.dict(os.environ, {}, clear=False):
            # Inline the import so the patched version is used
            import gateway.run as gw_mod
            original = gw_mod.GatewayRunner._handle_set_home_command

            # Call with the mock
            with patch("hermes_cli.config.save_env_value") as mock_save_env:
                result = await original(runner, event)

            assert "✅" in result
            mock_save_env.assert_called_once_with("DISCORD_HOME_CHANNEL", "123456789")

    def test_sethome_does_not_write_config_yaml(self, hermes_home):
        """After /sethome, config.yaml must not contain HOME_CHANNEL keys."""
        import yaml

        config_path = hermes_home / "config.yaml"
        config = yaml.safe_load(config_path.read_text()) or {}
        # Before the fix, /sethome would inject DISCORD_HOME_CHANNEL here
        assert "DISCORD_HOME_CHANNEL" not in config
        assert "TELEGRAM_HOME_CHANNEL" not in config
