"""Regression tests for internal system-message leaks into user chats (#7921).

Two related issues are covered:

1. The ``TELEGRAM_HOME_CHANNEL`` onboarding prompt used to be auto-sent on
   the first turn of every fresh session, which surfaced in Telegram DMs
   as confusing unsolicited assistant-looking text.  It is now opt-in via
   ``HERMES_HOME_CHANNEL_PROMPT``.

2. ``InterruptedError`` handling in ``run_agent.py`` sets ``final_response``
   to raw internal control-flow strings like ``"Operation interrupted:
   waiting for model response (4.9s elapsed)."``.  The gateway must not
   forward these to messaging platforms; they look like broken assistant
   output to end users.  CLI/webhook callers still see the text for
   debuggability.
"""

import os
from unittest.mock import patch


def _home_channel_prompt_should_send(env):
    """Mirror the opt-in gate added in gateway/run.py for the home channel prompt."""
    with patch.dict(os.environ, env, clear=True):
        return os.getenv("HERMES_HOME_CHANNEL_PROMPT", "").lower() in ("true", "1", "yes")


def _should_suppress_interrupt_response(
    *, interrupted, final_response, platform_is_messaging
):
    """Mirror the suppression logic added in gateway/run.py.

    Returns True when the raw interrupt text would be dropped before
    reaching the adapter.
    """
    return bool(
        interrupted
        and (final_response or "").startswith("Operation interrupted")
        and platform_is_messaging
    )


class TestHomeChannelPromptOptIn:
    """The proactive home-channel prompt must default to silent (#7921)."""

    def test_default_unset_does_not_prompt(self):
        assert _home_channel_prompt_should_send({}) is False

    def test_explicit_false_does_not_prompt(self):
        assert _home_channel_prompt_should_send({"HERMES_HOME_CHANNEL_PROMPT": "false"}) is False

    def test_empty_string_does_not_prompt(self):
        assert _home_channel_prompt_should_send({"HERMES_HOME_CHANNEL_PROMPT": ""}) is False

    def test_opt_in_true_prompts(self):
        assert _home_channel_prompt_should_send({"HERMES_HOME_CHANNEL_PROMPT": "true"}) is True

    def test_opt_in_one_prompts(self):
        assert _home_channel_prompt_should_send({"HERMES_HOME_CHANNEL_PROMPT": "1"}) is True

    def test_opt_in_yes_prompts(self):
        assert _home_channel_prompt_should_send({"HERMES_HOME_CHANNEL_PROMPT": "yes"}) is True

    def test_opt_in_mixed_case_prompts(self):
        assert _home_channel_prompt_should_send({"HERMES_HOME_CHANNEL_PROMPT": "TRUE"}) is True


class TestInterruptResponseSuppression:
    """Raw interrupt-status strings must not reach messaging-platform users (#7921)."""

    def test_suppressed_on_messaging_platform_when_interrupted(self):
        assert _should_suppress_interrupt_response(
            interrupted=True,
            final_response="Operation interrupted: waiting for model response (4.9s elapsed).",
            platform_is_messaging=True,
        ) is True

    def test_suppressed_for_other_interrupt_variants(self):
        variants = [
            "Operation interrupted during retry (429, attempt 2/3).",
            "Operation interrupted: handling API error (ConnectError: timed out).",
            "Operation interrupted: retrying API call after error (retry 1/3).",
        ]
        for text in variants:
            assert _should_suppress_interrupt_response(
                interrupted=True,
                final_response=text,
                platform_is_messaging=True,
            ) is True, f"Expected suppression for: {text!r}"

    def test_not_suppressed_on_cli_platform(self):
        # LOCAL/WEBHOOK must still see the text for debuggability.
        assert _should_suppress_interrupt_response(
            interrupted=True,
            final_response="Operation interrupted: waiting for model response (1.0s elapsed).",
            platform_is_messaging=False,
        ) is False

    def test_not_suppressed_when_not_interrupted(self):
        assert _should_suppress_interrupt_response(
            interrupted=False,
            final_response="Operation interrupted: waiting for model response (1.0s elapsed).",
            platform_is_messaging=True,
        ) is False

    def test_not_suppressed_for_real_assistant_text(self):
        # Ordinary assistant text that happens to be returned alongside
        # interrupted=True should not be touched.
        assert _should_suppress_interrupt_response(
            interrupted=True,
            final_response="Here's the summary you asked for…",
            platform_is_messaging=True,
        ) is False

    def test_empty_final_response_not_suppressed(self):
        assert _should_suppress_interrupt_response(
            interrupted=True,
            final_response="",
            platform_is_messaging=True,
        ) is False
