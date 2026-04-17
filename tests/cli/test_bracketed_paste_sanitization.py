"""Tests for defensive stripping of leaked bracketed-paste markers.

Regression for the Ghostty-on-Arch paste bug where prompt_toolkit's
bracketed-paste parser could leak literal ``ESC[200~`` / ``ESC[201~``
wrappers (or their ``^[[200~`` / ``^[[201~`` caret-escape renderings)
into the user buffer and persisted paste files.
"""
from unittest.mock import MagicMock, patch

from run_agent import _strip_bracketed_paste_markers


class TestStripBracketedPasteMarkers:
    def test_clean_text_unchanged_identity(self):
        text = "hello world — normal paste with unicode café"
        assert _strip_bracketed_paste_markers(text) is text

    def test_empty_string(self):
        assert _strip_bracketed_paste_markers("") == ""

    def test_strips_real_start_marker(self):
        dirty = "\x1b[200~Search personal knowledge bases"
        assert _strip_bracketed_paste_markers(dirty) == "Search personal knowledge bases"

    def test_strips_real_end_marker(self):
        dirty = "pasted content\x1b[201~"
        assert _strip_bracketed_paste_markers(dirty) == "pasted content"

    def test_strips_both_real_wrappers(self):
        dirty = "\x1b[200~multi line\npaste body\x1b[201~"
        assert _strip_bracketed_paste_markers(dirty) == "multi line\npaste body"

    def test_strips_caret_escape_start_form(self):
        # The caret-escape rendering terminals and `cat -v` use for ESC
        dirty = "qmd is built in? ^[[200~Search personal notes"
        assert _strip_bracketed_paste_markers(dirty) == "qmd is built in? Search personal notes"

    def test_strips_caret_escape_end_form(self):
        dirty = "notes^[[201~"
        assert _strip_bracketed_paste_markers(dirty) == "notes"

    def test_strips_mixed_real_and_caret_escape(self):
        dirty = "\x1b[200~body one^[[201~ and ^[[200~body two\x1b[201~"
        assert _strip_bracketed_paste_markers(dirty) == "body one and body two"

    def test_strips_multiple_leaked_markers(self):
        dirty = "^[[200~a^[[200~b^[[201~c^[[201~"
        assert _strip_bracketed_paste_markers(dirty) == "abc"

    def test_does_not_strip_unrelated_escape_sequences(self):
        # ANSI color sequences and unrelated CSI codes must survive
        color = "\x1b[31mred\x1b[0m and [200 in brackets"
        assert _strip_bracketed_paste_markers(color) == color

    def test_does_not_strip_lookalike_text(self):
        # "200~" without the ESC[ / ^[[ prefix is just text
        text = "code 200~ok and [201~foo"
        assert _strip_bracketed_paste_markers(text) == text


class TestChatStripsBracketedPasteMarkers:
    """Integration: AIAgent.run_conversation should strip leaked markers."""

    @patch("run_agent.AIAgent._build_system_prompt")
    @patch("run_agent.AIAgent._interruptible_streaming_api_call")
    @patch("run_agent.AIAgent._interruptible_api_call")
    def test_user_message_markers_stripped(self, mock_api, mock_stream, mock_sys):
        from run_agent import AIAgent

        mock_sys.return_value = "system prompt"
        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_choice.message.tool_calls = None
        mock_choice.message.refusal = None
        mock_choice.finish_reason = "stop"
        mock_choice.message.reasoning_content = None
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        mock_response.model = "test-model"
        mock_response.id = "test-id"
        mock_stream.return_value = mock_response
        mock_api.return_value = mock_response

        agent = AIAgent(
            model="test/model",
            api_key="test-key",
            base_url="http://localhost:1234/v1",
            quiet_mode=True,
            skip_memory=True,
            skip_context_files=True,
        )
        agent.client = MagicMock()

        result = agent.run_conversation(
            user_message="\x1b[200~pasted body^[[201~",
            conversation_history=[],
        )

        for msg in result.get("messages", []):
            if msg.get("role") == "user":
                content = msg["content"]
                assert "\x1b[200~" not in content
                assert "\x1b[201~" not in content
                assert "^[[200~" not in content
                assert "^[[201~" not in content
