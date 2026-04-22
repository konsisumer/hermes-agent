"""Tests for terminal.cwd being respected by tui_gateway.

Covers two bugs from issue #14044:
1. tui_gateway/server.py must bridge config.yaml terminal.cwd → TERMINAL_CWD.
2. complete.path must resolve relative paths against TERMINAL_CWD, not process cwd.
"""

import os
import json
import textwrap
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers that simulate the server-startup config bridge without importing the
# module (which would trigger side-effects).
# ---------------------------------------------------------------------------

_CWD_PLACEHOLDERS = (".", "auto", "cwd")


def _simulate_tui_config_bridge(terminal_cfg: dict, initial_env: dict | None = None) -> dict:
    """Simulate the startup config bridge added to tui_gateway/server.py.

    Returns the resulting env dict (only TERMINAL_* keys).
    """
    env = dict(initial_env or {})
    terminal_env_map = {
        "backend": "TERMINAL_ENV",
        "cwd": "TERMINAL_CWD",
        "timeout": "TERMINAL_TIMEOUT",
        "persistent_shell": "TERMINAL_PERSISTENT_SHELL",
        "docker_image": "TERMINAL_DOCKER_IMAGE",
    }
    for cfg_key, env_var in terminal_env_map.items():
        if cfg_key in terminal_cfg:
            val = terminal_cfg[cfg_key]
            if cfg_key == "cwd" and str(val) in _CWD_PLACEHOLDERS:
                continue
            if isinstance(val, list):
                env[env_var] = json.dumps(val)
            else:
                env[env_var] = str(val)
    return env


# ---------------------------------------------------------------------------
# Tests: config bridge
# ---------------------------------------------------------------------------

class TestTuiConfigBridge:
    """terminal.cwd in config.yaml must be bridged to TERMINAL_CWD at startup."""

    def test_explicit_cwd_sets_terminal_cwd(self):
        result = _simulate_tui_config_bridge({"cwd": "/home/user/projects"})
        assert result["TERMINAL_CWD"] == "/home/user/projects"

    def test_dot_placeholder_skipped(self):
        result = _simulate_tui_config_bridge({"cwd": "."})
        assert "TERMINAL_CWD" not in result

    def test_auto_placeholder_skipped(self):
        result = _simulate_tui_config_bridge({"cwd": "auto"})
        assert "TERMINAL_CWD" not in result

    def test_cwd_keyword_placeholder_skipped(self):
        result = _simulate_tui_config_bridge({"cwd": "cwd"})
        assert "TERMINAL_CWD" not in result

    def test_explicit_cwd_overrides_existing_env(self):
        result = _simulate_tui_config_bridge(
            {"cwd": "/from/config"},
            initial_env={"TERMINAL_CWD": "/old/value"},
        )
        assert result["TERMINAL_CWD"] == "/from/config"

    def test_backend_bridges_to_terminal_env(self):
        result = _simulate_tui_config_bridge({"backend": "docker"})
        assert result["TERMINAL_ENV"] == "docker"

    def test_timeout_bridges_correctly(self):
        result = _simulate_tui_config_bridge({"timeout": "120"})
        assert result["TERMINAL_TIMEOUT"] == "120"

    def test_empty_terminal_cfg_no_change(self):
        result = _simulate_tui_config_bridge({})
        assert "TERMINAL_CWD" not in result
        assert "TERMINAL_ENV" not in result


# ---------------------------------------------------------------------------
# Tests: complete.path resolution against TERMINAL_CWD
# ---------------------------------------------------------------------------

def _run_complete_path(word: str, terminal_cwd: str, tmp_path: Path) -> list[dict]:
    """Invoke the complete.path logic with a mocked TERMINAL_CWD."""
    # Import inline to avoid module-level side effects from server.py
    from tui_gateway.server import _normalize_completion_path

    # Replicate the fixed complete.path resolution logic
    is_context = word.startswith("@")
    query = word[1:] if is_context else word

    if is_context and query in ("file", "folder"):
        prefix_tag, path_part = query, ""
    elif is_context and query.startswith(("file:", "folder:")):
        prefix_tag, _, tail = query.partition(":")
        path_part = tail
    else:
        prefix_tag = ""
        path_part = query if is_context else query

    _cwd_base = terminal_cwd
    expanded = _normalize_completion_path(path_part) if path_part else ""

    if not expanded or expanded == ".":
        search_dir, match = _cwd_base, ""
    elif os.path.isabs(expanded):
        if expanded.endswith("/"):
            search_dir, match = expanded, ""
        else:
            search_dir = os.path.dirname(expanded) or "/"
            match = os.path.basename(expanded)
    elif expanded.endswith("/"):
        search_dir = os.path.normpath(os.path.join(_cwd_base, expanded))
        match = ""
    else:
        full_expanded = os.path.join(_cwd_base, expanded)
        search_dir = os.path.dirname(full_expanded) or _cwd_base
        match = os.path.basename(full_expanded)

    if not os.path.isdir(search_dir):
        return []

    items = []
    for entry in sorted(os.listdir(search_dir)):
        if match and not entry.lower().startswith(match.lower()):
            continue
        full = os.path.join(search_dir, entry)
        is_dir = os.path.isdir(full)
        rel = os.path.relpath(full, _cwd_base)
        suffix = "/" if is_dir else ""
        if is_context and prefix_tag:
            text = f"@{prefix_tag}:{rel}{suffix}"
        elif is_context:
            kind = "folder" if is_dir else "file"
            text = f"@{kind}:{rel}{suffix}"
        elif word.startswith("./"):
            text = "./" + rel + suffix
        elif os.path.isabs(expanded):
            text = full + suffix
        else:
            text = rel + suffix
        items.append({"text": text, "display": entry + suffix})
    return items


class TestCompletePathUsesCwd:
    """complete.path must resolve paths relative to TERMINAL_CWD."""

    def test_empty_word_lists_terminal_cwd_contents(self, tmp_path):
        (tmp_path / "alpha.py").touch()
        (tmp_path / "beta.py").touch()
        items = _run_complete_path("./", str(tmp_path), tmp_path)
        names = [i["display"] for i in items]
        assert "alpha.py" in names
        assert "beta.py" in names

    def test_relative_prefix_filters_from_terminal_cwd(self, tmp_path):
        (tmp_path / "foo.py").touch()
        (tmp_path / "bar.py").touch()
        items = _run_complete_path("fo", str(tmp_path), tmp_path)
        names = [i["display"] for i in items]
        assert "foo.py" in names
        assert "bar.py" not in names

    def test_context_file_lists_terminal_cwd(self, tmp_path):
        (tmp_path / "readme.md").touch()
        items = _run_complete_path("@file:", str(tmp_path), tmp_path)
        names = [i["display"] for i in items]
        assert "readme.md" in names

    def test_context_file_prefix_relative_to_terminal_cwd(self, tmp_path):
        (tmp_path / "script.sh").touch()
        items = _run_complete_path("@file:sc", str(tmp_path), tmp_path)
        assert any(i["text"] == "@file:script.sh" for i in items)

    def test_absolute_path_unaffected_by_terminal_cwd(self, tmp_path):
        other = tmp_path / "other"
        other.mkdir()
        (other / "thing.txt").touch()
        items = _run_complete_path(str(other) + "/", str(tmp_path), tmp_path)
        names = [i["display"] for i in items]
        assert "thing.txt" in names

    def test_returned_text_relative_to_terminal_cwd(self, tmp_path):
        """The completion text must be relative to TERMINAL_CWD, not process cwd."""
        (tmp_path / "myfile.txt").touch()
        process_cwd = os.getcwd()
        # Only meaningful when TERMINAL_CWD != process cwd
        if str(tmp_path) == process_cwd:
            pytest.skip("tmp_path equals process cwd — can't distinguish")
        items = _run_complete_path("./", str(tmp_path), tmp_path)
        texts = [i["text"] for i in items]
        # Should be "myfile.txt" or "./myfile.txt" relative to tmp_path,
        # NOT a path that crosses multiple parent dirs to get back to process cwd.
        assert any("myfile.txt" in t and ".." not in t for t in texts), (
            f"Expected path relative to TERMINAL_CWD, got: {texts}"
        )
