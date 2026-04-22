"""Tests for _is_zombie() zombie-process detection used in --replace wait loop."""

import os
import platform
from unittest.mock import mock_open, patch

import pytest

from gateway.run import _is_zombie


class TestIsZombie:
    def test_returns_false_on_non_linux(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        assert _is_zombie(12345) is False

    def test_returns_false_on_windows(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        assert _is_zombie(12345) is False

    def test_returns_false_when_proc_status_missing(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        assert _is_zombie(0x7FFFFFFF) is False

    def test_detects_zombie_state_from_proc_status(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        proc_status = "Name:\tpython3\nState:\tZ (zombie)\nPid:\t42\n"
        with patch("builtins.open", mock_open(read_data=proc_status)):
            assert _is_zombie(42) is True

    def test_returns_false_for_sleeping_process(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        proc_status = "Name:\tpython3\nState:\tS (sleeping)\nPid:\t42\n"
        with patch("builtins.open", mock_open(read_data=proc_status)):
            assert _is_zombie(42) is False

    def test_returns_false_for_running_process(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        proc_status = "Name:\tpython3\nState:\tR (running)\nPid:\t42\n"
        with patch("builtins.open", mock_open(read_data=proc_status)):
            assert _is_zombie(42) is False

    @pytest.mark.skipif(platform.system() != "Linux", reason="Linux only")
    def test_current_process_is_not_zombie(self):
        assert _is_zombie(os.getpid()) is False
