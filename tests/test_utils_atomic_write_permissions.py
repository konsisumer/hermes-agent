"""Tests for atomic write file permission handling (issue #9239).

tempfile.mkstemp() creates files with mode 0600.  atomic_json_write and
atomic_yaml_write must restore umask-respecting permissions so that
group-shared directories (e.g. NixOS managed mode) work correctly.
"""

import os
import stat

import pytest

from utils import atomic_json_write, atomic_yaml_write


def _file_mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


class TestAtomicWritePermissions:
    def test_json_write_honors_umask(self, tmp_path):
        target = tmp_path / "out.json"
        old = os.umask(0o022)
        try:
            atomic_json_write(target, {"key": "value"})
        finally:
            os.umask(old)
        assert _file_mode(target) == 0o644

    def test_json_write_honors_restrictive_umask(self, tmp_path):
        """Simulates NixOS managed mode umask 0007."""
        target = tmp_path / "out.json"
        old = os.umask(0o007)
        try:
            atomic_json_write(target, {"key": "value"})
        finally:
            os.umask(old)
        assert _file_mode(target) == 0o660

    def test_yaml_write_honors_umask(self, tmp_path):
        target = tmp_path / "out.yaml"
        old = os.umask(0o022)
        try:
            atomic_yaml_write(target, {"key": "value"})
        finally:
            os.umask(old)
        assert _file_mode(target) == 0o644

    def test_yaml_write_honors_restrictive_umask(self, tmp_path):
        target = tmp_path / "out.yaml"
        old = os.umask(0o007)
        try:
            atomic_yaml_write(target, {"key": "value"})
        finally:
            os.umask(old)
        assert _file_mode(target) == 0o660

    def test_json_overwrite_updates_permissions(self, tmp_path):
        """When overwriting an existing file, permissions should reflect current umask."""
        target = tmp_path / "out.json"
        target.write_text("{}")
        target.chmod(0o600)

        old = os.umask(0o022)
        try:
            atomic_json_write(target, {"updated": True})
        finally:
            os.umask(old)
        assert _file_mode(target) == 0o644
