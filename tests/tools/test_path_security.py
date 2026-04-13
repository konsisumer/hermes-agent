"""Tests for tools.path_security — validate_media_path."""

import tempfile
from pathlib import Path

import pytest

from tools.path_security import validate_media_path


class TestValidateMediaPath:
    def test_allows_tmp_files(self, tmp_path: Path):
        f = tmp_path / "image.png"
        f.write_bytes(b"\x89PNG")
        validate_media_path(f)

    def test_blocks_etc_passwd(self):
        with pytest.raises(PermissionError, match="sensitive system directory"):
            validate_media_path(Path("/etc/passwd"))

    def test_blocks_etc_hosts(self):
        with pytest.raises(PermissionError, match="sensitive system directory"):
            validate_media_path(Path("/etc/hosts"))

    def test_blocks_proc_self_environ(self):
        with pytest.raises(PermissionError, match="sensitive system directory"):
            validate_media_path(Path("/proc/self/environ"))

    def test_blocks_var_log(self):
        with pytest.raises(PermissionError):
            validate_media_path(Path("/var/log/syslog"))

    def test_blocks_traversal(self):
        with pytest.raises(PermissionError, match="traversal"):
            validate_media_path(Path("/tmp/../etc/passwd"))

    def test_blocks_home_ssh(self):
        with pytest.raises(PermissionError, match="outside allowed directories"):
            validate_media_path(Path.home() / ".ssh" / "id_rsa")

    def test_blocks_arbitrary_home_file(self):
        with pytest.raises(PermissionError, match="outside allowed directories"):
            validate_media_path(Path.home() / "secret.txt")

    def test_allows_extra_dirs(self, tmp_path: Path):
        custom = tmp_path / "custom"
        custom.mkdir()
        f = custom / "photo.jpg"
        f.write_bytes(b"\xff\xd8")
        validate_media_path(f, extra_allowed=[custom])

    def test_file_url_traversal_scenario(self):
        with pytest.raises(PermissionError):
            validate_media_path(Path("/etc/passwd"))
