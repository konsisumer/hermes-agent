from unittest.mock import MagicMock, patch


def _patch_skin_default(banner):
    """Patch _skin_branding so tests don't depend on the active skin."""
    return patch.object(banner, "_skin_branding", side_effect=lambda key, default: default)


def test_format_banner_version_label_without_git_state():
    from hermes_cli import banner

    with patch.object(banner, "get_git_banner_state", return_value=None), \
         _patch_skin_default(banner):
        value = banner.format_banner_version_label()

    assert value == f"Hermes Agent v{banner.VERSION} ({banner.RELEASE_DATE})"


def test_format_banner_version_label_on_upstream_main():
    from hermes_cli import banner

    with patch.object(
        banner,
        "get_git_banner_state",
        return_value={"upstream": "b2f477a3", "local": "b2f477a3", "ahead": 0},
    ), _patch_skin_default(banner):
        value = banner.format_banner_version_label()

    assert value.endswith("· upstream b2f477a3")
    assert "local" not in value


def test_format_banner_version_label_with_carried_commits():
    from hermes_cli import banner

    with patch.object(
        banner,
        "get_git_banner_state",
        return_value={"upstream": "b2f477a3", "local": "af8aad31", "ahead": 3},
    ), _patch_skin_default(banner):
        value = banner.format_banner_version_label()

    assert "upstream b2f477a3" in value
    assert "local af8aad31" in value
    assert "+3 carried commits" in value


def test_format_banner_version_label_uses_skin_agent_name():
    from hermes_cli import banner

    with patch.object(banner, "get_git_banner_state", return_value=None), \
         patch.object(banner, "_skin_branding", side_effect=lambda key, default: "My Agent" if key == "agent_name" else default):
        value = banner.format_banner_version_label()

    assert value == f"My Agent v{banner.VERSION} ({banner.RELEASE_DATE})"


def test_get_git_banner_state_reads_origin_and_head(tmp_path):
    from hermes_cli import banner

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    results = {
        ("git", "rev-parse", "--short=8", "origin/main"): MagicMock(returncode=0, stdout="b2f477a3\n"),
        ("git", "rev-parse", "--short=8", "HEAD"): MagicMock(returncode=0, stdout="af8aad31\n"),
        ("git", "rev-list", "--count", "origin/main..HEAD"): MagicMock(returncode=0, stdout="3\n"),
    }

    def fake_run(cmd, **kwargs):
        key = tuple(cmd)
        if key not in results:
            raise AssertionError(f"unexpected command: {cmd}")
        return results[key]

    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run):
        state = banner.get_git_banner_state(repo_dir)

    assert state == {"upstream": "b2f477a3", "local": "af8aad31", "ahead": 3}
