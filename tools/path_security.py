"""Shared path validation helpers for tool implementations.

Extracts the ``resolve() + relative_to()`` and ``..`` traversal check
patterns previously duplicated across skill_manager_tool, skills_tool,
skills_hub, cronjob_tools, and credential_files.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


def validate_within_dir(path: Path, root: Path) -> Optional[str]:
    """Ensure *path* resolves to a location within *root*.

    Returns an error message string if validation fails, or ``None`` if the
    path is safe.  Uses ``Path.resolve()`` to follow symlinks and normalize
    ``..`` components.

    Usage::

        error = validate_within_dir(user_path, allowed_root)
        if error:
            return json.dumps({"error": error})
    """
    try:
        resolved = path.resolve()
        root_resolved = root.resolve()
        resolved.relative_to(root_resolved)
    except (ValueError, OSError) as exc:
        return f"Path escapes allowed directory: {exc}"
    return None


def has_traversal_component(path_str: str) -> bool:
    """Return True if *path_str* contains ``..`` traversal components.

    Quick check for obvious traversal attempts before doing full resolution.
    """
    parts = Path(path_str).parts
    return ".." in parts


_SENSITIVE_PREFIXES = (
    "/etc",
    "/var",
    "/sys",
    "/proc",
    "/dev",
    "/private/etc",
)


def _default_allowed_media_dirs() -> List[Path]:
    hermes = get_hermes_home()
    return [
        hermes / "cache",
        hermes / "data",
        hermes / "media",
        Path(tempfile.gettempdir()),
    ]


def validate_media_path(
    local_path: Path,
    *,
    extra_allowed: Optional[List[Path]] = None,
) -> None:
    """Raise ``PermissionError`` if *local_path* is outside allowed media directories."""
    resolved = local_path.resolve()

    if has_traversal_component(str(local_path)):
        raise PermissionError(
            f"Path traversal detected in media path: {local_path}"
        )

    resolved_str = str(resolved)
    for prefix in _SENSITIVE_PREFIXES:
        if resolved_str == prefix or resolved_str.startswith(prefix + os.sep):
            raise PermissionError(
                f"Media path points to sensitive system directory: {resolved}"
            )

    allowed = _default_allowed_media_dirs()
    if extra_allowed:
        allowed.extend(extra_allowed)

    for allowed_dir in allowed:
        if validate_within_dir(resolved, allowed_dir) is None:
            return

    raise PermissionError(
        f"Media path '{local_path}' is outside allowed directories. "
        "Only files in hermes cache/data/media or tmp directories can be sent as media."
    )
