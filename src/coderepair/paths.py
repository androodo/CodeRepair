"""Path validation for untrusted repository and model-provided paths.

Repository paths are deliberately limited to a portable, relative subset. In
particular, callers cannot use absolute paths, parent components, symlinks, or
platform-specific aliases to escape a workspace.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import TypeAlias

PathInput: TypeAlias = str | os.PathLike[str]


class PathSecurityError(ValueError):
    """Base class for a path that is unsafe for repository access."""


class InvalidRelativePathError(PathSecurityError):
    """Raised when a repository path is not a safe portable relative path."""


class PathTraversalError(PathSecurityError):
    """Raised when a path could resolve outside the workspace."""


class SymlinkPathError(PathSecurityError):
    """Raised when any requested path component is a symbolic link."""


class SensitivePathError(PathSecurityError):
    """Raised when a path looks like a credential or secret file."""


class WorkspacePathNotFoundError(PathSecurityError):
    """Raised when a requested workspace path does not exist."""


class DirectoryNotAllowedError(PathSecurityError):
    """Raised when a caller requested a file but selected a directory."""


_WINDOWS_RESERVED_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}

_SENSITIVE_DIRECTORIES = {
    ".aws",
    ".azure",
    ".config/gcloud",
    ".gnupg",
    ".kube",
    ".ssh",
}

_SENSITIVE_FILENAMES = {
    ".git-credentials",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "_netrc",
    "account-key.json",
    "auth.json",
    "credentials",
    "credentials.json",
    "docker-config.json",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
    "service-account.json",
    "service_account.json",
    "token.json",
}

_SENSITIVE_SUFFIXES = (
    ".jks",
    ".kdbx",
    ".key",
    ".keystore",
    ".p12",
    ".pem",
    ".pfx",
    ".tfstate",
    ".tfstate.backup",
)

_SAFE_ENV_EXAMPLES = {
    ".env.dist",
    ".env.example",
    ".env.sample",
    ".env.template",
}


def _is_link_like(status: os.stat_result) -> bool:
    attributes = getattr(status, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(status.st_mode) or bool(attributes & reparse_flag)


def _raw_path(path: PathInput) -> str:
    try:
        raw = os.fspath(path)
    except TypeError as exc:
        raise InvalidRelativePathError("repository path must be a string or path") from exc
    if not isinstance(raw, str):
        raise InvalidRelativePathError("byte paths are not supported")
    return raw


def normalize_relative_path(
    raw: PathInput,
    *,
    reject_sensitive: bool = True,
) -> PurePosixPath:
    """Return a validated, portable repository-relative path.

    Both slash styles are interpreted as separators so a traversal written for
    one operating system is also rejected on the other. The result always uses
    POSIX separators, matching Git's path representation.
    """

    value = _raw_path(raw)
    if not value:
        raise InvalidRelativePathError("repository path cannot be empty")
    if "\x00" in value:
        raise InvalidRelativePathError("repository path contains a NUL byte")

    windows_path = PureWindowsPath(value)
    posix_path = PurePosixPath(value)
    if windows_path.drive or windows_path.is_absolute() or posix_path.is_absolute():
        raise PathTraversalError("absolute repository paths are not allowed")

    portable_value = value.replace("\\", "/")
    parts = portable_value.split("/")
    if any(part in {"", "."} for part in parts):
        raise InvalidRelativePathError("empty and '.' path components are not allowed")
    if any(part == ".." for part in parts):
        raise PathTraversalError("parent path components are not allowed")
    # Strings come from model/tool input and must use Git's portable slash
    # spelling. Path objects are also used internally and stringify with
    # backslashes on Windows, so accept those after validating their parts.
    if isinstance(raw, str) and "\\" in value:
        raise InvalidRelativePathError("repository paths must use POSIX '/' separators")

    for part in parts:
        if any(ord(character) < 32 or ord(character) == 127 for character in part):
            raise InvalidRelativePathError("control characters are not allowed in paths")
        if ":" in part:
            raise InvalidRelativePathError("':' is not allowed in portable paths")
        if part.rstrip(" .") != part:
            raise InvalidRelativePathError(
                "path components cannot end with a space or period"
            )
        folded = part.casefold()
        if folded == ".git":
            raise PathTraversalError("access to Git metadata is not allowed")
        device_name = folded.split(".", 1)[0]
        if device_name in _WINDOWS_RESERVED_NAMES:
            raise InvalidRelativePathError(
                f"reserved platform path component is not allowed: {part!r}"
            )

    normalized = PurePosixPath(*parts)
    if reject_sensitive and is_sensitive_path(normalized):
        raise SensitivePathError(f"sensitive repository path is not allowed: {normalized}")
    return normalized


def is_sensitive_path(relative: PathInput | PurePath) -> bool:
    """Return whether a repository-relative path is likely to contain secrets.

    This intentionally uses conservative filename rules rather than inspecting
    contents. Example/template .env files are allowed because they are a common
    part of documented project configuration.
    """

    try:
        raw = os.fspath(relative)
    except TypeError:
        return True
    if not isinstance(raw, str) or not raw:
        return True

    folded_parts = tuple(
        part.casefold() for part in raw.replace("\\", "/").split("/") if part
    )
    if not folded_parts:
        return True

    joined = "/".join(folded_parts)
    filename = folded_parts[-1]

    if any(part == ".git" for part in folded_parts):
        return True
    if any(
        joined == directory or joined.startswith(f"{directory}/")
        for directory in _SENSITIVE_DIRECTORIES
    ):
        return True
    if "/.config/gcloud/" in f"/{joined}/":
        return True
    if any(
        part in {".ssh", ".aws", ".azure", ".gnupg", ".kube"}
        for part in folded_parts
    ):
        return True
    if filename.startswith(".env") and filename not in _SAFE_ENV_EXAMPLES:
        return True
    if filename in _SENSITIVE_FILENAMES:
        return True
    if filename.endswith(_SENSITIVE_SUFFIXES):
        return True
    if filename in {
        "secret.json",
        "secret.toml",
        "secret.yaml",
        "secret.yml",
        "secrets.json",
        "secrets.toml",
        "secrets.yaml",
        "secrets.yml",
    }:
        return True
    return (
        filename.startswith("service-account-")
        or filename.startswith("service_account_")
    ) and filename.endswith(".json")


def validate_relative_path(
    root: PathInput,
    raw: PathInput,
    *,
    must_exist: bool,
    allow_directory: bool = False,
    reject_sensitive: bool = True,
) -> Path:
    """Resolve a safe path below root without following symbolic links.

    With must_exist=False a missing final path is permitted for creation, while
    every existing parent is still validated. Existing directories are only
    returned when allow_directory is true.
    """

    root_path = Path(root)
    try:
        root_status = root_path.stat(follow_symlinks=False)
    except (FileNotFoundError, OSError) as exc:
        raise WorkspacePathNotFoundError(
            f"workspace root does not exist: {root_path}"
        ) from exc
    if _is_link_like(root_status):
        raise SymlinkPathError("workspace root cannot be a symbolic link or junction")
    try:
        root_resolved = root_path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise WorkspacePathNotFoundError(
            f"workspace root does not exist: {root_path}"
        ) from exc
    if not root_resolved.is_dir():
        raise InvalidRelativePathError(f"workspace root is not a directory: {root_path}")

    relative = normalize_relative_path(raw, reject_sensitive=reject_sensitive)
    candidate = root_resolved.joinpath(*relative.parts)

    current = root_resolved
    for index, part in enumerate(relative.parts):
        current = current / part
        try:
            status = current.stat(follow_symlinks=False)
        except FileNotFoundError:
            break
        except OSError as exc:
            raise PathSecurityError(f"cannot inspect repository path: {relative}") from exc
        if _is_link_like(status):
            raise SymlinkPathError(f"links and junctions are not allowed: {relative}")
        if index < len(relative.parts) - 1 and not stat.S_ISDIR(status.st_mode):
            raise InvalidRelativePathError(
                f"non-directory component in repository path: {relative}"
            )

    try:
        resolved_candidate = candidate.resolve(strict=False)
        resolved_candidate.relative_to(root_resolved)
    except (OSError, RuntimeError, ValueError) as exc:
        raise PathTraversalError(f"path escapes the workspace: {relative}") from exc

    exists = candidate.exists()
    if must_exist and not exists:
        raise WorkspacePathNotFoundError(f"repository path does not exist: {relative}")
    if exists:
        try:
            status = candidate.stat(follow_symlinks=False)
        except OSError as exc:
            raise PathSecurityError(f"cannot inspect repository path: {relative}") from exc
        if _is_link_like(status):
            raise SymlinkPathError(f"links and junctions are not allowed: {relative}")
        if stat.S_ISDIR(status.st_mode):
            if not allow_directory:
                raise DirectoryNotAllowedError(
                    f"repository path is a directory, not a file: {relative}"
                )
        elif not stat.S_ISREG(status.st_mode):
            raise InvalidRelativePathError(
                f"special filesystem entries are not allowed: {relative}"
            )

    return resolved_candidate


__all__ = [
    "DirectoryNotAllowedError",
    "InvalidRelativePathError",
    "PathSecurityError",
    "PathTraversalError",
    "SensitivePathError",
    "SymlinkPathError",
    "WorkspacePathNotFoundError",
    "is_sensitive_path",
    "normalize_relative_path",
    "validate_relative_path",
]
