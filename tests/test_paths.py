from __future__ import annotations

import os
from pathlib import Path

import pytest

from coderepair.paths import (
    DirectoryNotAllowedError,
    InvalidRelativePathError,
    PathTraversalError,
    SensitivePathError,
    SymlinkPathError,
    WorkspacePathNotFoundError,
    is_sensitive_path,
    normalize_relative_path,
    validate_relative_path,
)


def test_validate_existing_file_and_missing_creation_path(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    source = package / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")

    assert validate_relative_path(
        tmp_path,
        "package/module.py",
        must_exist=True,
    ) == source.resolve()
    assert validate_relative_path(
        tmp_path,
        Path("package") / "new.py",
        must_exist=False,
    ) == (package / "new.py").resolve()


def test_missing_path_and_directory_policy(tmp_path: Path) -> None:
    directory = tmp_path / "src"
    directory.mkdir()

    with pytest.raises(WorkspacePathNotFoundError):
        validate_relative_path(tmp_path, "missing.py", must_exist=True)
    with pytest.raises(DirectoryNotAllowedError):
        validate_relative_path(tmp_path, "src", must_exist=True)
    assert validate_relative_path(
        tmp_path,
        "src",
        must_exist=True,
        allow_directory=True,
    ) == directory.resolve()


@pytest.mark.parametrize(
    "raw",
    [
        "../outside.py",
        r"..\outside.py",
        "/absolute.py",
        r"C:\absolute.py",
        r"\\server\share\file.py",
        "src/../../outside.py",
        ".git/config",
        "src/.GIT/index",
    ],
)
def test_traversal_and_git_metadata_paths_are_rejected(
    tmp_path: Path, raw: str
) -> None:
    with pytest.raises(PathTraversalError):
        validate_relative_path(tmp_path, raw, must_exist=False)


@pytest.mark.parametrize(
    "raw",
    [
        "",
        ".",
        "src//module.py",
        "src/./module.py",
        "src/module.py ",
        r"src\module.py",
        "src/name:stream",
        "con.txt",
        "src/LPT1",
        "line\nbreak.py",
    ],
)
def test_ambiguous_nonportable_paths_are_rejected(tmp_path: Path, raw: str) -> None:
    with pytest.raises(InvalidRelativePathError):
        validate_relative_path(tmp_path, raw, must_exist=False)


@pytest.mark.parametrize(
    "raw",
    [
        ".env",
        ".env.local",
        ".npmrc",
        "credentials.json",
        "config/secrets.yml",
        "certificates/private.pem",
        ".ssh/id_rsa.pub",
        ".aws/credentials",
        "project/.config/gcloud/application_default_credentials.json",
        "terraform.tfstate",
    ],
)
def test_sensitive_paths_are_recognized_and_rejected(
    tmp_path: Path, raw: str
) -> None:
    assert is_sensitive_path(raw)
    with pytest.raises(SensitivePathError):
        validate_relative_path(tmp_path, raw, must_exist=False)


@pytest.mark.parametrize(
    "raw",
    [
        ".env.example",
        ".env.sample",
        "src/secrets.py",
        "docs/public-key.txt",
        "config/settings.toml",
    ],
)
def test_non_secret_project_paths_are_allowed(tmp_path: Path, raw: str) -> None:
    assert not is_sensitive_path(raw)
    assert normalize_relative_path(raw).as_posix() == raw


def test_sensitive_check_can_be_explicitly_disabled(tmp_path: Path) -> None:
    expected = tmp_path / ".env"
    assert validate_relative_path(
        tmp_path,
        ".env",
        must_exist=False,
        reject_sensitive=False,
    ) == expected.resolve()


def test_symlink_component_is_rejected_even_when_target_stays_inside(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "module.py").write_text("pass\n", encoding="utf-8")
    link = tmp_path / "link"
    try:
        os.symlink(real, link, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")

    with pytest.raises(SymlinkPathError):
        validate_relative_path(tmp_path, "link/module.py", must_exist=True)


def test_broken_symlink_is_rejected_for_creation_path(tmp_path: Path) -> None:
    link = tmp_path / "broken"
    try:
        os.symlink(tmp_path / "missing-target", link)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")

    with pytest.raises(SymlinkPathError):
        validate_relative_path(tmp_path, "broken", must_exist=False)
