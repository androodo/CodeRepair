from __future__ import annotations

import hashlib
import os
import subprocess
from collections.abc import Iterable
from pathlib import Path

import pytest

from coderepair.workspace import (
    DirtyRepositoryError,
    InvalidHeadError,
    NotGitRepositoryError,
    UnsafeRepositoryError,
    Workspace,
    WorkspaceClosedError,
    create_workspace,
)


def _git(repository: Path, *arguments: str) -> str:
    environment = dict(os.environ)
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    environment["GIT_CONFIG_GLOBAL"] = os.devnull
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
    )
    if completed.returncode != 0:
        pytest.fail(
            f"Git setup failed ({' '.join(arguments)}): "
            f"{completed.stderr or completed.stdout}"
        )
    return completed.stdout.strip()


def _repository(tmp_path: Path, files: dict[str, str] | None = None) -> Path:
    repository = tmp_path / "source"
    repository.mkdir()
    _git(repository, "init", "--quiet", "--initial-branch=main")
    _git(repository, "config", "user.name", "Workspace Test")
    _git(repository, "config", "user.email", "workspace@example.invalid")
    for relative, content in (files or {"app.py": "VALUE = 1\n"}).items():
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _git(repository, "add", "--all")
    _git(repository, "commit", "--quiet", "--message", "initial")
    return repository


def _git_metadata_snapshot(repository: Path) -> dict[str, str]:
    git_entry = repository / ".git"
    snapshot: dict[str, str] = {}
    for path in sorted(git_entry.rglob("*")):
        if path.is_file():
            relative = path.relative_to(git_entry).as_posix()
            snapshot[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


def _create_tracked_symlink(
    repository: Path, link_name: str, target: str
) -> None:
    link = repository / link_name
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")
    _git(repository, "add", "--", link_name)
    _git(repository, "commit", "--quiet", "--message", "add symlink")


def test_workspace_is_exact_isolated_snapshot_and_source_git_is_unchanged(
    tmp_path: Path,
) -> None:
    repository = _repository(
        tmp_path,
        {
            "app.py": "VALUE = 1\n",
            "pkg/module.py": "def answer():\n    return 42\n",
            ".env.example": "TOKEN=\n",
        },
    )
    commit = _git(repository, "rev-parse", "HEAD")
    metadata_before = _git_metadata_snapshot(repository)

    workspace = create_workspace(repository)
    try:
        assert workspace.root != repository
        assert workspace.source_root == repository.resolve()
        assert workspace.source_repo == repository.resolve()
        assert workspace.original_commit == commit
        assert (workspace.root / "app.py").read_text(encoding="utf-8") == "VALUE = 1\n"
        assert not (workspace.root / ".git").exists()
        assert workspace.diff() == ""
        assert workspace.modified_files() == ()
        assert _git_metadata_snapshot(repository) == metadata_before
    finally:
        workspace.cleanup()


def test_workspace_changes_never_modify_source_and_diff_includes_all_change_kinds(
    tmp_path: Path,
) -> None:
    repository = _repository(
        tmp_path,
        {"change.py": "OLD = True\n", "delete.py": "DELETE = True\n"},
    )

    with Workspace.create(repository) as workspace:
        (workspace.root / "change.py").write_text("NEW = True\n", encoding="utf-8")
        (workspace.root / "new.py").write_text("CREATED = True\n", encoding="utf-8")
        (workspace.root / "delete.py").unlink()

        assert workspace.modified_files() == ("change.py", "delete.py", "new.py")
        patch = workspace.diff()
        assert "diff --git a/change.py b/change.py" in patch
        assert "diff --git a/delete.py b/delete.py" in patch
        assert "diff --git a/new.py b/new.py" in patch
        assert "+NEW = True" in patch
        assert "+CREATED = True" in patch

        assert (repository / "change.py").read_text(encoding="utf-8") == "OLD = True\n"
        assert (repository / "delete.py").is_file()
        assert not (repository / "new.py").exists()


def test_context_manager_removes_temporary_workspace(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    with create_workspace(repository, temp_parent=tmp_path) as workspace:
        root = workspace.root
        assert root.is_dir()

    assert workspace.closed
    assert not root.exists()
    with pytest.raises(WorkspaceClosedError):
        workspace.diff()
    workspace.cleanup()


@pytest.mark.parametrize("untracked", [False, True])
def test_dirty_source_repository_is_rejected(
    tmp_path: Path, untracked: bool
) -> None:
    repository = _repository(tmp_path)
    target = repository / ("untracked.py" if untracked else "app.py")
    target.write_text("CHANGED = True\n", encoding="utf-8")

    with pytest.raises(DirtyRepositoryError) as error:
        create_workspace(repository)

    assert error.value.changes
    assert not tuple(tmp_path.glob("coderepair-*"))


def test_repository_without_commit_is_rejected(tmp_path: Path) -> None:
    repository = tmp_path / "empty-repository"
    repository.mkdir()
    _git(repository, "init", "--quiet", "--initial-branch=main")

    with pytest.raises(InvalidHeadError):
        create_workspace(repository)


def test_non_repository_and_nested_repository_path_are_rejected(
    tmp_path: Path,
) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(NotGitRepositoryError):
        create_workspace(plain)

    repository = _repository(tmp_path)
    nested = repository / "nested"
    nested.mkdir()
    with pytest.raises(NotGitRepositoryError):
        create_workspace(nested)


def test_tracked_symlink_is_rejected(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _create_tracked_symlink(repository, "linked.py", "app.py")

    with pytest.raises(UnsafeRepositoryError, match="symbolic link"):
        create_workspace(repository)


def test_gitlink_submodule_entry_is_rejected(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    dependency = tmp_path / "dependency"
    dependency.mkdir()
    _git(dependency, "init", "--quiet", "--initial-branch=main")
    _git(dependency, "config", "user.name", "Workspace Test")
    _git(dependency, "config", "user.email", "workspace@example.invalid")
    (dependency / "library.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(dependency, "add", "library.py")
    _git(dependency, "commit", "--quiet", "--message", "dependency")
    _git(
        repository,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "--quiet",
        str(dependency),
        "vendor/dependency",
    )
    _git(repository, "commit", "--quiet", "--message", "add gitlink")

    with pytest.raises(UnsafeRepositoryError, match="submodule"):
        create_workspace(repository)


@pytest.mark.parametrize(
    "sensitive_name",
    [".env", "credentials.json", "private.pem", ".ssh/id_rsa"],
)
def test_tracked_sensitive_path_is_rejected(
    tmp_path: Path, sensitive_name: str
) -> None:
    repository = _repository(tmp_path)
    path = repository / sensitive_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not-a-real-secret\n", encoding="utf-8")
    _git(repository, "add", "--", sensitive_name)
    _git(repository, "commit", "--quiet", "--message", "add unsafe path")

    with pytest.raises(UnsafeRepositoryError, match="unsafe tracked"):
        create_workspace(repository)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("filter.untrusted.clean", "python attacker.py"),
        ("include.path", "../outside.gitconfig"),
        ("core.fsmonitor", "watchman"),
        ("diff.untrusted.textconv", "python attacker.py"),
    ],
)
def test_executable_or_external_local_git_config_is_rejected(
    tmp_path: Path, key: str, value: str
) -> None:
    repository = _repository(tmp_path)
    _git(repository, "config", "--local", key, value)

    with pytest.raises(UnsafeRepositoryError, match="Git configuration"):
        create_workspace(repository)


@pytest.mark.parametrize("kind", ["symlink", "sensitive"])
def test_diff_refuses_unsafe_new_workspace_entries(
    tmp_path: Path, kind: str
) -> None:
    repository = _repository(tmp_path)
    with create_workspace(repository) as workspace:
        if kind == "sensitive":
            (workspace.root / ".env").write_text("TOKEN=value\n", encoding="utf-8")
        else:
            link = workspace.root / "escape"
            try:
                os.symlink(repository / "app.py", link)
            except (OSError, NotImplementedError) as exc:
                pytest.skip(f"symbolic links are unavailable: {exc}")

        with pytest.raises(UnsafeRepositoryError):
            workspace.diff()


def test_baseline_uses_latest_pinned_head(tmp_path: Path) -> None:
    repository = _repository(tmp_path, {"app.py": "VERSION = 1\n"})
    (repository / "app.py").write_text("VERSION = 2\n", encoding="utf-8")
    _git(repository, "add", "app.py")
    _git(repository, "commit", "--quiet", "--message", "second")
    expected_commit = _git(repository, "rev-parse", "HEAD")

    with create_workspace(repository) as workspace:
        assert workspace.original_commit == expected_commit
        assert (workspace.root / "app.py").read_text(encoding="utf-8") == "VERSION = 2\n"


def test_cleanup_occurs_if_repository_is_unsafe(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    secret = repository / ".env"
    secret.write_text("TOKEN=value\n", encoding="utf-8")
    _git(repository, "add", ".env")
    _git(repository, "commit", "--quiet", "--message", "unsafe")

    before: Iterable[Path] = tuple(tmp_path.glob("coderepair-*"))
    with pytest.raises(UnsafeRepositoryError):
        create_workspace(repository, temp_parent=tmp_path)
    assert tuple(tmp_path.glob("coderepair-*")) == tuple(before)
