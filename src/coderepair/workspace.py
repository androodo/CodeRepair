"""Immutable-source temporary Git workspaces.

The source repository is only queried with read-only Git operations. A pinned
HEAD archive is validated and extracted into a private temporary directory,
where a fresh Git repository provides the baseline used for later diffs.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from .paths import (
    PathSecurityError,
    SymlinkPathError,
    normalize_relative_path,
    validate_relative_path,
)

_GIT_TIMEOUT_SECONDS = 120.0
_MAX_ARCHIVE_FILES = 100_000
_MAX_ARCHIVE_BYTES = 1_073_741_824


class WorkspaceError(RuntimeError):
    """Base class for temporary-workspace failures."""


class GitNotAvailableError(WorkspaceError):
    """Raised when the Git executable is unavailable."""


class GitCommandError(WorkspaceError):
    """Raised when an internal Git command fails."""

    def __init__(
        self,
        command: Sequence[str],
        returncode: int,
        stderr: str,
    ) -> None:
        self.command = tuple(command)
        self.returncode = returncode
        self.stderr = stderr.strip()
        detail = self.stderr or "no error output"
        super().__init__(f"Git command failed with exit code {returncode}: {detail}")


class NotGitRepositoryError(WorkspaceError):
    """Raised when the selected path is not a Git working-tree root."""


class InvalidHeadError(WorkspaceError):
    """Raised when the source repository does not have a valid HEAD commit."""


class DirtyRepositoryError(WorkspaceError):
    """Raised when tracked or untracked source changes are present."""

    def __init__(self, changes: Sequence[str]) -> None:
        self.changes = tuple(changes)
        preview = ", ".join(self.changes[:10])
        if len(self.changes) > 10:
            preview = f"{preview}, and {len(self.changes) - 10} more"
        super().__init__(
            "source repository has uncommitted changes"
            + (f": {preview}" if preview else "")
        )


class UnsafeRepositoryError(WorkspaceError):
    """Raised when a repository contains an unsupported unsafe entry."""


class RepositoryChangedError(WorkspaceError):
    """Raised when HEAD or worktree state changes while a snapshot is made."""


class WorkspaceClosedError(WorkspaceError):
    """Raised when an operation uses an already-cleaned workspace."""


def _git_executable() -> str:
    executable = shutil.which("git")
    if executable is None:
        raise GitNotAvailableError("Git is required to create a repair workspace")
    return executable


def _git_environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    environment = dict(os.environ)
    for key in tuple(environment):
        if key.upper().startswith("GIT_"):
            del environment[key]
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    if extra:
        environment.update(extra)
    return environment


def _git_command(
    repository: Path,
    arguments: Sequence[str],
    *,
    git_directory: Path | None = None,
) -> list[str]:
    metadata = git_directory if git_directory is not None else repository / ".git"
    disabled_hooks = metadata / "coderepair-disabled-hooks"
    command = [
        _git_executable(),
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        "-c",
        f"core.hooksPath={disabled_hooks}",
        "-c",
        f"core.worktree={repository}",
        "--no-pager",
    ]
    if git_directory is not None:
        command.extend(
            [
                f"--git-dir={git_directory}",
                f"--work-tree={repository}",
            ]
        )
    command.extend(["-C", str(repository), *arguments])
    return command


def _run_git(
    repository: Path,
    arguments: Sequence[str],
    *,
    text: bool = True,
    extra_environment: Mapping[str, str] | None = None,
    git_directory: Path | None = None,
    timeout: float = _GIT_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    command = _git_command(
        repository,
        arguments,
        git_directory=git_directory,
    )
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=text,
            env=_git_environment(extra_environment),
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise GitCommandError(command, -1, f"timed out after {timeout:g} seconds") from exc
    except OSError as exc:
        raise GitCommandError(command, -1, str(exc)) from exc
    if completed.returncode != 0:
        error = completed.stderr
        error_text = (
            error.decode("utf-8", errors="replace")
            if isinstance(error, bytes)
            else error
        )
        raise GitCommandError(command, completed.returncode, error_text)
    return completed


def _same_path(first: Path, second: Path) -> bool:
    return os.path.normcase(str(first)) == os.path.normcase(str(second))


def _repository_root(repository: str | os.PathLike[str]) -> Path:
    supplied = Path(repository)
    if supplied.is_symlink():
        raise UnsafeRepositoryError("the source repository path cannot be a symlink")
    try:
        resolved = supplied.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise NotGitRepositoryError(f"repository does not exist: {supplied}") from exc
    if not resolved.is_dir():
        raise NotGitRepositoryError(f"repository is not a directory: {supplied}")

    git_entry = resolved / ".git"
    if not git_entry.exists():
        raise NotGitRepositoryError(
            f"repository path must be the working-tree root: {resolved}"
        )
    if git_entry.is_symlink():
        raise UnsafeRepositoryError("a symlinked .git entry is not supported")

    try:
        top_level_result = _run_git(resolved, ["rev-parse", "--show-toplevel"])
        bare_result = _run_git(resolved, ["rev-parse", "--is-bare-repository"])
    except GitCommandError as exc:
        raise NotGitRepositoryError(f"not a usable Git working tree: {resolved}") from exc

    top_level = Path(str(top_level_result.stdout).strip()).resolve()
    if not _same_path(top_level, resolved):
        raise NotGitRepositoryError(
            "repository path must identify the Git working-tree root"
        )
    if str(bare_result.stdout).strip().casefold() != "false":
        raise NotGitRepositoryError("bare Git repositories are not supported")
    return resolved


def _head_commit(repository: Path) -> str:
    try:
        result = _run_git(repository, ["rev-parse", "--verify", "HEAD^{commit}"])
    except GitCommandError as exc:
        raise InvalidHeadError("repository HEAD does not identify a commit") from exc
    commit = str(result.stdout).strip()
    if not commit or any(character not in "0123456789abcdefABCDEF" for character in commit):
        raise InvalidHeadError("Git returned an invalid HEAD object identifier")
    return commit.lower()


def _assert_safe_local_config(repository: Path) -> None:
    """Reject repository-local configuration that can invoke host commands."""

    result = _run_git(
        repository,
        [
            "config",
            "--local",
            "--no-includes",
            "--name-only",
            "--list",
            "-z",
        ],
        text=False,
    )
    keys = {
        raw_key.decode("utf-8", errors="replace").casefold()
        for raw_key in bytes(result.stdout).split(b"\0")
        if raw_key
    }
    unsafe_keys = sorted(
        key
        for key in keys
        if key.startswith(("include.", "includeif.", "filter."))
        or (
            key.startswith(("credential.", "diff.", "merge."))
            and key.rsplit(".", 1)[-1]
            in {
                "clean",
                "command",
                "driver",
                "helper",
                "process",
                "program",
                "smudge",
                "textconv",
            }
        )
        or key
        in {
            "core.attributesfile",
            "core.excludesfile",
            "core.fsmonitor",
            "core.sshcommand",
        }
    )
    if unsafe_keys:
        rendered = ", ".join(unsafe_keys[:10])
        raise UnsafeRepositoryError(
            f"repository has executable or external-path Git configuration: {rendered}"
        )


def _status_entries(repository: Path) -> tuple[str, ...]:
    result = _run_git(
        repository,
        [
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ],
        text=False,
    )
    output = bytes(result.stdout)
    return tuple(
        entry.decode("utf-8", errors="replace")
        for entry in output.split(b"\0")
        if entry
    )


def _assert_clean(repository: Path) -> None:
    changes = _status_entries(repository)
    if changes:
        raise DirtyRepositoryError(changes)


def _tree_entries(repository: Path, commit: str) -> Iterator[tuple[str, str, str]]:
    result = _run_git(
        repository,
        ["ls-tree", "-rz", "--full-tree", commit],
        text=False,
    )
    for raw_entry in bytes(result.stdout).split(b"\0"):
        if not raw_entry:
            continue
        try:
            header, raw_path = raw_entry.split(b"\t", 1)
            raw_mode, raw_type, _raw_object = header.split(b" ", 2)
        except ValueError as exc:
            raise UnsafeRepositoryError("Git returned a malformed tree entry") from exc
        yield (
            raw_mode.decode("ascii", errors="strict"),
            raw_type.decode("ascii", errors="strict"),
            raw_path.decode("utf-8", errors="surrogateescape"),
        )


def _assert_safe_commit_tree(repository: Path, commit: str) -> None:
    for count, (mode, object_type, relative) in enumerate(
        _tree_entries(repository, commit), start=1
    ):
        if count > _MAX_ARCHIVE_FILES:
            raise UnsafeRepositoryError(
                f"repository exceeds the {_MAX_ARCHIVE_FILES:,}-file workspace limit"
            )
        if mode == "120000":
            raise UnsafeRepositoryError(
                f"tracked symbolic links are not supported: {relative}"
            )
        if mode == "160000" or object_type == "commit":
            raise UnsafeRepositoryError(f"Git submodules are not supported: {relative}")
        if object_type != "blob":
            raise UnsafeRepositoryError(
                f"unsupported Git tree entry type {object_type!r}: {relative}"
            )
        try:
            normalize_relative_path(relative)
        except (PathSecurityError, UnicodeError) as exc:
            raise UnsafeRepositoryError(
                f"unsafe tracked repository path: {relative!r}"
            ) from exc


def _create_archive(repository: Path, commit: str, archive_path: Path) -> None:
    command = _git_command(repository, ["archive", "--format=tar", commit])
    try:
        with archive_path.open("xb") as output:
            completed = subprocess.run(
                command,
                check=False,
                stdout=output,
                stderr=subprocess.PIPE,
                env=_git_environment(),
                timeout=_GIT_TIMEOUT_SECONDS,
            )
    except subprocess.TimeoutExpired as exc:
        raise GitCommandError(
            command, -1, f"timed out after {_GIT_TIMEOUT_SECONDS:g} seconds"
        ) from exc
    except OSError as exc:
        raise WorkspaceError(f"could not create repository archive: {exc}") from exc
    if completed.returncode != 0:
        error = completed.stderr.decode("utf-8", errors="replace")
        raise GitCommandError(command, completed.returncode, error)


def _open_new_file(path: Path) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(path, flags, 0o600)


def _extract_archive(archive_path: Path, destination: Path) -> None:
    file_count = 0
    total_bytes = 0
    try:
        archive = tarfile.open(archive_path, mode="r:")  # noqa: SIM115
    except (tarfile.TarError, OSError) as exc:
        raise UnsafeRepositoryError("Git produced an unreadable repository archive") from exc

    with archive:
        for member in archive:
            file_count += 1
            if file_count > _MAX_ARCHIVE_FILES:
                raise UnsafeRepositoryError(
                    f"archive exceeds the {_MAX_ARCHIVE_FILES:,}-entry workspace limit"
                )
            if member.size < 0:
                raise UnsafeRepositoryError("archive contains an invalid entry size")
            total_bytes += member.size
            if total_bytes > _MAX_ARCHIVE_BYTES:
                raise UnsafeRepositoryError(
                    "archive exceeds the 1 GiB uncompressed workspace limit"
                )

            member_name = member.name
            if member.isdir() and member_name.endswith("/"):
                member_name = member_name[:-1]
            try:
                relative = normalize_relative_path(member_name)
                target = validate_relative_path(
                    destination,
                    relative,
                    must_exist=False,
                    allow_directory=True,
                )
            except PathSecurityError as exc:
                raise UnsafeRepositoryError(
                    f"archive contains an unsafe path: {member.name!r}"
                ) from exc

            if member.isdir():
                if target.exists() and not target.is_dir():
                    raise UnsafeRepositoryError(
                        f"archive path has conflicting entry types: {relative}"
                    )
                target.mkdir(parents=True, exist_ok=True)
                target.chmod(0o755)
                continue
            if not member.isreg():
                raise UnsafeRepositoryError(
                    f"archive contains a link or special entry: {relative}"
                )
            if target.exists():
                raise UnsafeRepositoryError(
                    f"archive contains a duplicate entry: {relative}"
                )

            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise UnsafeRepositoryError(
                    f"archive file content is unavailable: {relative}"
                )
            descriptor: int | None = None
            try:
                descriptor = _open_new_file(target)
                with os.fdopen(descriptor, "wb") as output:
                    descriptor = None
                    shutil.copyfileobj(source, output, length=1024 * 1024)
            finally:
                source.close()
                if descriptor is not None:
                    os.close(descriptor)
            target.chmod(0o755 if member.mode & 0o111 else 0o644)


def _initialize_baseline(repository: Path, git_directory: Path) -> str:
    initialization_command = [
        _git_executable(),
        "-c",
        "init.defaultBranch=workspace",
        "init",
        "--quiet",
        "--bare",
        str(git_directory),
    ]
    try:
        initialized = subprocess.run(
            initialization_command,
            check=False,
            capture_output=True,
            text=True,
            env=_git_environment(),
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WorkspaceError(f"could not initialize workspace baseline: {exc}") from exc
    if initialized.returncode != 0:
        raise GitCommandError(
            initialization_command,
            initialized.returncode,
            initialized.stderr,
        )

    _run_git(
        repository,
        ["config", "--local", "core.autocrlf", "false"],
        git_directory=git_directory,
    )
    _run_git(
        repository,
        ["config", "--local", "commit.gpgsign", "false"],
        git_directory=git_directory,
    )
    _run_git(
        repository,
        ["config", "--local", "core.quotePath", "false"],
        git_directory=git_directory,
    )
    _run_git(
        repository,
        ["add", "--all", "--force", "--", "."],
        git_directory=git_directory,
    )
    tree_result = _run_git(
        repository,
        ["write-tree"],
        git_directory=git_directory,
    )
    tree = str(tree_result.stdout).strip()
    identity = {
        "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
        "GIT_AUTHOR_EMAIL": "workspace@coderepair.invalid",
        "GIT_AUTHOR_NAME": "CodeRepair",
        "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
        "GIT_COMMITTER_EMAIL": "workspace@coderepair.invalid",
        "GIT_COMMITTER_NAME": "CodeRepair",
    }
    commit_result = _run_git(
        repository,
        ["commit-tree", tree, "-m", "CodeRepair workspace baseline"],
        extra_environment=identity,
        git_directory=git_directory,
    )
    baseline_commit = str(commit_result.stdout).strip()
    _run_git(
        repository,
        ["update-ref", "refs/heads/workspace", baseline_commit],
        git_directory=git_directory,
    )
    _run_git(
        repository,
        ["symbolic-ref", "HEAD", "refs/heads/workspace"],
        git_directory=git_directory,
    )
    return baseline_commit


def _is_reparse_point(mode_result: os.stat_result) -> bool:
    attributes = getattr(mode_result, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attributes & reparse_flag)


def _validate_workspace_tree(root: Path) -> None:
    pending = [root]
    entry_count = 0
    total_bytes = 0
    while pending:
        directory = pending.pop()
        try:
            entries = tuple(directory.iterdir())
        except OSError as exc:
            raise UnsafeRepositoryError(
                f"cannot inspect temporary workspace directory: {directory}"
            ) from exc
        for entry in entries:
            entry_count += 1
            if entry_count > _MAX_ARCHIVE_FILES:
                raise UnsafeRepositoryError(
                    f"workspace exceeds the {_MAX_ARCHIVE_FILES:,}-entry limit"
                )
            relative = entry.relative_to(root)
            try:
                normalized = normalize_relative_path(relative)
                mode_result = entry.stat(follow_symlinks=False)
            except (OSError, PathSecurityError) as exc:
                raise UnsafeRepositoryError(
                    f"unsafe temporary workspace entry: {relative}"
                ) from exc
            mode = mode_result.st_mode
            if stat.S_ISLNK(mode) or _is_reparse_point(mode_result):
                raise UnsafeRepositoryError(
                    f"links are not allowed in the temporary workspace: {normalized}"
                )
            if mode_result.st_nlink > 1 and stat.S_ISREG(mode):
                raise UnsafeRepositoryError(
                    f"hard-linked files are not allowed: {normalized}"
                )
            if stat.S_ISDIR(mode):
                try:
                    validate_relative_path(
                        root,
                        normalized,
                        must_exist=True,
                        allow_directory=True,
                    )
                except (PathSecurityError, SymlinkPathError) as exc:
                    raise UnsafeRepositoryError(
                        f"unsafe temporary workspace directory: {normalized}"
                    ) from exc
                pending.append(entry)
            elif stat.S_ISREG(mode):
                total_bytes += mode_result.st_size
                if total_bytes > _MAX_ARCHIVE_BYTES:
                    raise UnsafeRepositoryError(
                        "workspace exceeds the 1 GiB file-content limit"
                    )
            else:
                raise UnsafeRepositoryError(
                    f"special filesystem entries are not allowed: {normalized}"
                )


@dataclass(slots=True)
class Workspace:
    """A temporary editable snapshot with an independent Git baseline."""

    root: Path
    source_root: Path
    original_commit: str
    baseline_commit: str
    _temporary_directory: tempfile.TemporaryDirectory[str] = field(repr=False)
    _git_directory: Path = field(repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @classmethod
    def create(
        cls,
        repository: str | os.PathLike[str],
        *,
        temp_parent: str | os.PathLike[str] | None = None,
    ) -> Workspace:
        return create_workspace(repository, temp_parent=temp_parent)

    @property
    def source_repo(self) -> Path:
        """Compatibility alias for the immutable source repository root."""

        return self.source_root

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise WorkspaceClosedError("temporary workspace has already been cleaned up")

    @contextmanager
    def _temporary_index(self) -> Iterator[Mapping[str, str]]:
        self._ensure_open()
        descriptor, raw_path = tempfile.mkstemp(
            prefix=".coderepair-index-",
            dir=self.root.parent,
        )
        os.close(descriptor)
        index_path = Path(raw_path)
        index_path.unlink()
        environment = {"GIT_INDEX_FILE": str(index_path)}
        try:
            _run_git(
                self.root,
                ["read-tree", "HEAD"],
                extra_environment=environment,
                git_directory=self._git_directory,
            )
            _run_git(
                self.root,
                ["add", "--all", "--", "."],
                extra_environment=environment,
                git_directory=self._git_directory,
            )
            yield environment
        finally:
            index_path.unlink(missing_ok=True)
            Path(f"{index_path}.lock").unlink(missing_ok=True)

    def diff(self) -> str:
        """Return a binary-capable Git patch including new, changed, and deleted files."""

        self._ensure_open()
        _validate_workspace_tree(self.root)
        with self._temporary_index() as environment:
            result = _run_git(
                self.root,
                [
                    "diff",
                    "--cached",
                    "--binary",
                    "--full-index",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--no-renames",
                    "HEAD",
                    "--",
                ],
                extra_environment=environment,
                git_directory=self._git_directory,
            )
        return str(result.stdout)

    def modified_files(self) -> tuple[str, ...]:
        """Return sorted Git paths represented by the current workspace patch."""

        self._ensure_open()
        _validate_workspace_tree(self.root)
        with self._temporary_index() as environment:
            result = _run_git(
                self.root,
                [
                    "diff",
                    "--cached",
                    "--name-only",
                    "-z",
                    "--diff-filter=ACDMRTUXB",
                    "HEAD",
                    "--",
                ],
                text=False,
                extra_environment=environment,
                git_directory=self._git_directory,
            )
        names: list[str] = []
        for raw_name in bytes(result.stdout).split(b"\0"):
            if not raw_name:
                continue
            name = raw_name.decode("utf-8", errors="surrogateescape")
            try:
                normalized = normalize_relative_path(name)
            except PathSecurityError as exc:
                raise UnsafeRepositoryError(
                    f"Git reported an unsafe modified path: {name!r}"
                ) from exc
            names.append(normalized.as_posix())
        return tuple(sorted(set(names)))

    def cleanup(self) -> None:
        """Remove the temporary snapshot. This operation is idempotent."""

        if self._closed:
            return
        self._closed = True
        self._temporary_directory.cleanup()

    def __enter__(self) -> Workspace:
        self._ensure_open()
        return self

    def __exit__(self, *_exception: object) -> None:
        self.cleanup()


def create_workspace(
    repository: str | os.PathLike[str],
    *,
    temp_parent: str | os.PathLike[str] | None = None,
) -> Workspace:
    """Create an isolated editable snapshot of a clean repository at exact HEAD."""

    source_root = _repository_root(repository)
    _assert_safe_local_config(source_root)
    original_commit = _head_commit(source_root)
    _assert_safe_commit_tree(source_root, original_commit)
    _assert_clean(source_root)

    parent: str | None = None
    if temp_parent is not None:
        candidate_parent = Path(temp_parent)
        if candidate_parent.is_symlink():
            raise UnsafeRepositoryError("temporary workspace parent cannot be a symlink")
        try:
            resolved_parent = candidate_parent.resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise WorkspaceError(
                f"temporary workspace parent does not exist: {candidate_parent}"
            ) from exc
        if not resolved_parent.is_dir():
            raise WorkspaceError(
                f"temporary workspace parent is not a directory: {resolved_parent}"
            )
        parent = str(resolved_parent)

    temporary_directory = tempfile.TemporaryDirectory(
        prefix="coderepair-",
        dir=parent,
    )
    temporary_root = Path(temporary_directory.name)
    workspace_root = temporary_root / "repo"
    git_directory = temporary_root / "baseline.git"
    archive_path = temporary_root / "snapshot.tar"
    try:
        workspace_root.mkdir(mode=0o700)
        _create_archive(source_root, original_commit, archive_path)
        _extract_archive(archive_path, workspace_root)
        archive_path.unlink(missing_ok=True)

        _assert_safe_local_config(source_root)
        if _head_commit(source_root) != original_commit:
            raise RepositoryChangedError(
                "source repository HEAD changed while its snapshot was created"
            )
        later_changes = _status_entries(source_root)
        if later_changes:
            raise RepositoryChangedError(
                "source repository changed while its snapshot was created"
            )

        baseline_commit = _initialize_baseline(workspace_root, git_directory)
        return Workspace(
            root=workspace_root,
            source_root=source_root,
            original_commit=original_commit,
            baseline_commit=baseline_commit,
            _temporary_directory=temporary_directory,
            _git_directory=git_directory,
        )
    except Exception:
        temporary_directory.cleanup()
        raise


__all__ = [
    "DirtyRepositoryError",
    "GitCommandError",
    "GitNotAvailableError",
    "InvalidHeadError",
    "NotGitRepositoryError",
    "RepositoryChangedError",
    "UnsafeRepositoryError",
    "Workspace",
    "WorkspaceClosedError",
    "WorkspaceError",
    "create_workspace",
]
