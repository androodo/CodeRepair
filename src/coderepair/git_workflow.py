"""Explicitly approved application of a completed run to an original repository."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

BRANCH_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")


class GitWorkflowError(RuntimeError):
    """Approval, repository-state, or publication failure."""


@dataclass(frozen=True)
class GitWorkflowResult:
    branch: str
    commit: str
    pushed: bool
    draft_pr_url: str | None
    modified_files: tuple[str, ...]


Runner = Callable[..., subprocess.CompletedProcess[str]]


def apply_run_to_branch(
    *,
    repository: Path,
    run_directory: Path,
    branch: str,
    commit_message: str,
    approved: bool,
    push: bool = False,
    create_draft_pr: bool = False,
    pr_title: str | None = None,
    pr_body: str | None = None,
    runner: Runner = subprocess.run,
) -> GitWorkflowResult:
    """Apply one run artifact after an explicit, command-line approval gate.

    All checks occur before branch creation. If a later Git or network operation
    fails, the new branch is intentionally left in place for human inspection;
    this function never resets or discards repository data automatically.
    """

    if not approved:
        raise GitWorkflowError("refusing to modify the repository without --approve")
    if create_draft_pr and not push:
        raise GitWorkflowError("--draft-pr requires --push")
    if not BRANCH_PATTERN.fullmatch(branch) or ".." in branch or branch.endswith("/"):
        raise GitWorkflowError("branch name contains unsafe or unsupported characters")
    if not commit_message.strip() or "\x00" in commit_message:
        raise GitWorkflowError("commit message cannot be empty or contain NUL")

    repository = repository.resolve(strict=True)
    run_directory = run_directory.resolve(strict=True)
    result_path = run_directory / "result.json"
    patch_path = run_directory / "diff.patch"
    if not result_path.is_file() or not patch_path.is_file():
        raise GitWorkflowError("run directory must contain result.json and diff.patch")
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise GitWorkflowError(f"invalid result.json: {exc}") from exc
    expected_commit = result.get("source", {}).get("commit")
    if not isinstance(expected_commit, str) or not re.fullmatch(
        r"[0-9a-fA-F]{40,64}", expected_commit
    ):
        raise GitWorkflowError("result.json does not contain a valid source commit")
    patch = patch_path.read_text(encoding="utf-8")
    if not patch.strip():
        raise GitWorkflowError("run diff is empty")
    expected_files = _paths_from_patch(patch)

    top = _git(repository, ("rev-parse", "--show-toplevel"), runner=runner).strip()
    if Path(top).resolve() != repository:
        raise GitWorkflowError("--repo must be the Git repository root")
    if _git(repository, ("status", "--porcelain=v1", "--untracked-files=all"), runner=runner):
        raise GitWorkflowError("original repository must be clean before applying a run")
    actual_commit = _git(repository, ("rev-parse", "HEAD"), runner=runner).strip()
    if actual_commit.lower() != expected_commit.lower():
        raise GitWorkflowError(
            f"source commit mismatch: run expects {expected_commit}, repository is {actual_commit}"
        )
    branch_ref = _run(
        ("git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"),
        cwd=repository,
        runner=runner,
        check=False,
    )
    if branch_ref.returncode == 0:
        raise GitWorkflowError(f"branch already exists: {branch}")
    _git(repository, ("apply", "--check", "--whitespace=nowarn", str(patch_path)), runner=runner)

    _git(repository, ("switch", "--create", branch), runner=runner)
    _git(
        repository,
        ("apply", "--index", "--whitespace=nowarn", str(patch_path)),
        runner=runner,
    )
    staged = tuple(
        sorted(
            line
            for line in _git(
                repository,
                ("diff", "--cached", "--name-only", "--diff-filter=ACDMRT"),
                runner=runner,
            ).splitlines()
            if line
        )
    )
    if staged != expected_files:
        raise GitWorkflowError(
            f"staged paths differ from reviewed patch: expected {expected_files}, got {staged}"
        )

    # Disable repository hooks for the automated commit. Hooks are code on the
    # host and are outside the approved patch operation.
    with tempfile.TemporaryDirectory(prefix="coderepair-empty-hooks-") as hooks:
        _git(
            repository,
            ("-c", f"core.hooksPath={hooks}", "commit", "--message", commit_message.strip()),
            runner=runner,
        )
    commit = _git(repository, ("rev-parse", "HEAD"), runner=runner).strip()

    if push:
        _git(repository, ("push", "--set-upstream", "origin", branch), runner=runner)

    draft_pr_url: str | None = None
    if create_draft_pr:
        title = (pr_title or commit_message).strip()
        body = pr_body or (
            "Draft patch prepared by CodeRepair after explicit human approval.\n\n"
            "Review the complete diff and CI results before marking ready."
        )
        completed = _run(
            (
                "gh",
                "pr",
                "create",
                "--draft",
                "--head",
                branch,
                "--title",
                title,
                "--body",
                body,
            ),
            cwd=repository,
            runner=runner,
        )
        draft_pr_url = completed.stdout.strip() or None

    return GitWorkflowResult(branch, commit, push, draft_pr_url, staged)


def _paths_from_patch(patch: str) -> tuple[str, ...]:
    paths: set[str] = set()
    in_header = False
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            in_header = True
            continue
        if line.startswith("@@") or line.startswith("GIT binary patch"):
            in_header = False
            continue
        if not in_header or not line.startswith(("--- ", "+++ ")):
            continue
        value = line[4:]
        if value == "/dev/null":
            continue
        if not value.startswith(("a/", "b/")):
            raise GitWorkflowError("patch contains a non-canonical file header")
        relative = value[2:]
        candidate = PurePosixPath(relative)
        if (
            not relative
            or candidate.is_absolute()
            or ".." in candidate.parts
            or any(part.casefold() == ".git" for part in candidate.parts)
            or "\x00" in relative
            or "\\" in relative
        ):
            raise GitWorkflowError(f"unsafe path in patch: {relative!r}")
        paths.add(relative)
    if not paths:
        raise GitWorkflowError("patch has no Git diff headers")
    return tuple(sorted(paths))


def _git(repository: Path, arguments: Sequence[str], *, runner: Runner) -> str:
    completed = _run(("git", *arguments), cwd=repository, runner=runner)
    return completed.stdout.strip()


def _run(
    arguments: Sequence[str],
    *,
    cwd: Path,
    runner: Runner,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = runner(
            list(arguments),
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            env=_minimal_git_environment(),
        )
    except OSError as exc:
        raise GitWorkflowError(f"could not execute {arguments[0]}: {exc}") from exc
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise GitWorkflowError(
            f"command failed ({' '.join(arguments[:3])}, exit {completed.returncode}): {detail}"
        )
    return completed


def _minimal_git_environment() -> dict[str, str]:
    allowed = {
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "WINDIR",
        "TEMP",
        "TMP",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "SSH_AUTH_SOCK",
        "USERPROFILE",
        "HOMEDRIVE",
        "HOMEPATH",
        "HOME",
        "GH_TOKEN",
        "GITHUB_TOKEN",
    }
    environment = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    environment.update(
        {
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_CONFIG_NOSYSTEM": "1",
        }
    )
    return environment
