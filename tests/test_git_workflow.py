from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from coderepair.git_workflow import (
    GitWorkflowError,
    _paths_from_patch,
    apply_run_to_branch,
)


def git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr)
    return completed.stdout.strip()


@pytest.fixture
def repository_and_run(tmp_path: Path) -> tuple[Path, Path, str, str]:
    if shutil.which("git") is None:
        pytest.skip("Git is not installed")
    repository = tmp_path / "repo"
    repository.mkdir()
    git(repository, "init", "--quiet")
    git(repository, "config", "user.name", "CodeRepair Tests")
    git(repository, "config", "user.email", "tests@coderepair.invalid")
    git(repository, "config", "commit.gpgsign", "false")
    git(repository, "config", "core.autocrlf", "false")
    source = repository / "app.py"
    original = b"value = 1\n"
    source.write_bytes(original)
    git(repository, "add", "app.py")
    git(repository, "commit", "--quiet", "-m", "baseline")
    commit = git(repository, "rev-parse", "HEAD")

    source.write_bytes(b"value = 2\n")
    patch_result = subprocess.run(
        [
            "git",
            "diff",
            "--binary",
            "--full-index",
            "--no-ext-diff",
            "--",
            "app.py",
        ],
        cwd=repository,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    patch = patch_result.stdout
    source.write_bytes(original)
    assert git(repository, "status", "--porcelain=v1") == ""

    run_directory = tmp_path / "run"
    run_directory.mkdir()
    (run_directory / "result.json").write_text(
        json.dumps({"source": {"commit": commit}}), encoding="utf-8"
    )
    (run_directory / "diff.patch").write_text(patch, encoding="utf-8", newline="\n")
    return repository, run_directory, commit, patch


def test_refuses_every_repository_write_without_explicit_approval(
    repository_and_run: tuple[Path, Path, str, str],
) -> None:
    repository, run_directory, commit, _patch = repository_and_run

    with pytest.raises(GitWorkflowError, match="without --approve"):
        apply_run_to_branch(
            repository=repository,
            run_directory=run_directory,
            branch="coderepair/fix",
            commit_message="Apply fix",
            approved=False,
        )

    assert git(repository, "rev-parse", "HEAD") == commit
    assert git(repository, "branch", "--show-current") != "coderepair/fix"
    assert git(repository, "status", "--porcelain=v1") == ""


def test_applies_reviewed_patch_to_new_branch_and_commits_only_expected_files(
    repository_and_run: tuple[Path, Path, str, str],
) -> None:
    repository, run_directory, commit, _patch = repository_and_run

    result = apply_run_to_branch(
        repository=repository,
        run_directory=run_directory,
        branch="coderepair/fix-boundary",
        commit_message="Fix boundary",
        approved=True,
    )

    assert result.branch == "coderepair/fix-boundary"
    assert result.modified_files == ("app.py",)
    assert result.pushed is False
    assert result.draft_pr_url is None
    assert git(repository, "branch", "--show-current") == "coderepair/fix-boundary"
    assert git(repository, "rev-parse", "HEAD^") == commit
    assert git(repository, "show", "HEAD:app.py") == "value = 2"
    assert git(repository, "show", "--format=%s", "--no-patch", "HEAD") == "Fix boundary"
    assert git(repository, "status", "--porcelain=v1") == ""


def test_commit_mismatch_is_checked_before_branch_creation(
    repository_and_run: tuple[Path, Path, str, str],
) -> None:
    repository, run_directory, _commit, _patch = repository_and_run
    (run_directory / "result.json").write_text(
        json.dumps({"source": {"commit": "f" * 40}}), encoding="utf-8"
    )

    with pytest.raises(GitWorkflowError, match="source commit mismatch"):
        apply_run_to_branch(
            repository=repository,
            run_directory=run_directory,
            branch="coderepair/wrong-start",
            commit_message="Fix",
            approved=True,
        )

    assert "coderepair/wrong-start" not in git(repository, "branch", "--list")


def test_dirty_repository_is_rejected_before_branch_creation(
    repository_and_run: tuple[Path, Path, str, str],
) -> None:
    repository, run_directory, _commit, _patch = repository_and_run
    (repository / "untracked.txt").write_text("mine", encoding="utf-8")

    with pytest.raises(GitWorkflowError, match="must be clean"):
        apply_run_to_branch(
            repository=repository,
            run_directory=run_directory,
            branch="coderepair/dirty",
            commit_message="Fix",
            approved=True,
        )

    assert "coderepair/dirty" not in git(repository, "branch", "--list")


@pytest.mark.parametrize(
    "branch",
    ["-danger", "../escape", "feature..lock", "trailing/", "with space"],
)
def test_unsafe_branch_names_are_rejected_before_git(
    repository_and_run: tuple[Path, Path, str, str], branch: str
) -> None:
    repository, run_directory, _commit, _patch = repository_and_run

    with pytest.raises(GitWorkflowError, match="branch name"):
        apply_run_to_branch(
            repository=repository,
            run_directory=run_directory,
            branch=branch,
            commit_message="Fix",
            approved=True,
        )


def test_draft_pr_requires_push_before_any_repository_change(
    repository_and_run: tuple[Path, Path, str, str],
) -> None:
    repository, run_directory, commit, _patch = repository_and_run

    with pytest.raises(GitWorkflowError, match="requires --push"):
        apply_run_to_branch(
            repository=repository,
            run_directory=run_directory,
            branch="coderepair/draft",
            commit_message="Fix",
            approved=True,
            create_draft_pr=True,
        )

    assert git(repository, "rev-parse", "HEAD") == commit


class PublishingRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, arguments: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        argv = tuple(arguments)
        self.calls.append(argv)
        if argv[:2] == ("git", "push"):
            return subprocess.CompletedProcess(list(argv), 0, "", "")
        if argv[:3] == ("gh", "pr", "create"):
            return subprocess.CompletedProcess(
                list(argv), 0, "https://github.example/pull/123\n", ""
            )
        return subprocess.run(list(argv), **kwargs)


def test_push_and_draft_pr_use_explicit_safe_commands_after_approval(
    repository_and_run: tuple[Path, Path, str, str],
) -> None:
    repository, run_directory, _commit, _patch = repository_and_run
    runner = PublishingRunner()

    result = apply_run_to_branch(
        repository=repository,
        run_directory=run_directory,
        branch="coderepair/publish",
        commit_message="Fix safely",
        approved=True,
        push=True,
        create_draft_pr=True,
        pr_title="Draft fix",
        runner=runner,
    )

    assert result.pushed is True
    assert result.draft_pr_url == "https://github.example/pull/123"
    assert ("git", "push", "--set-upstream", "origin", "coderepair/publish") in runner.calls
    gh_call = next(call for call in runner.calls if call[:3] == ("gh", "pr", "create"))
    assert "--draft" in gh_call
    assert gh_call[gh_call.index("--head") + 1] == "coderepair/publish"
    commit_call = next(call for call in runner.calls if "commit" in call)
    assert commit_call[:2] == ("git", "-c")
    assert commit_call[2].startswith("core.hooksPath=")


@pytest.mark.parametrize(
    "patch",
    [
        "diff --git a/../secret b/../secret\n",
        "diff --git a/.git/config b/.git/config\n",
        "diff --git a/old.py b/new.py\n",
        "diff --git /absolute /absolute\n",
    ],
)
def test_patch_path_parser_rejects_unsafe_or_renamed_paths(patch: str) -> None:
    with pytest.raises(GitWorkflowError):
        _paths_from_patch(patch)
