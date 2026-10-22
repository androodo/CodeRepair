"""Reproducible, leakage-resistant evaluation of repair agents.

The evaluator deliberately keeps benchmark configuration narrower than the
interactive CLI.  A manifest pins every source snapshot, test commands are
approved before an agent is constructed, and independent tests are mounted
only after the agent has stopped.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from decimal import Decimal
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, Protocol

from .agent import Agent, AgentLimits, AgentOutcome, RunStatus
from .model import JsonObject, ModelClient
from .reporting import Pricing
from .sandbox import (
    ApprovedPytestCommand,
    DockerSandbox,
    ReadOnlyMount,
    approve_pytest_command,
)
from .tools import DEFAULT_PROTECTED_GLOBS, ToolDispatcher
from .workspace import create_workspace

MANIFEST_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1
_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_FULL_OBJECT_ID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_TIMEOUT_SECONDS = 120.0


class EvaluationError(RuntimeError):
    """Base class for trusted evaluation configuration and execution errors."""


class ManifestError(EvaluationError, ValueError):
    """A benchmark manifest is invalid or ambiguous."""


class SourceVerificationError(EvaluationError):
    """An immutable source pin does not match the selected source."""


class UnsafeEvaluationPathError(EvaluationError):
    """A source or hidden-evaluation tree contains an unsafe filesystem entry."""


class EvaluationMode(StrEnum):
    BASELINE = "baseline"
    ITERATIVE = "iterative"


@dataclass(frozen=True, slots=True)
class SourceSpec:
    path: Path
    tree_sha256: str | None = None
    commit: str | None = None

    @property
    def pin_kind(self) -> str:
        return "tree_sha256" if self.tree_sha256 is not None else "commit"

    @property
    def pin(self) -> str:
        value = self.tree_sha256 if self.tree_sha256 is not None else self.commit
        assert value is not None
        return value


@dataclass(frozen=True, slots=True)
class HiddenEvaluationSpec:
    path: Path
    target_test_command: ApprovedPytestCommand
    regression_test_command: ApprovedPytestCommand


@dataclass(frozen=True, slots=True)
class BenchmarkTask:
    id: str
    task: str
    source: SourceSpec
    visible_test_command: ApprovedPytestCommand
    hidden_evaluation: HiddenEvaluationSpec
    limits: AgentLimits
    editable_globs: tuple[str, ...] = ("**",)
    protected_globs: tuple[str, ...] = DEFAULT_PROTECTED_GLOBS


@dataclass(frozen=True, slots=True)
class BenchmarkManifest:
    schema_version: int
    tasks: tuple[BenchmarkTask, ...]
    manifest_path: Path | None = None

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> BenchmarkManifest:
        return load_manifest(path)


@dataclass(frozen=True, slots=True)
class EvaluationCheck:
    status: str
    passed: bool | None
    infrastructure_error: bool
    result: JsonObject | None = None
    error: str | None = None

    @classmethod
    def not_run(cls, reason: str) -> EvaluationCheck:
        return cls("not_run", None, True, error=reason)

    @classmethod
    def not_applicable(cls, reason: str) -> EvaluationCheck:
        return cls("not_applicable", None, False, error=reason)

    def to_dict(self) -> JsonObject:
        return {
            "status": self.status,
            "passed": self.passed,
            "infrastructure_error": self.infrastructure_error,
            "result": self.result,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class ModeResult:
    task_id: str
    mode: EvaluationMode
    status: str
    duration_seconds: float
    source: JsonObject
    agent: JsonObject | None
    modified_files: tuple[str, ...]
    diff: str
    visible_tests: EvaluationCheck
    hidden_target_tests: EvaluationCheck
    pristine_regression_tests: EvaluationCheck
    regression_tests: EvaluationCheck
    verified_task_success: bool | None
    estimated_cost_usd: str | None
    infrastructure_error: str | None = None

    def to_dict(self) -> JsonObject:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "task_id": self.task_id,
            "mode": self.mode.value,
            "status": self.status,
            "duration_seconds": round(self.duration_seconds, 6),
            "source": self.source,
            "agent": self.agent,
            "modified_files": list(self.modified_files),
            "diff": self.diff,
            "verification": {
                "visible_tests": self.visible_tests.to_dict(),
                "hidden_target_tests": self.hidden_target_tests.to_dict(),
                "pristine_regression_tests": self.pristine_regression_tests.to_dict(),
                "regression_tests": self.regression_tests.to_dict(),
                "verified_task_success": self.verified_task_success,
            },
            "estimated_cost_usd": self.estimated_cost_usd,
            "infrastructure_error": self.infrastructure_error,
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    results: tuple[ModeResult, ...]
    aggregate: JsonObject
    output_directory: Path


class ModelFactory(Protocol):
    """Construct an independent model client for one task/mode run."""

    def __call__(self, task_id: str, mode: EvaluationMode) -> ModelClient: ...


WorkspaceFactory = Callable[..., Any]
AgentFactory = Callable[..., Agent]


def load_manifest(path: str | os.PathLike[str]) -> BenchmarkManifest:
    """Load a strict v1 JSON manifest, resolving paths beside the manifest."""

    manifest_path = Path(path).resolve(strict=True)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"could not read manifest: {exc}") from exc
    root = _object(raw, "manifest")
    _keys(root, "manifest", required={"schema_version", "tasks"})
    version = _integer(root["schema_version"], "manifest.schema_version")
    if version != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            f"manifest.schema_version must be {MANIFEST_SCHEMA_VERSION}, got {version}"
        )
    tasks_value = root["tasks"]
    if not isinstance(tasks_value, list):
        raise ManifestError("manifest.tasks must be an array")
    base = manifest_path.parent
    tasks = tuple(_parse_task(value, index, base) for index, value in enumerate(tasks_value))
    identifiers = [task.id for task in tasks]
    duplicates = sorted(
        identifier for identifier, count in Counter(identifiers).items() if count > 1
    )
    if duplicates:
        raise ManifestError(f"manifest task ids must be unique: {', '.join(duplicates)}")
    return BenchmarkManifest(version, tasks, manifest_path)


def _parse_task(value: object, index: int, base: Path) -> BenchmarkTask:
    label = f"manifest.tasks[{index}]"
    item = _object(value, label)
    _keys(
        item,
        label,
        required={
            "id",
            "task",
            "source",
            "visible_test_argv",
            "hidden_evaluation",
            "limits",
        },
        optional={"editable_globs", "protected_globs"},
    )
    identifier = _string(item["id"], f"{label}.id")
    if not _TASK_ID.fullmatch(identifier):
        raise ManifestError(f"{label}.id must match {_TASK_ID.pattern!r} (maximum 64 characters)")
    task_text = _string(item["task"], f"{label}.task").strip()
    if not task_text or len(task_text) > 20_000:
        raise ManifestError(f"{label}.task must contain 1 to 20000 non-whitespace characters")

    source_value = _object(item["source"], f"{label}.source")
    _keys(
        source_value,
        f"{label}.source",
        required={"path"},
        optional={"tree_sha256", "commit"},
    )
    has_tree = "tree_sha256" in source_value
    has_commit = "commit" in source_value
    if has_tree == has_commit:
        raise ManifestError(f"{label}.source must contain exactly one of tree_sha256 or commit")
    source_path = _resolve_directory(base, source_value["path"], f"{label}.source.path")
    tree_sha256: str | None = None
    commit: str | None = None
    if has_tree:
        tree_sha256 = _string(source_value["tree_sha256"], f"{label}.source.tree_sha256")
        if not _SHA256.fullmatch(tree_sha256):
            raise ManifestError(f"{label}.source.tree_sha256 must be 64 lowercase hex digits")
    else:
        commit = _string(source_value["commit"], f"{label}.source.commit")
        if not _FULL_OBJECT_ID.fullmatch(commit):
            raise ManifestError(f"{label}.source.commit must be a full lowercase Git object id")

    visible = _approved_argv(item["visible_test_argv"], f"{label}.visible_test_argv")
    if any(_references_evaluation_mount(argument) for argument in visible.argv):
        raise ManifestError(f"{label}.visible_test_argv cannot reference /evaluation")

    hidden_value = _object(item["hidden_evaluation"], f"{label}.hidden_evaluation")
    _keys(
        hidden_value,
        f"{label}.hidden_evaluation",
        required={"path", "target_test_argv", "regression_test_argv"},
    )
    hidden_path = _resolve_directory(base, hidden_value["path"], f"{label}.hidden_evaluation.path")
    target = _approved_argv(
        hidden_value["target_test_argv"],
        f"{label}.hidden_evaluation.target_test_argv",
    )
    regression = _approved_argv(
        hidden_value["regression_test_argv"],
        f"{label}.hidden_evaluation.regression_test_argv",
    )
    for name, command in (("target_test_argv", target), ("regression_test_argv", regression)):
        if not any(_references_evaluation_mount(argument) for argument in command.argv):
            raise ManifestError(f"{label}.hidden_evaluation.{name} must reference /evaluation")
    if _paths_overlap(source_path, hidden_path):
        raise ManifestError(f"{label} source and hidden-evaluation directories must be separate")

    limits = _parse_limits(item["limits"], f"{label}.limits")
    editable = _glob_array(item.get("editable_globs", ["**"]), f"{label}.editable_globs")
    protected = _glob_array(
        item.get("protected_globs", list(DEFAULT_PROTECTED_GLOBS)),
        f"{label}.protected_globs",
    )
    return BenchmarkTask(
        id=identifier,
        task=task_text,
        source=SourceSpec(source_path, tree_sha256=tree_sha256, commit=commit),
        visible_test_command=visible,
        hidden_evaluation=HiddenEvaluationSpec(hidden_path, target, regression),
        limits=limits,
        editable_globs=editable,
        protected_globs=protected,
    )


def _parse_limits(value: object, label: str) -> AgentLimits:
    item = _object(value, label)
    required = {"max_iterations", "max_tool_calls", "max_total_tokens", "max_seconds"}
    optional = {"max_output_tokens", "max_api_retries"}
    _keys(item, label, required=required, optional=optional)
    defaults = AgentLimits()
    values = {
        "max_iterations": _integer(item["max_iterations"], f"{label}.max_iterations"),
        "max_tool_calls": _integer(item["max_tool_calls"], f"{label}.max_tool_calls"),
        "max_total_tokens": _integer(item["max_total_tokens"], f"{label}.max_total_tokens"),
        "max_seconds": _number(item["max_seconds"], f"{label}.max_seconds"),
        "max_output_tokens": _integer(
            item.get("max_output_tokens", defaults.max_output_tokens),
            f"{label}.max_output_tokens",
        ),
        "max_api_retries": _integer(
            item.get("max_api_retries", defaults.max_api_retries),
            f"{label}.max_api_retries",
        ),
    }
    try:
        return AgentLimits(**values)
    except ValueError as exc:
        raise ManifestError(f"{label}: {exc}") from exc


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ManifestError(f"{label} must be a JSON object with string keys")
    return value


def _keys(
    value: Mapping[str, object],
    label: str,
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    allowed = required | (optional or set())
    missing = sorted(required - value.keys())
    unknown = sorted(value.keys() - allowed)
    if missing:
        raise ManifestError(f"{label} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise ManifestError(f"{label} contains unknown fields: {', '.join(unknown)}")


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ManifestError(f"{label} must be a string")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{label} must be an integer")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{label} must be a number")
    result = float(value)
    if not result < float("inf"):
        raise ManifestError(f"{label} must be finite")
    return result


def _resolve_directory(base: Path, value: object, label: str) -> Path:
    text = _string(value, label)
    if not text or "\x00" in text:
        raise ManifestError(f"{label} must be a non-empty filesystem path")
    supplied = Path(text)
    candidate = supplied if supplied.is_absolute() else base / supplied
    if candidate.is_symlink():
        raise ManifestError(f"{label} cannot be a symbolic link")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ManifestError(f"{label} does not identify an existing directory") from exc
    if not resolved.is_dir():
        raise ManifestError(f"{label} must identify a directory")
    return resolved


def _approved_argv(value: object, label: str) -> ApprovedPytestCommand:
    if not isinstance(value, list):
        raise ManifestError(f"{label} must be an argv array")
    try:
        return approve_pytest_command(value)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{label}: {exc}") from exc


def _glob_array(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ManifestError(f"{label} must be a non-empty string array")
    globs: list[str] = []
    for index, candidate in enumerate(value):
        glob = _string(candidate, f"{label}[{index}]")
        if not glob or len(glob) > 500 or "\x00" in glob or "\\" in glob:
            raise ManifestError(f"{label}[{index}] is not a valid portable glob")
        if Path(glob).is_absolute() or ".." in glob.split("/"):
            raise ManifestError(f"{label}[{index}] must stay within the workspace")
        globs.append(glob)
    return tuple(globs)


def _references_evaluation_mount(argument: str) -> bool:
    return argument == "/evaluation" or argument.startswith("/evaluation/")


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def canonical_source_tree_sha256(root: str | os.PathLike[str]) -> str:
    """Return the v1 digest of a link-free ordinary source directory.

    Files are ordered by their UTF-8 POSIX relative path.  Each entry hashes
    ``path NUL normalized-mode NUL decimal-size NUL bytes NUL`` after a domain
    separator.  Modes are normalized to Git's ``100644``/``100755`` values.
    Empty directories are intentionally absent because Git cannot track them.
    """

    directory = _safe_directory(root, "source tree")
    digest = hashlib.sha256(b"coderepair-source-tree-v1\0")
    for relative, path, mode_result in _regular_files(directory, reject_git=True):
        try:
            relative_bytes = relative.encode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise UnsafeEvaluationPathError(
                f"source path is not valid UTF-8: {relative!r}"
            ) from exc
        before = path.stat(follow_symlinks=False)
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise UnsafeEvaluationPathError(f"cannot read source file: {relative}") from exc
        after = path.stat(follow_symlinks=False)
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or not stat.S_ISREG(after.st_mode)
        ):
            raise SourceVerificationError(f"source changed while hashing: {relative}")
        normalized_mode = b"100755" if mode_result.st_mode & 0o111 else b"100644"
        digest.update(relative_bytes)
        digest.update(b"\0")
        digest.update(normalized_mode)
        digest.update(b"\0")
        digest.update(str(len(data)).encode("ascii"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return digest.hexdigest()


def _safe_directory(root: str | os.PathLike[str], label: str) -> Path:
    supplied = Path(root)
    if supplied.is_symlink():
        raise UnsafeEvaluationPathError(f"{label} cannot be a symbolic link")
    try:
        resolved = supplied.resolve(strict=True)
        mode_result = resolved.stat(follow_symlinks=False)
    except OSError as exc:
        raise UnsafeEvaluationPathError(f"{label} is unavailable") from exc
    if not stat.S_ISDIR(mode_result.st_mode) or _is_reparse_point(mode_result):
        raise UnsafeEvaluationPathError(f"{label} must be an ordinary directory")
    return resolved


def _regular_files(
    root: Path, *, reject_git: bool, skip_root_git: bool = False
) -> Iterator[tuple[str, Path, os.stat_result]]:
    found: list[tuple[str, Path, os.stat_result]] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = tuple(os.scandir(directory))
        except OSError as exc:
            raise UnsafeEvaluationPathError(f"cannot inspect directory: {directory}") from exc
        for entry in entries:
            path = Path(entry.path)
            relative_path = path.relative_to(root)
            relative = relative_path.as_posix()
            if entry.name == ".git":
                if skip_root_git and directory == root:
                    continue
                if reject_git or skip_root_git:
                    raise UnsafeEvaluationPathError(
                        f"source trees pinned by digest cannot contain .git: {relative}"
                    )
            try:
                mode_result = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise UnsafeEvaluationPathError(f"cannot inspect entry: {relative}") from exc
            mode = mode_result.st_mode
            if stat.S_ISLNK(mode) or _is_reparse_point(mode_result):
                raise UnsafeEvaluationPathError(f"links are not permitted: {relative}")
            if stat.S_ISDIR(mode):
                pending.append(path)
            elif stat.S_ISREG(mode):
                found.append((relative, path, mode_result))
            else:
                raise UnsafeEvaluationPathError(
                    f"special filesystem entries are not permitted: {relative}"
                )
    found.sort(key=lambda item: item[0].encode("utf-8", errors="strict"))
    yield from found


def _is_reparse_point(mode_result: os.stat_result) -> bool:
    attributes = getattr(mode_result, "st_file_attributes", 0)
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attributes & flag)


def verify_source(source: SourceSpec) -> str:
    """Verify a source pin and return its actual digest or commit."""

    if source.tree_sha256 is not None:
        actual = canonical_source_tree_sha256(source.path)
        if actual != source.tree_sha256:
            raise SourceVerificationError(
                f"source tree digest mismatch for {source.path}: "
                f"expected {source.tree_sha256}, got {actual}"
            )
        return actual
    assert source.commit is not None
    actual = _verify_git_source(source.path)
    if actual != source.commit:
        raise SourceVerificationError(
            f"source commit mismatch for {source.path}: expected {source.commit}, got {actual}"
        )
    return actual


def _git_environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    environment = dict(os.environ)
    for key in tuple(environment):
        if key.upper().startswith("GIT_"):
            del environment[key]
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
        }
    )
    if extra:
        environment.update(extra)
    return environment


def _run_git(repository: Path, arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    command = [
        "git",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.hooksPath=.git/coderepair-disabled-hooks",
        "--no-pager",
        "-C",
        str(repository),
        *arguments,
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_git_environment(),
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SourceVerificationError(f"could not run Git for {repository}: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "no error output"
        raise SourceVerificationError(f"Git verification failed for {repository}: {detail}")
    return completed


def _verify_git_source(repository: Path) -> str:
    top = Path(_run_git(repository, ["rev-parse", "--show-toplevel"]).stdout.strip()).resolve()
    if os.path.normcase(str(top)) != os.path.normcase(str(repository.resolve())):
        raise SourceVerificationError("source.path must identify the Git working-tree root")
    commit = _run_git(repository, ["rev-parse", "--verify", "HEAD^{commit}"]).stdout.strip().lower()
    if not _FULL_OBJECT_ID.fullmatch(commit):
        raise SourceVerificationError("Git returned an invalid full HEAD object id")
    status = _run_git(
        repository,
        ["status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"],
    ).stdout
    if status:
        raise SourceVerificationError("commit-pinned source repository must be clean")
    tree = _run_git(repository, ["ls-tree", "-r", "-z", "--full-tree", commit]).stdout
    for entry in tree.split("\0"):
        if not entry:
            continue
        header, _, path = entry.partition("\t")
        mode, object_type, _object_id = header.split(" ", 2)
        if mode in {"120000", "160000"} or object_type != "blob":
            raise SourceVerificationError(f"unsupported linked Git entry: {path}")
    return commit


@contextmanager
def _source_repository_for_mode(source: SourceSpec, *, temp_parent: Path | None) -> Iterator[Path]:
    verify_source(source)
    if source.commit is not None:
        yield source.path
        return
    parent_text = str(temp_parent) if temp_parent is not None else None
    with tempfile.TemporaryDirectory(prefix="coderepair-eval-source-", dir=parent_text) as raw:
        repository = Path(raw) / "repository"
        _copy_plain_tree(source.path, repository, skip_root_git=False)
        assert source.tree_sha256 is not None
        copied_digest = canonical_source_tree_sha256(repository)
        if copied_digest != source.tree_sha256:
            raise SourceVerificationError("source changed while creating the per-mode snapshot")
        _initialize_repository(repository)
        yield repository


def _initialize_repository(repository: Path) -> str:
    repository.mkdir(parents=True, exist_ok=True)
    _run_git(repository, ["init", "--quiet", "--initial-branch=coderepair-evaluation"])
    _run_git(repository, ["config", "--local", "core.autocrlf", "false"])
    _run_git(repository, ["config", "--local", "commit.gpgsign", "false"])
    _run_git(repository, ["add", "--all", "--force", "--", "."])
    identity = {
        "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
        "GIT_AUTHOR_EMAIL": "evaluation@coderepair.invalid",
        "GIT_AUTHOR_NAME": "CodeRepair Evaluation",
        "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
        "GIT_COMMITTER_EMAIL": "evaluation@coderepair.invalid",
        "GIT_COMMITTER_NAME": "CodeRepair Evaluation",
    }
    command = [
        "git",
        "-c",
        "core.hooksPath=.git/coderepair-disabled-hooks",
        "--no-pager",
        "-C",
        str(repository),
        "commit",
        "--quiet",
        "--allow-empty",
        "-m",
        "CodeRepair evaluation source",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_git_environment(identity),
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SourceVerificationError(f"could not initialize source repository: {exc}") from exc
    if completed.returncode != 0:
        raise SourceVerificationError(
            "could not commit source repository: " + (completed.stderr.strip() or "unknown error")
        )
    return _verify_git_source(repository)


def _copy_plain_tree(source: Path, destination: Path, *, skip_root_git: bool) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for relative, path, mode_result in _regular_files(
        source, reject_git=not skip_root_git, skip_root_git=skip_root_git
    ):
        target = destination.joinpath(*relative.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("rb") as input_file, target.open("xb") as output_file:
                shutil.copyfileobj(input_file, output_file, length=1024 * 1024)
            target.chmod(0o755 if mode_result.st_mode & 0o111 else 0o644)
        except OSError as exc:
            raise UnsafeEvaluationPathError(f"could not copy evaluation file: {relative}") from exc


class EvaluationEngine:
    """Run baseline and iterative configurations over a verified manifest."""

    def __init__(
        self,
        model_factory: ModelFactory,
        *,
        sandbox: Any | None = None,
        workspace_factory: WorkspaceFactory = create_workspace,
        agent_factory: AgentFactory = Agent,
        pricing: Pricing | None = None,
        model_name: str | None = None,
        temp_parent: str | os.PathLike[str] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.model_factory = model_factory
        self.sandbox = sandbox if sandbox is not None else DockerSandbox()
        self.workspace_factory = workspace_factory
        self.agent_factory = agent_factory
        self.pricing = pricing
        self.model_name = model_name
        self.temp_parent = Path(temp_parent).resolve() if temp_parent is not None else None
        self.clock = clock

    def run(
        self,
        manifest: BenchmarkManifest | str | os.PathLike[str],
        output_directory: str | os.PathLike[str],
        *,
        modes: Sequence[EvaluationMode | str] = (
            EvaluationMode.BASELINE,
            EvaluationMode.ITERATIVE,
        ),
    ) -> EvaluationReport:
        parsed = load_manifest(manifest) if isinstance(manifest, (str, os.PathLike)) else manifest
        selected_modes = _normalize_modes(modes)
        output = Path(output_directory).resolve()

        # This complete preflight intentionally precedes output creation and every
        # model-factory call. One bad task cannot spend model tokens on earlier tasks.
        self._preflight(parsed, output)
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise EvaluationError("evaluation output directory must be absent or empty")
        output.mkdir(parents=True, exist_ok=True)

        results: list[ModeResult] = []
        for task in parsed.tasks:
            for mode in selected_modes:
                result = self._run_mode(task, mode)
                results.append(result)
                result_path = output / "tasks" / task.id / f"{mode.value}.json"
                _write_json(result_path, result.to_dict())

        aggregate = aggregate_results(results, task_count=len(parsed.tasks), modes=selected_modes)
        aggregate["model"] = self.model_name
        _write_json(output / "aggregate.json", aggregate)
        rendered_report = render_markdown_summary(aggregate)
        _write_text(output / "report.md", rendered_report)
        _write_text(output / "summary.md", rendered_report)
        return EvaluationReport(tuple(results), aggregate, output)

    def _preflight(self, manifest: BenchmarkManifest, output: Path) -> None:
        if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ManifestError(f"unsupported manifest schema version: {manifest.schema_version}")
        for task in manifest.tasks:
            verify_source(task.source)
            _validate_hidden_tree(task.hidden_evaluation.path)
            if _paths_overlap(output, task.source.path):
                raise EvaluationError("evaluation output directory cannot overlap a source tree")
            if _paths_overlap(output, task.hidden_evaluation.path):
                raise EvaluationError("evaluation output directory cannot overlap hidden tests")

    def _run_mode(self, task: BenchmarkTask, mode: EvaluationMode) -> ModeResult:
        started = self.clock()
        source_result: JsonObject = {
            "path": str(task.source.path),
            "pin_kind": task.source.pin_kind,
            "pin": task.source.pin,
            "verified": True,
            "workspace_original_commit": None,
        }
        outcome: AgentOutcome | None = None
        modified_files: tuple[str, ...] = ()
        diff = ""
        agent_error: str | None = None
        visible = EvaluationCheck.not_run("candidate workspace was not created")
        hidden = EvaluationCheck.not_run("candidate workspace was not created")
        pristine_regression = EvaluationCheck.not_run(
            "pristine workspace was not created"
        )
        regression = EvaluationCheck.not_run("candidate workspace was not created")

        try:
            with _source_repository_for_mode(
                task.source, temp_parent=self.temp_parent
            ) as repository:
                workspace_object = self.workspace_factory(repository, temp_parent=self.temp_parent)
                with workspace_object as workspace:
                    source_result["workspace_original_commit"] = str(workspace.original_commit)
                    if (
                        task.source.commit is not None
                        and workspace.original_commit != task.source.commit
                    ):
                        raise SourceVerificationError(
                            "workspace factory returned a snapshot from the wrong commit"
                        )
                    with tempfile.TemporaryDirectory(
                        prefix="coderepair-eval-pristine-",
                        dir=str(self.temp_parent) if self.temp_parent is not None else None,
                    ) as pristine_parent:
                        pristine = Path(pristine_parent) / "workspace"
                        _copy_plain_tree(Path(workspace.root), pristine, skip_root_git=True)
                        dispatcher = ToolDispatcher(
                            Path(workspace.root),
                            test_runner=self.sandbox if mode is EvaluationMode.ITERATIVE else None,
                            approved_test_command=(
                                task.visible_test_command
                                if mode is EvaluationMode.ITERATIVE
                                else None
                            ),
                            enable_test_tool=mode is EvaluationMode.ITERATIVE,
                            editable_globs=task.editable_globs,
                            protected_globs=task.protected_globs,
                        )
                        try:
                            model = self.model_factory(task.id, mode)
                            agent = self.agent_factory(model, limits=task.limits)
                            outcome = agent.run(task=task.task, tools=dispatcher)
                        except Exception as exc:
                            agent_error = _exception_text("agent execution", exc)
                        try:
                            diff = workspace.diff()
                            modified_files = tuple(workspace.modified_files())
                        except Exception as exc:
                            agent_error = _join_error(
                                agent_error, _exception_text("candidate diff capture", exc)
                            )

                        with tempfile.TemporaryDirectory(
                            prefix="coderepair-eval-candidate-",
                            dir=str(self.temp_parent) if self.temp_parent is not None else None,
                        ) as candidate_parent:
                            candidate = Path(candidate_parent) / "workspace"
                            _copy_plain_tree(Path(workspace.root), candidate, skip_root_git=True)
                            visible = self._run_check(candidate, task.visible_test_command)
                            _validate_hidden_tree(task.hidden_evaluation.path)
                            mount = (ReadOnlyMount(task.hidden_evaluation.path, "/evaluation"),)
                            hidden = self._run_check(
                                candidate,
                                task.hidden_evaluation.target_test_command,
                                extra_mounts=mount,
                            )
                            pristine_regression = self._run_check(
                                pristine,
                                task.hidden_evaluation.regression_test_command,
                                extra_mounts=mount,
                            )
                            candidate_regression = self._run_check(
                                candidate,
                                task.hidden_evaluation.regression_test_command,
                                extra_mounts=mount,
                            )
                            regression = _classify_regression(
                                pristine_regression, candidate_regression
                            )
        except SourceVerificationError:
            raise
        except Exception as exc:
            agent_error = _join_error(agent_error, _exception_text("evaluation setup", exc))

        if outcome is not None and outcome.status is RunStatus.MODEL_ERROR:
            agent_error = _join_error(
                agent_error, f"model request failed: {outcome.error or 'unspecified model error'}"
            )
        check_infrastructure = any(
            check.infrastructure_error for check in (visible, hidden, regression)
        )
        infrastructure = agent_error is not None or check_infrastructure
        verified = (
            None if infrastructure else bool(visible.passed and hidden.passed and regression.passed)
        )
        agent_result = _agent_result(outcome)
        estimated_cost = _estimated_cost(self.pricing, outcome)
        return ModeResult(
            task_id=task.id,
            mode=mode,
            status="infrastructure_error" if infrastructure else "completed",
            duration_seconds=max(0.0, self.clock() - started),
            source=source_result,
            agent=agent_result,
            modified_files=modified_files,
            diff=diff,
            visible_tests=visible,
            hidden_target_tests=hidden,
            pristine_regression_tests=pristine_regression,
            regression_tests=regression,
            verified_task_success=verified,
            estimated_cost_usd=estimated_cost,
            infrastructure_error=agent_error,
        )

    def _run_check(
        self,
        workspace: Path,
        command: ApprovedPytestCommand,
        *,
        extra_mounts: Sequence[ReadOnlyMount] = (),
    ) -> EvaluationCheck:
        try:
            result = self.sandbox.run_tests(
                workspace,
                command,
                extra_mounts=tuple(extra_mounts),
            )
            rendered = _result_dictionary(result)
            infrastructure = bool(
                getattr(result, "infrastructure_error", rendered.get("infrastructure_error", False))
            )
            passed = bool(getattr(result, "passed", rendered.get("passed", False)))
            if infrastructure:
                return EvaluationCheck("infrastructure_error", None, True, rendered)
            return EvaluationCheck("passed" if passed else "failed", passed, False, rendered)
        except Exception as exc:
            return EvaluationCheck(
                "infrastructure_error",
                None,
                True,
                error=_exception_text("sandbox execution", exc),
            )


def _classify_regression(
    baseline: EvaluationCheck, candidate: EvaluationCheck
) -> EvaluationCheck:
    evidence = {"baseline": baseline.to_dict(), "candidate": candidate.to_dict()}
    if baseline.infrastructure_error:
        return EvaluationCheck(
            "infrastructure_error",
            None,
            True,
            evidence,
            "could not establish the regression baseline",
        )
    if baseline.passed is not True:
        return EvaluationCheck(
            "invalid_baseline",
            None,
            True,
            evidence,
            "regression suite did not pass on the untouched source snapshot",
        )
    if candidate.infrastructure_error:
        return EvaluationCheck(
            "infrastructure_error",
            None,
            True,
            evidence,
            candidate.error,
        )
    return EvaluationCheck(
        "passed" if candidate.passed else "failed",
        candidate.passed,
        False,
        evidence,
        candidate.error,
    )


def run_evaluation(
    manifest_path: str | os.PathLike[str],
    output_directory: str | os.PathLike[str],
    model_factory: Callable[[], ModelClient],
    sandbox: Any | None = None,
    modes: Sequence[EvaluationMode | str] = (
        EvaluationMode.BASELINE,
        EvaluationMode.ITERATIVE,
    ),
    model_name: str | None = None,
) -> JsonObject:
    """CLI-friendly facade for a model factory with no task-specific inputs.

    A new client is requested for every task/mode pair, which keeps histories
    independent while ensuring benchmark metadata and hidden paths are never
    supplied to the model factory.
    """

    def per_run_factory(_task_id: str, _mode: EvaluationMode) -> ModelClient:
        return model_factory()

    report = EvaluationEngine(
        per_run_factory,
        sandbox=sandbox,
        model_name=model_name,
    ).run(manifest_path, output_directory, modes=modes)
    return report.aggregate


def _validate_hidden_tree(path: Path) -> None:
    root = _safe_directory(path, "hidden-evaluation tree")
    # Exhausting the iterator performs link/special-file validation. Hidden
    # data are deliberately not read or copied into an agent-visible location.
    tuple(_regular_files(root, reject_git=False))


def _normalize_modes(modes: Sequence[EvaluationMode | str]) -> tuple[EvaluationMode, ...]:
    if not modes:
        raise EvaluationError("at least one evaluation mode is required")
    normalized: list[EvaluationMode] = []
    for value in modes:
        try:
            mode = value if isinstance(value, EvaluationMode) else EvaluationMode(value)
        except ValueError as exc:
            raise EvaluationError(f"unknown evaluation mode: {value!r}") from exc
        if mode in normalized:
            raise EvaluationError(f"duplicate evaluation mode: {mode.value}")
        normalized.append(mode)
    return tuple(normalized)


def _agent_result(outcome: AgentOutcome | None) -> JsonObject | None:
    if outcome is None:
        return None
    result = outcome.metrics_dict()
    result["final_text"] = outcome.final_text
    return result


def _estimated_cost(pricing: Pricing | None, outcome: AgentOutcome | None) -> str | None:
    if pricing is None or outcome is None:
        return None
    return format(pricing.estimate(outcome.usage).quantize(Decimal("0.000001")), "f")


def _result_dictionary(value: Any) -> JsonObject:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        raw = to_dict()
    elif isinstance(value, Mapping):
        raw = dict(value)
    elif is_dataclass(value) and not isinstance(value, type):
        raw = asdict(value)
    else:
        raw = {
            "passed": bool(getattr(value, "passed", False)),
            "infrastructure_error": bool(getattr(value, "infrastructure_error", False)),
        }
    safe = _json_safe(raw)
    if not isinstance(safe, dict):
        raise TypeError("sandbox result must serialize to a JSON object")
    return safe


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _exception_text(context: str, exc: Exception) -> str:
    detail = str(exc).strip()
    return f"{context} failed ({type(exc).__name__})" + (f": {detail}" if detail else "")


def _join_error(first: str | None, second: str) -> str:
    return second if first is None else f"{first}; {second}"


def aggregate_results(
    results: Sequence[ModeResult],
    *,
    task_count: int,
    modes: Sequence[EvaluationMode] | None = None,
) -> JsonObject:
    """Calculate metrics with explicit denominators and undefined zero cases."""

    selected = tuple(modes) if modes is not None else tuple(dict.fromkeys(r.mode for r in results))
    by_mode: JsonObject = {}
    for mode in selected:
        current = [result for result in results if result.mode is mode]
        evaluable = [result for result in current if result.status != "infrastructure_error"]
        successes = sum(result.verified_task_success is True for result in evaluable)
        visible_completed = [
            result for result in evaluable if result.visible_tests.passed is not None
        ]
        hidden_completed = [
            result for result in evaluable if result.hidden_target_tests.passed is not None
        ]
        regression_completed = [
            result for result in evaluable if result.regression_tests.passed is not None
        ]
        outcomes = [result.agent for result in current if result.agent is not None]
        total_iterations = sum(int(outcome["iterations"]) for outcome in outcomes)
        total_tool_calls = sum(int(outcome["tool_calls"]) for outcome in outcomes)
        total_model_requests = sum(int(outcome["model_requests"]) for outcome in outcomes)
        total_duration = sum(result.duration_seconds for result in current)
        input_tokens = sum(int(outcome["usage"]["input_tokens"]) for outcome in outcomes)
        output_tokens = sum(int(outcome["usage"]["output_tokens"]) for outcome in outcomes)
        cache_creation = sum(
            int(outcome["usage"]["cache_creation_input_tokens"]) for outcome in outcomes
        )
        cache_read = sum(int(outcome["usage"]["cache_read_input_tokens"]) for outcome in outcomes)
        costs = [
            Decimal(result.estimated_cost_usd)
            for result in current
            if result.estimated_cost_usd is not None
        ]
        by_mode[mode.value] = {
            "runs": len(current),
            "evaluable_runs": len(evaluable),
            "infrastructure_errors": len(current) - len(evaluable),
            "repair_successes": successes,
            "repair_success_rate": _rate(successes, len(evaluable)),
            "visible_tests": {
                "completed": len(visible_completed),
                "passed": sum(result.visible_tests.passed is True for result in visible_completed),
                "pass_rate": _rate(
                    sum(result.visible_tests.passed is True for result in visible_completed),
                    len(visible_completed),
                ),
            },
            "hidden_target_tests": {
                "completed": len(hidden_completed),
                "passed": sum(
                    result.hidden_target_tests.passed is True for result in hidden_completed
                ),
                "pass_rate": _rate(
                    sum(result.hidden_target_tests.passed is True for result in hidden_completed),
                    len(hidden_completed),
                ),
            },
            "regressions": {
                "completed": len(regression_completed),
                "failures": sum(
                    result.regression_tests.passed is False for result in regression_completed
                ),
                "regression_rate": _rate(
                    sum(result.regression_tests.passed is False for result in regression_completed),
                    len(regression_completed),
                ),
            },
            "agent_outcomes": len(outcomes),
            "iterations": {
                "total": total_iterations,
                "mean": _mean(total_iterations, len(outcomes)),
            },
            "tool_calls": {
                "total": total_tool_calls,
                "mean": _mean(total_tool_calls, len(outcomes)),
            },
            "model_requests": {
                "total": total_model_requests,
                "mean": _mean(total_model_requests, len(outcomes)),
            },
            "latency_seconds": {
                "total": round(total_duration, 6),
                "mean": _mean(total_duration, len(current)),
            },
            "tokens": {
                "input": input_tokens,
                "output": output_tokens,
                "cache_creation_input": cache_creation,
                "cache_read_input": cache_read,
                "total": input_tokens + output_tokens + cache_creation + cache_read,
            },
            "estimated_cost_usd": (
                format(sum(costs, Decimal(0)).quantize(Decimal("0.000001")), "f") if costs else None
            ),
        }

    paired: JsonObject = {
        "evaluable_pairs": 0,
        "iterative_wins": 0,
        "baseline_wins": 0,
        "ties": 0,
        "repair_success_rate_delta": None,
    }
    if EvaluationMode.BASELINE in selected and EvaluationMode.ITERATIVE in selected:
        baseline = {r.task_id: r for r in results if r.mode is EvaluationMode.BASELINE}
        iterative = {r.task_id: r for r in results if r.mode is EvaluationMode.ITERATIVE}
        for identifier in sorted(baseline.keys() & iterative.keys()):
            first = baseline[identifier]
            second = iterative[identifier]
            if first.verified_task_success is None or second.verified_task_success is None:
                continue
            paired["evaluable_pairs"] += 1
            if second.verified_task_success and not first.verified_task_success:
                paired["iterative_wins"] += 1
            elif first.verified_task_success and not second.verified_task_success:
                paired["baseline_wins"] += 1
            else:
                paired["ties"] += 1
        baseline_rate = by_mode[EvaluationMode.BASELINE.value]["repair_success_rate"]
        iterative_rate = by_mode[EvaluationMode.ITERATIVE.value]["repair_success_rate"]
        if baseline_rate is not None and iterative_rate is not None:
            paired["repair_success_rate_delta"] = round(iterative_rate - baseline_rate, 6)

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "task_count": task_count,
        "run_count": len(results),
        "infrastructure_failures": sum(
            result.status == "infrastructure_error" for result in results
        ),
        "modes": by_mode,
        "paired_comparison": paired,
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _mean(total: int | float, denominator: int) -> float | None:
    return round(total / denominator, 6) if denominator else None


def render_markdown_summary(aggregate: Mapping[str, Any]) -> str:
    """Render only measured aggregate values; undefined rates become ``N/A``."""

    lines = [
        "# CodeRepair evaluation",
        "",
        f"Tasks: {aggregate['task_count']}  ",
        f"Runs: {aggregate['run_count']}",
    ]
    if aggregate.get("model") is not None:
        lines.append(f"Model: `{aggregate['model']}`")
    lines.extend(
        [
            "",
        "| Mode | Evaluable | Infrastructure errors | Repair success | Visible pass | "
        "Hidden target pass | Regression rate | Tool calls | Iterations | Latency (s) | "
        "Tokens | Cost (USD) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mode, metrics in aggregate["modes"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    mode,
                    f"{metrics['evaluable_runs']}/{metrics['runs']}",
                    str(metrics["infrastructure_errors"]),
                    _fraction_rate(
                        metrics["repair_successes"],
                        metrics["evaluable_runs"],
                        metrics["repair_success_rate"],
                    ),
                    _fraction_rate(
                        metrics["visible_tests"]["passed"],
                        metrics["visible_tests"]["completed"],
                        metrics["visible_tests"]["pass_rate"],
                    ),
                    _fraction_rate(
                        metrics["hidden_target_tests"]["passed"],
                        metrics["hidden_target_tests"]["completed"],
                        metrics["hidden_target_tests"]["pass_rate"],
                    ),
                    _fraction_rate(
                        metrics["regressions"]["failures"],
                        metrics["regressions"]["completed"],
                        metrics["regressions"]["regression_rate"],
                    ),
                    str(metrics["tool_calls"]["total"]),
                    str(metrics["iterations"]["total"]),
                    f"{metrics['latency_seconds']['total']:.6f}",
                    str(metrics["tokens"]["total"]),
                    metrics["estimated_cost_usd"] or "N/A",
                ]
            )
            + " |"
        )
    comparison = aggregate["paired_comparison"]
    lines.extend(
        [
            "",
            "## Paired comparison",
            "",
            f"- Evaluable pairs: {comparison['evaluable_pairs']}",
            f"- Iterative wins: {comparison['iterative_wins']}",
            f"- Baseline wins: {comparison['baseline_wins']}",
            f"- Ties: {comparison['ties']}",
            "- Repair-success-rate delta: " + _percent(comparison["repair_success_rate_delta"]),
            "",
            "Infrastructure-error runs are counted explicitly and excluded from pass-rate "
            "denominators. Repair success requires visible, hidden-target, and regression "
            "tests to pass in the independent post-run evaluation.",
            "",
        ]
    )
    return "\n".join(lines)


def _fraction_rate(numerator: int, denominator: int, rate: float | None) -> str:
    if rate is None:
        return f"{numerator}/{denominator} (N/A)"
    return f"{numerator}/{denominator} ({rate * 100:.1f}%)"


def _percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:+.1f} percentage points"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_text(
        path, json.dumps(_json_safe(value), indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "BenchmarkManifest",
    "BenchmarkTask",
    "EvaluationCheck",
    "EvaluationEngine",
    "EvaluationError",
    "EvaluationMode",
    "EvaluationReport",
    "HiddenEvaluationSpec",
    "ManifestError",
    "ModeResult",
    "SourceSpec",
    "SourceVerificationError",
    "UnsafeEvaluationPathError",
    "aggregate_results",
    "canonical_source_tree_sha256",
    "load_manifest",
    "render_markdown_summary",
    "run_evaluation",
    "verify_source",
]
