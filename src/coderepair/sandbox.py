"""Restricted Docker execution for approved pytest commands.

This module deliberately has a narrow interface.  A caller approves a pytest
command before an agent starts and gives the resulting
:class:`ApprovedPytestCommand` to :class:`DockerSandbox`.  The agent therefore
cannot replace the test command, image, mounts, or container limits in a tool
call.

Docker is a useful containment boundary, but it is not a perfect security
boundary.  The implementation layers the inexpensive protections Docker
provides and, importantly, never falls back to executing repository code on
the host.
"""

from __future__ import annotations

import re
import shlex
import subprocess
import threading
import time
import uuid
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

DEFAULT_SANDBOX_IMAGE = "coderepair-sandbox:py311"
_DOCKER_CHECK_TIMEOUT_SECONDS = 10.0
_CLEANUP_TIMEOUT_SECONDS = 10.0
_TRUNCATION_MARKER = b"\n...[output truncated]...\n"
_IMAGE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@:-]{0,254}$")


class SandboxError(RuntimeError):
    """Base class for sandbox setup and infrastructure failures."""


class InvalidTestCommandError(SandboxError, ValueError):
    """The trusted test command is not a supported pytest invocation."""


class DockerUnavailableError(SandboxError):
    """The Docker CLI or daemon cannot be used."""


class SandboxImageUnavailableError(SandboxError):
    """The configured prebuilt image is not present locally."""


class InvalidMountError(SandboxError, ValueError):
    """A workspace or evaluator mount is unsafe or cannot be represented."""


@dataclass(frozen=True, slots=True)
class ApprovedPytestCommand:
    """A normalized pytest argv approved outside the model/tool boundary."""

    argv: tuple[str, ...]

    def __post_init__(self) -> None:
        normalized = _normalize_pytest_argv(self.argv)
        object.__setattr__(self, "argv", normalized)

    @classmethod
    def parse(cls, command: str | Sequence[str]) -> ApprovedPytestCommand:
        """Parse a trusted command string or argv without involving a shell."""

        if isinstance(command, str):
            try:
                argv = tuple(shlex.split(command, posix=True))
            except ValueError as exc:
                raise InvalidTestCommandError(f"invalid pytest command quoting: {exc}") from exc
        else:
            argv = tuple(command)
        return cls(argv)


def approve_pytest_command(
    command: str | Sequence[str],
) -> ApprovedPytestCommand:
    """Convenience wrapper used while constructing trusted run configuration."""

    return ApprovedPytestCommand.parse(command)


def _normalize_pytest_argv(argv: Sequence[str]) -> tuple[str, ...]:
    if not argv:
        raise InvalidTestCommandError("pytest command cannot be empty")
    if len(argv) > 128:
        raise InvalidTestCommandError("pytest command has too many arguments")

    checked: list[str] = []
    total_length = 0
    for argument in argv:
        if not isinstance(argument, str):
            raise InvalidTestCommandError("every pytest argument must be a string")
        if not argument:
            # Empty argv values are legal in general, but are never useful to
            # pytest and make audit logs needlessly ambiguous.
            raise InvalidTestCommandError("pytest arguments cannot be empty")
        if len(argument) > 4096:
            raise InvalidTestCommandError("a pytest argument is too long")
        if any(character in argument for character in ("\x00", "\n", "\r")):
            raise InvalidTestCommandError("pytest arguments cannot contain control lines")
        # No shell is used, so these characters would not execute anything.
        # Rejecting them still catches accidentally pasted shell pipelines at
        # the approval boundary instead of silently changing their meaning.
        if any(character in argument for character in (";", "&", "|", "`", ">", "<")):
            raise InvalidTestCommandError(
                "shell operators are not allowed in an approved pytest command"
            )
        if "$(" in argument:
            raise InvalidTestCommandError(
                "shell substitution is not allowed in an approved pytest command"
            )
        total_length += len(argument)
        checked.append(argument)

    if total_length > 16_384:
        raise InvalidTestCommandError("pytest command is too large")

    # Normalize all accepted spellings to the interpreter in the fixed image.
    if checked[0] == "pytest":
        pytest_arguments = checked[1:]
    elif (
        len(checked) >= 3
        and checked[0] in {"python", "python3", "python3.11"}
        and checked[1:3] == ["-m", "pytest"]
    ):
        pytest_arguments = checked[3:]
    else:
        raise InvalidTestCommandError(
            "approved command must start with 'pytest' or 'python -m pytest'"
        )
    return ("python", "-m", "pytest", *pytest_arguments)


@dataclass(frozen=True, slots=True)
class SandboxLimits:
    """Resource and capture limits for one test container."""

    timeout_seconds: float = 120.0
    cpus: float = 1.0
    memory_mb: int = 512
    pids_limit: int = 128
    tmpfs_mb: int = 64
    nofile_limit: int = 256
    max_output_bytes: int = 128 * 1024

    def __post_init__(self) -> None:
        if not 0 < self.timeout_seconds <= 3600:
            raise ValueError("timeout_seconds must be in (0, 3600]")
        if not 0 < self.cpus <= 64:
            raise ValueError("cpus must be in (0, 64]")
        if not 64 <= self.memory_mb <= 65_536:
            raise ValueError("memory_mb must be in [64, 65536]")
        if not 16 <= self.pids_limit <= 4096:
            raise ValueError("pids_limit must be in [16, 4096]")
        if not 16 <= self.tmpfs_mb <= 4096:
            raise ValueError("tmpfs_mb must be in [16, 4096]")
        if not 32 <= self.nofile_limit <= 65_536:
            raise ValueError("nofile_limit must be in [32, 65536]")
        if not 1024 <= self.max_output_bytes <= 16 * 1024 * 1024:
            raise ValueError("max_output_bytes must be in [1024, 16777216]")


@dataclass(frozen=True, slots=True)
class ReadOnlyMount:
    """A trusted evaluator input mounted below ``/evaluation``."""

    source: Path
    target: str


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Bounded result returned by a command runner."""

    returncode: int
    stdout: bytes = b""
    stderr: bytes = b""
    timed_out: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False


class CommandRunner(Protocol):
    """Injectable process boundary used by offline tests."""

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> CommandResult:
        """Run argv without a shell and return bounded byte output."""


class _BoundedBuffer:
    """Keep useful head/tail output while continuously draining a pipe."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._head_limit = limit // 2
        self._tail_limit = limit - self._head_limit
        self._head = bytearray()
        self._tail = bytearray()
        self._seen = 0

    def append(self, chunk: bytes) -> None:
        if not chunk:
            return
        self._seen += len(chunk)
        head_remaining = self._head_limit - len(self._head)
        if head_remaining > 0:
            taken = min(head_remaining, len(chunk))
            self._head.extend(chunk[:taken])
            chunk = chunk[taken:]
        if chunk and self._tail_limit:
            self._tail.extend(chunk)
            if len(self._tail) > self._tail_limit:
                del self._tail[: len(self._tail) - self._tail_limit]

    @property
    def truncated(self) -> bool:
        return self._seen > self._limit

    def render(self) -> bytes:
        if not self.truncated:
            return bytes(self._head + self._tail)
        data_budget = max(0, self._limit - len(_TRUNCATION_MARKER))
        head_budget = data_budget // 2
        tail_budget = data_budget - head_budget
        return (
            bytes(self._head[:head_budget])
            + _TRUNCATION_MARKER[: self._limit]
            + bytes(self._tail[-tail_budget:] if tail_budget else b"")
        )[: self._limit]


class SubprocessCommandRunner:
    """Run commands using bounded, concurrently drained stdout/stderr pipes."""

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> CommandResult:
        # shell=False is intentional and should remain explicit at this most
        # security-sensitive process boundary.
        process = subprocess.Popen(
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
        )
        assert process.stdout is not None
        assert process.stderr is not None

        stdout_buffer = _BoundedBuffer(max_output_bytes)
        stderr_buffer = _BoundedBuffer(max_output_bytes)

        def drain(stream: object, output: _BoundedBuffer) -> None:
            try:
                while True:
                    chunk = stream.read(8192)  # type: ignore[attr-defined]
                    if not chunk:
                        return
                    output.append(chunk)
            except OSError:
                # Killing a process can close a pipe while a reader is active.
                return

        threads = [
            threading.Thread(target=drain, args=(process.stdout, stdout_buffer), daemon=True),
            threading.Thread(target=drain, args=(process.stderr, stderr_buffer), daemon=True),
        ]
        for thread in threads:
            thread.start()

        timed_out = False
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait()
        finally:
            for thread in threads:
                thread.join(timeout=2.0)
            process.stdout.close()
            process.stderr.close()

        return CommandResult(
            returncode=process.returncode,
            stdout=stdout_buffer.render(),
            stderr=stderr_buffer.render(),
            timed_out=timed_out,
            stdout_truncated=stdout_buffer.truncated,
            stderr_truncated=stderr_buffer.truncated,
        )


@dataclass(frozen=True, slots=True)
class SandboxTestResult:
    """Structured result from an approved pytest container."""

    command: tuple[str, ...]
    container_name: str
    image: str
    image_id: str | None
    exit_code: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool
    stdout_truncated: bool
    stderr_truncated: bool

    @property
    def passed(self) -> bool:
        return not self.timed_out and self.exit_code == 0

    @property
    def infrastructure_error(self) -> bool:
        # Docker reserves these statuses for invocation/runtime failures.
        return not self.timed_out and self.exit_code in {125, 126, 127}

    def to_dict(self) -> dict[str, object]:
        """Return the complete model/report-facing representation."""

        return {
            "command": list(self.command),
            "container_name": self.container_name,
            "image": self.image,
            "image_id": self.image_id,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": self.duration_seconds,
            "timed_out": self.timed_out,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
            "passed": self.passed,
            "infrastructure_error": self.infrastructure_error,
        }


class DockerSandbox:
    """Execute one pre-approved pytest argv in a hardened Docker container."""

    def __init__(
        self,
        *,
        image: str = DEFAULT_SANDBOX_IMAGE,
        limits: SandboxLimits | None = None,
        runner: CommandRunner | None = None,
        docker_binary: str = "docker",
        container_user: str = "10001:10001",
    ) -> None:
        if not _IMAGE_PATTERN.fullmatch(image) or image.startswith("-"):
            raise ValueError("invalid Docker image reference")
        if not docker_binary or "\x00" in docker_binary:
            raise ValueError("invalid Docker executable")
        if not re.fullmatch(r"[0-9]+:[0-9]+", container_user):
            raise ValueError("container_user must be a numeric uid:gid")
        self._image = image
        self._limits = limits or SandboxLimits()
        self._runner = runner or SubprocessCommandRunner()
        self._docker_binary = docker_binary
        self._container_user = container_user

    @property
    def image(self) -> str:
        return self._image

    def run_tests(
        self,
        workspace: Path | str,
        command: ApprovedPytestCommand,
        *,
        extra_mounts: Sequence[ReadOnlyMount] = (),
    ) -> SandboxTestResult:
        """Run approved tests, returning failures instead of running on the host.

        ``extra_mounts`` is for trusted, independent evaluator inputs.  Its
        targets are restricted to ``/evaluation`` and are always read-only.
        Repository/model data must never be allowed to construct these mounts.
        """

        if not isinstance(command, ApprovedPytestCommand):
            raise TypeError("run_tests requires an ApprovedPytestCommand created by trusted setup")
        workspace_path = self._validate_source_path(workspace, "workspace")
        if not workspace_path.is_dir():
            raise InvalidMountError("workspace must be a directory")
        mounts = self._validate_extra_mounts(extra_mounts)

        self._check_docker()
        image_id = self._check_image()

        container_name = f"coderepair-{uuid.uuid4().hex}"
        docker_argv = self._docker_argv(workspace_path, command.argv, mounts, container_name)
        started_at = time.monotonic()
        try:
            execution = self._runner.run(
                docker_argv,
                timeout_seconds=self._limits.timeout_seconds,
                max_output_bytes=self._limits.max_output_bytes,
            )
        except FileNotFoundError as exc:
            # A race can remove Docker after the preflight check.  It is still
            # an infrastructure failure and must not trigger a host fallback.
            raise DockerUnavailableError("Docker CLI is not available") from exc
        duration = time.monotonic() - started_at

        if execution.timed_out:
            self._cleanup_timed_out_container(container_name)

        return SandboxTestResult(
            command=command.argv,
            container_name=container_name,
            image=self._image,
            image_id=image_id,
            exit_code=None if execution.timed_out else execution.returncode,
            stdout=execution.stdout.decode("utf-8", errors="replace"),
            stderr=execution.stderr.decode("utf-8", errors="replace"),
            duration_seconds=duration,
            timed_out=execution.timed_out,
            stdout_truncated=execution.stdout_truncated,
            stderr_truncated=execution.stderr_truncated,
        )

    def _check_docker(self) -> None:
        try:
            result = self._runner.run(
                [self._docker_binary, "version", "--format", "{{.Server.Version}}"],
                timeout_seconds=_DOCKER_CHECK_TIMEOUT_SECONDS,
                max_output_bytes=4096,
            )
        except (FileNotFoundError, OSError) as exc:
            raise DockerUnavailableError("Docker CLI is not available") from exc
        if result.timed_out:
            raise DockerUnavailableError("timed out while contacting the Docker daemon")
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            suffix = f": {detail}" if detail else ""
            raise DockerUnavailableError(f"Docker daemon is not available{suffix}")

    def _check_image(self) -> str | None:
        try:
            result = self._runner.run(
                [
                    self._docker_binary,
                    "image",
                    "inspect",
                    "--format",
                    "{{.Id}}",
                    self._image,
                ],
                timeout_seconds=_DOCKER_CHECK_TIMEOUT_SECONDS,
                max_output_bytes=4096,
            )
        except (FileNotFoundError, OSError) as exc:
            raise DockerUnavailableError("Docker CLI is not available") from exc
        if result.timed_out:
            raise DockerUnavailableError("timed out while inspecting the sandbox image")
        if result.returncode != 0:
            raise SandboxImageUnavailableError(
                f"sandbox image {self._image!r} is not available locally; "
                "build it before running tests"
            )
        inspected_id = result.stdout.decode("utf-8", errors="replace").strip()
        return inspected_id or None

    def _cleanup_timed_out_container(self, container_name: str) -> None:
        # `docker run --rm` normally removes a killed container.  The explicit
        # rm handles races and daemon/client disconnects; both commands target
        # the unguessable name generated for this invocation only.
        for arguments in (
            [self._docker_binary, "kill", container_name],
            [self._docker_binary, "rm", "--force", container_name],
        ):
            # Preserve the primary timeout result. A daemon disappearing during
            # cleanup can be diagnosed independently from that result.
            with suppress(FileNotFoundError, OSError):
                self._runner.run(
                    arguments,
                    timeout_seconds=_CLEANUP_TIMEOUT_SECONDS,
                    max_output_bytes=4096,
                )

    def _docker_argv(
        self,
        workspace: Path,
        pytest_argv: tuple[str, ...],
        mounts: tuple[tuple[Path, str], ...],
        container_name: str,
    ) -> list[str]:
        limits = self._limits
        memory = f"{limits.memory_mb}m"
        arguments = [
            self._docker_binary,
            "run",
            "--rm",
            "--pull=never",
            "--name",
            container_name,
            "--network=none",
            f"--user={self._container_user}",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true",
            "--read-only",
            f"--pids-limit={limits.pids_limit}",
            f"--memory={memory}",
            f"--memory-swap={memory}",
            f"--cpus={limits.cpus:g}",
            f"--ulimit=nofile={limits.nofile_limit}:{limits.nofile_limit}",
            "--stop-timeout=1",
            "--log-driver=none",
            "--tmpfs",
            f"/tmp:rw,nosuid,nodev,noexec,size={limits.tmpfs_mb}m",
            "--mount",
            self._mount_argument(workspace, "/workspace"),
            "--workdir=/workspace",
            "--env=HOME=/tmp",
            "--env=TMPDIR=/tmp",
            "--env=PYTHONDONTWRITEBYTECODE=1",
            "--env=PYTHONUNBUFFERED=1",
            "--env=PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
            "--env=PYTHONPATH=/workspace/src:/workspace",
        ]
        for source, target in mounts:
            arguments.extend(["--mount", self._mount_argument(source, target)])
        arguments.append(self._image)
        arguments.extend(pytest_argv)
        return arguments

    @staticmethod
    def _mount_argument(source: Path, target: str) -> str:
        # Docker's --mount grammar uses commas as separators and offers no
        # portable escaping for a comma in a bind source.  Failing clearly is
        # safer than mounting a different path than the caller inspected.
        source_text = str(source)
        if "," in source_text or "\x00" in source_text:
            raise InvalidMountError("mount source path contains an unsupported character")
        return f"type=bind,source={source_text},target={target},readonly"

    @staticmethod
    def _validate_source_path(source: Path | str, label: str) -> Path:
        try:
            path = Path(source).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise InvalidMountError(f"{label} does not exist or cannot be resolved") from exc
        if not (path.is_dir() or path.is_file()):
            raise InvalidMountError(f"{label} must be a regular file or directory")
        if "," in str(path) or "\x00" in str(path):
            raise InvalidMountError(
                f"{label} path contains a character unsupported by Docker mounts"
            )
        return path

    def _validate_extra_mounts(
        self, mounts: Sequence[ReadOnlyMount]
    ) -> tuple[tuple[Path, str], ...]:
        validated: list[tuple[Path, str]] = []
        targets: set[str] = set()
        for mount in mounts:
            if not isinstance(mount, ReadOnlyMount):
                raise InvalidMountError("extra mounts must be ReadOnlyMount values")
            source = self._validate_source_path(mount.source, "extra mount source")
            target = PurePosixPath(mount.target)
            if (
                not target.is_absolute()
                or ".." in target.parts
                or any(character in mount.target for character in (",", "\x00", "\n", "\r"))
                or (
                    str(target) not in {"/evaluation"}
                    and not str(target).startswith("/evaluation/")
                )
            ):
                raise InvalidMountError("extra mount targets must be /evaluation or below it")
            target_text = str(target)
            if target_text in targets:
                raise InvalidMountError("extra mount targets must be unique")
            targets.add(target_text)
            validated.append((source, target_text))
        return tuple(validated)


__all__ = [
    "DEFAULT_SANDBOX_IMAGE",
    "ApprovedPytestCommand",
    "CommandResult",
    "CommandRunner",
    "DockerSandbox",
    "DockerUnavailableError",
    "InvalidMountError",
    "InvalidTestCommandError",
    "ReadOnlyMount",
    "SandboxError",
    "SandboxImageUnavailableError",
    "SandboxLimits",
    "SandboxTestResult",
    "SubprocessCommandRunner",
    "approve_pytest_command",
]
