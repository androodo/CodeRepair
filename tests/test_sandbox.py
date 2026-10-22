from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from coderepair.sandbox import (
    CommandResult,
    DockerSandbox,
    DockerUnavailableError,
    InvalidMountError,
    InvalidTestCommandError,
    ReadOnlyMount,
    SandboxImageUnavailableError,
    SandboxLimits,
    SubprocessCommandRunner,
    approve_pytest_command,
)


class RecordingRunner:
    def __init__(
        self,
        *,
        docker_available: bool = True,
        image_available: bool = True,
        execution: CommandResult | None = None,
        missing_executable: bool = False,
    ) -> None:
        self.docker_available = docker_available
        self.image_available = image_available
        self.execution = execution or CommandResult(0, b"1 passed\n", b"")
        self.missing_executable = missing_executable
        self.calls: list[tuple[tuple[str, ...], float, int]] = []

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> CommandResult:
        call = tuple(argv)
        self.calls.append((call, timeout_seconds, max_output_bytes))
        if self.missing_executable:
            raise FileNotFoundError(call[0])
        if len(call) >= 2 and call[1] == "version":
            return CommandResult(0 if self.docker_available else 1, b"", b"daemon down")
        if len(call) >= 3 and call[1:3] == ("image", "inspect"):
            return CommandResult(0 if self.image_available else 1)
        if len(call) >= 2 and call[1] == "run":
            return self.execution
        return CommandResult(0)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("pytest -q tests/unit", ("python", "-m", "pytest", "-q", "tests/unit")),
        (
            'python -m pytest -k "login and not expired"',
            ("python", "-m", "pytest", "-k", "login and not expired"),
        ),
        (
            ["python3.11", "-m", "pytest", "tests/test api.py"],
            ("python", "-m", "pytest", "tests/test api.py"),
        ),
    ],
)
def test_approved_pytest_command_is_normalized(
    raw: str | list[str], expected: tuple[str, ...]
) -> None:
    assert approve_pytest_command(raw).argv == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "python script.py",
        "bash -c pytest",
        "pytest -q; touch /tmp/pwned",
        ["pytest", "$(touch /tmp/pwned)"],
        ["pytest", "line\nbreak"],
    ],
)
def test_approved_pytest_command_rejects_non_pytest_or_shell_syntax(
    raw: str | list[str],
) -> None:
    with pytest.raises(InvalidTestCommandError):
        approve_pytest_command(raw)


def test_container_invocation_has_hardening_and_read_only_mounts(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace with spaces"
    hidden = tmp_path / "hidden"
    workspace.mkdir()
    hidden.mkdir()
    runner = RecordingRunner()
    sandbox = DockerSandbox(runner=runner)

    result = sandbox.run_tests(
        workspace,
        approve_pytest_command("pytest -q tests"),
        extra_mounts=(ReadOnlyMount(hidden, "/evaluation/tests"),),
    )

    assert result.passed
    result_data = result.to_dict()
    assert result_data["passed"] is True
    assert result_data["infrastructure_error"] is False
    assert result_data["image"] == "coderepair-sandbox:py311"
    run_argv = next(call[0] for call in runner.calls if call[0][1] == "run")
    required_options = {
        "--pull=never",
        "--network=none",
        "--user=10001:10001",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges:true",
        "--read-only",
        "--pids-limit=128",
        "--memory=512m",
        "--memory-swap=512m",
        "--cpus=1",
        "--ulimit=nofile=256:256",
        "--log-driver=none",
        "--workdir=/workspace",
    }
    assert required_options.issubset(run_argv)
    assert not any("docker.sock" in argument for argument in run_argv)
    assert not any(str(Path.home()) == argument for argument in run_argv)

    mount_values = [
        run_argv[index + 1] for index, argument in enumerate(run_argv) if argument == "--mount"
    ]
    assert f"type=bind,source={workspace.resolve()},target=/workspace,readonly" in mount_values
    assert f"type=bind,source={hidden.resolve()},target=/evaluation/tests,readonly" in mount_values
    image_index = run_argv.index("coderepair-sandbox:py311")
    assert run_argv[image_index + 1 :] == (
        "python",
        "-m",
        "pytest",
        "-q",
        "tests",
    )


def test_model_facing_call_requires_preapproved_command(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner = RecordingRunner()

    with pytest.raises(TypeError, match="ApprovedPytestCommand"):
        DockerSandbox(runner=runner).run_tests(  # type: ignore[arg-type]
            workspace, ["pytest", "-q"]
        )

    assert runner.calls == []


def test_unavailable_docker_never_attempts_test_or_host_fallback(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner = RecordingRunner(docker_available=False)

    with pytest.raises(DockerUnavailableError, match="daemon"):
        DockerSandbox(runner=runner).run_tests(workspace, approve_pytest_command("pytest"))

    assert len(runner.calls) == 1
    assert runner.calls[0][0][1] == "version"


def test_missing_docker_executable_is_a_clear_infrastructure_error(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner = RecordingRunner(missing_executable=True)

    with pytest.raises(DockerUnavailableError, match="CLI"):
        DockerSandbox(runner=runner).run_tests(workspace, approve_pytest_command("pytest"))


def test_missing_image_does_not_pull_or_start_a_container(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner = RecordingRunner(image_available=False)

    with pytest.raises(SandboxImageUnavailableError, match="build it"):
        DockerSandbox(runner=runner).run_tests(workspace, approve_pytest_command("pytest"))

    assert [call[0][1] for call in runner.calls] == ["version", "image"]


def test_timeout_kills_and_removes_only_the_named_container(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner = RecordingRunner(
        execution=CommandResult(
            returncode=-9,
            stdout=b"partial output",
            stderr=b"",
            timed_out=True,
        )
    )

    result = DockerSandbox(runner=runner).run_tests(workspace, approve_pytest_command("pytest -q"))

    assert result.timed_out
    assert result.exit_code is None
    assert result.stdout == "partial output"
    run_argv = runner.calls[2][0]
    name = run_argv[run_argv.index("--name") + 1]
    assert name == result.container_name
    assert runner.calls[3][0] == ("docker", "kill", name)
    assert runner.calls[4][0] == ("docker", "rm", "--force", name)


@pytest.mark.parametrize(
    "target",
    [
        "evaluation/tests",
        "/workspace/hidden",
        "/tmp/hidden",
        "/evaluation/../tmp",
        "/evaluation/tests,readonly=false",
    ],
)
def test_extra_mount_target_must_stay_below_evaluation(tmp_path: Path, target: str) -> None:
    workspace = tmp_path / "workspace"
    hidden = tmp_path / "hidden"
    workspace.mkdir()
    hidden.mkdir()

    with pytest.raises(InvalidMountError, match="/evaluation"):
        DockerSandbox(runner=RecordingRunner()).run_tests(
            workspace,
            approve_pytest_command("pytest"),
            extra_mounts=(ReadOnlyMount(hidden, target),),
        )


def test_mount_source_with_comma_fails_before_docker(tmp_path: Path) -> None:
    workspace = tmp_path / "work,space"
    workspace.mkdir()
    runner = RecordingRunner()

    with pytest.raises(InvalidMountError, match="unsupported"):
        DockerSandbox(runner=runner).run_tests(workspace, approve_pytest_command("pytest"))

    assert runner.calls == []


def test_non_utf8_output_is_decoded_without_losing_the_result(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner = RecordingRunner(
        execution=CommandResult(
            1,
            stdout=b"failure: \xff",
            stderr=b"trace: \xfe",
            stdout_truncated=True,
        )
    )

    result = DockerSandbox(runner=runner).run_tests(workspace, approve_pytest_command("pytest"))

    assert not result.passed
    assert result.exit_code == 1
    assert "\ufffd" in result.stdout
    assert "\ufffd" in result.stderr
    assert result.stdout_truncated


def test_subprocess_runner_drains_but_bounds_large_output() -> None:
    runner = SubprocessCommandRunner()
    result = runner.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.buffer.write(b'a' * 20000); "
            "sys.stderr.buffer.write(b'b' * 20000)",
        ],
        timeout_seconds=5,
        max_output_bytes=1024,
    )

    assert result.returncode == 0
    assert result.stdout_truncated
    assert result.stderr_truncated
    assert len(result.stdout) <= 1024
    assert len(result.stderr) <= 1024
    assert b"output truncated" in result.stdout
    assert b"output truncated" in result.stderr


def test_sandbox_limit_validation_rejects_unbounded_values() -> None:
    with pytest.raises(ValueError, match="timeout"):
        SandboxLimits(timeout_seconds=0)
    with pytest.raises(ValueError, match="max_output"):
        SandboxLimits(max_output_bytes=100_000_000)
