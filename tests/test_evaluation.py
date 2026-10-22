from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from coderepair.evaluation import (
    EvaluationEngine,
    EvaluationMode,
    ManifestError,
    SourceVerificationError,
    aggregate_results,
    canonical_source_tree_sha256,
    load_manifest,
    render_markdown_summary,
)
from coderepair.model import FakeModel, ModelResponse, ModelUsage
from coderepair.sandbox import ApprovedPytestCommand, ReadOnlyMount
from coderepair.workspace import create_workspace


def _fixture_trees(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    hidden = tmp_path / "hidden"
    (source / "src").mkdir(parents=True)
    (source / "tests").mkdir()
    hidden.mkdir()
    (source / "src" / "calculator.py").write_text(
        "def add(left, right):\n    return left + right\n", encoding="utf-8"
    )
    (source / "tests" / "test_visible.py").write_text(
        "from src.calculator import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    (hidden / "test_target.py").write_text(
        "from src.calculator import add\n\ndef test_target():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    (hidden / "test_regression.py").write_text(
        "from src.calculator import add\n\ndef test_regression():\n    assert add(0, 0) == 0\n",
        encoding="utf-8",
    )
    return source, hidden


def _manifest_data(source: Path, hidden: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "tasks": [
            {
                "id": "addition",
                "task": "Ensure the calculator adds two integers correctly.",
                "source": {
                    "path": str(source),
                    "tree_sha256": canonical_source_tree_sha256(source),
                },
                "visible_test_argv": ["python", "-m", "pytest", "-q", "tests"],
                "hidden_evaluation": {
                    "path": str(hidden),
                    "target_test_argv": [
                        "python",
                        "-m",
                        "pytest",
                        "-q",
                        "/evaluation/test_target.py",
                    ],
                    "regression_test_argv": [
                        "python",
                        "-m",
                        "pytest",
                        "-q",
                        "/evaluation/test_regression.py",
                    ],
                },
                "limits": {
                    "max_iterations": 3,
                    "max_tool_calls": 5,
                    "max_total_tokens": 1000,
                    "max_seconds": 30,
                    "max_output_tokens": 200,
                    "max_api_retries": 0,
                },
            }
        ],
    }


def _write_manifest(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_manifest_is_versioned_strict_and_requires_exactly_one_source_pin(
    tmp_path: Path,
) -> None:
    source, hidden = _fixture_trees(tmp_path)
    data = _manifest_data(source, hidden)
    manifest = load_manifest(_write_manifest(tmp_path, data))

    assert manifest.schema_version == 1
    assert manifest.tasks[0].source.tree_sha256 == canonical_source_tree_sha256(source)
    assert manifest.tasks[0].source.commit is None
    assert manifest.tasks[0].visible_test_command.argv[:3] == ("python", "-m", "pytest")

    unknown = copy.deepcopy(data)
    unknown["surprise"] = True
    with pytest.raises(ManifestError, match="unknown fields"):
        load_manifest(_write_manifest(tmp_path, unknown))

    ambiguous = copy.deepcopy(data)
    ambiguous["tasks"][0]["source"]["commit"] = "a" * 40
    with pytest.raises(ManifestError, match="exactly one"):
        load_manifest(_write_manifest(tmp_path, ambiguous))

    future = copy.deepcopy(data)
    future["schema_version"] = 2
    with pytest.raises(ManifestError, match="schema_version must be 1"):
        load_manifest(_write_manifest(tmp_path, future))


def test_source_digest_mismatch_aborts_before_model_or_output(tmp_path: Path) -> None:
    source, hidden = _fixture_trees(tmp_path)
    data = _manifest_data(source, hidden)
    data["tasks"][0]["source"]["tree_sha256"] = "0" * 64
    manifest_path = _write_manifest(tmp_path, data)
    model_calls: list[tuple[str, EvaluationMode]] = []

    def model_factory(task_id: str, mode: EvaluationMode) -> FakeModel:
        model_calls.append((task_id, mode))
        return FakeModel([])

    output = tmp_path / "results"
    with pytest.raises(SourceVerificationError, match="digest mismatch"):
        EvaluationEngine(model_factory, sandbox=object()).run(manifest_path, output)

    assert model_calls == []
    assert not output.exists()


@dataclass(frozen=True)
class _SandboxResult:
    command: tuple[str, ...]
    passed: bool = True
    infrastructure_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": list(self.command),
            "passed": self.passed,
            "infrastructure_error": self.infrastructure_error,
            "exit_code": 0 if self.passed else 1,
        }


class _RecordingSandbox:
    def __init__(self, events: list[str], *, fail_hidden: bool = False) -> None:
        self.events = events
        self.fail_hidden = fail_hidden
        self.calls: list[dict[str, Any]] = []

    def run_tests(
        self,
        workspace: Path,
        command: ApprovedPytestCommand,
        *,
        extra_mounts: tuple[ReadOnlyMount, ...] = (),
    ) -> _SandboxResult:
        call = {
            "workspace": Path(workspace),
            "command": command.argv,
            "mounts": tuple(extra_mounts),
            "files": tuple(
                sorted(
                    path.relative_to(workspace).as_posix()
                    for path in Path(workspace).rglob("*")
                    if path.is_file() and ".git" not in path.parts
                )
            ),
        }
        self.calls.append(call)
        is_hidden = any(argument.startswith("/evaluation") for argument in command.argv)
        self.events.append("sandbox:hidden" if is_hidden else "sandbox:visible")
        if is_hidden and self.fail_hidden:
            raise RuntimeError("simulated evaluator outage")
        return _SandboxResult(command.argv)


def test_modes_use_fresh_workspaces_and_hidden_tests_only_after_agent_stop(
    tmp_path: Path,
) -> None:
    source, hidden = _fixture_trees(tmp_path)
    manifest_path = _write_manifest(tmp_path, _manifest_data(source, hidden))
    events: list[str] = []
    models: dict[EvaluationMode, FakeModel] = {}

    def model_factory(task_id: str, mode: EvaluationMode) -> FakeModel:
        assert task_id == "addition"

        def response(
            index: int,
            _system: str,
            _messages: object,
            _tools: object,
        ) -> ModelResponse:
            events.append(f"model:{mode.value}:{index}")
            if mode is EvaluationMode.ITERATIVE and index == 0:
                return ModelResponse(
                    content=[
                        {
                            "type": "tool_use",
                            "id": "visible-tests",
                            "name": "run_tests",
                            "input": {},
                        }
                    ],
                    stop_reason="tool_use",
                    usage=ModelUsage(input_tokens=10, output_tokens=5),
                )
            return ModelResponse(
                content=[{"type": "text", "text": "Finished."}],
                stop_reason="end_turn",
                usage=ModelUsage(input_tokens=7, output_tokens=2),
            )

        model = FakeModel(response)
        models[mode] = model
        return model

    workspace_roots: list[Path] = []

    def workspace_factory(repository: Path, *, temp_parent: Path | None = None):
        workspace = create_workspace(repository, temp_parent=temp_parent)
        workspace_roots.append(workspace.root)
        return workspace

    sandbox = _RecordingSandbox(events)
    output = tmp_path / "results"
    report = EvaluationEngine(
        model_factory,
        sandbox=sandbox,
        workspace_factory=workspace_factory,
    ).run(manifest_path, output)

    assert len(workspace_roots) == 2
    assert workspace_roots[0] != workspace_roots[1]
    baseline_tools = {tool["name"] for tool in models[EvaluationMode.BASELINE].calls[0]["tools"]}
    iterative_tools = {tool["name"] for tool in models[EvaluationMode.ITERATIVE].calls[0]["tools"]}
    assert "run_tests" not in baseline_tools
    assert "run_tests" in iterative_tools

    # Four post-run checks per mode plus the iterative agent's one visible-test call.
    # The extra check establishes that regression tests pass on the pristine source.
    assert len(sandbox.calls) == 9
    agent_calls = [call for call in sandbox.calls if call["workspace"] in workspace_roots]
    assert len(agent_calls) == 1
    assert agent_calls[0]["mounts"] == ()
    post_calls = [call for call in sandbox.calls if call["workspace"] not in workspace_roots]
    assert len(post_calls) == 8
    assert all("hidden" not in call["files"] for call in post_calls)
    hidden_calls = [call for call in sandbox.calls if call["mounts"]]
    assert len(hidden_calls) == 6
    assert all(call["mounts"][0].source == hidden.resolve() for call in hidden_calls)
    assert all(call["mounts"][0].target == "/evaluation" for call in hidden_calls)

    hidden_event_positions = [
        index for index, event in enumerate(events) if event == "sandbox:hidden"
    ]
    final_model_positions = [
        index for index, event in enumerate(events) if event.startswith("model:")
    ]
    assert min(hidden_event_positions) > final_model_positions[0]
    assert max(final_model_positions) < max(hidden_event_positions)

    serialized_model_calls = json.dumps({mode.value: model.calls for mode, model in models.items()})
    assert str(hidden.resolve()) not in serialized_model_calls
    assert "/evaluation" not in serialized_model_calls

    assert all(result.status == "completed" for result in report.results)
    assert all(result.verified_task_success is True for result in report.results)
    assert (output / "tasks" / "addition" / "baseline.json").is_file()
    assert (output / "tasks" / "addition" / "iterative.json").is_file()
    assert (output / "aggregate.json").is_file()
    assert (output / "summary.md").is_file()
    aggregate = json.loads((output / "aggregate.json").read_text(encoding="utf-8"))
    assert aggregate["modes"]["baseline"]["repair_success_rate"] == 1.0
    assert aggregate["modes"]["iterative"]["tool_calls"]["total"] == 1


def test_infrastructure_failures_are_not_counted_as_repairs(tmp_path: Path) -> None:
    source, hidden = _fixture_trees(tmp_path)
    manifest_path = _write_manifest(tmp_path, _manifest_data(source, hidden))

    def model_factory(_task_id: str, _mode: EvaluationMode) -> FakeModel:
        return FakeModel(
            [
                ModelResponse(
                    content=[{"type": "text", "text": "Done."}],
                    stop_reason="end_turn",
                )
            ]
        )

    sandbox = _RecordingSandbox([], fail_hidden=True)
    report = EvaluationEngine(model_factory, sandbox=sandbox).run(
        manifest_path,
        tmp_path / "infrastructure-results",
        modes=(EvaluationMode.BASELINE,),
    )

    result = report.results[0]
    metrics = report.aggregate["modes"]["baseline"]
    assert result.status == "infrastructure_error"
    assert result.verified_task_success is None
    assert result.visible_tests.passed is True
    assert result.hidden_target_tests.infrastructure_error is True
    assert result.regression_tests.infrastructure_error is True
    assert metrics["evaluable_runs"] == 0
    assert metrics["infrastructure_errors"] == 1
    assert metrics["repair_success_rate"] is None


def test_zero_task_aggregates_are_defined_without_fabricated_rates() -> None:
    aggregate = aggregate_results(
        [],
        task_count=0,
        modes=(EvaluationMode.BASELINE, EvaluationMode.ITERATIVE),
    )

    for mode in ("baseline", "iterative"):
        metrics = aggregate["modes"][mode]
        assert metrics["runs"] == 0
        assert metrics["repair_success_rate"] is None
        assert metrics["visible_tests"]["pass_rate"] is None
        assert metrics["latency_seconds"]["mean"] is None
    markdown = render_markdown_summary(aggregate)
    assert "N/A" in markdown
    assert "nan" not in markdown.lower()


def test_canonical_source_digest_covers_paths_content_and_mode(tmp_path: Path) -> None:
    source = tmp_path / "tree"
    source.mkdir()
    file = source / "module.py"
    file.write_text("VALUE = 1\n", encoding="utf-8")
    first = canonical_source_tree_sha256(source)
    assert canonical_source_tree_sha256(source) == first

    file.write_text("VALUE = 2\n", encoding="utf-8")
    assert canonical_source_tree_sha256(source) != first

    link = source / "link.py"
    try:
        os.symlink(file, link)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")
    with pytest.raises(Exception, match="links are not permitted"):
        canonical_source_tree_sha256(source)
