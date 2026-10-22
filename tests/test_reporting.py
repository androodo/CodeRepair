from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from coderepair.agent import AgentOutcome, RunStatus
from coderepair.model import ModelUsage
from coderepair.reporting import TRACE_EVENT_LIMIT, Pricing, TraceWriter, write_run_artifacts


def outcome(*, visible_passed: bool = True) -> AgentOutcome:
    return AgentOutcome(
        status=(
            RunStatus.VISIBLE_TESTS_PASSED if visible_passed else RunStatus.COMPLETED_UNVERIFIED
        ),
        final_text="Fixed the boundary condition.",
        iterations=3,
        model_requests=3,
        tool_calls=7,
        usage=ModelUsage(100, 50, 10, 20),
        duration_seconds=1.25,
        visible_tests_passed=visible_passed,
        messages=[],
        stop_reason="end_turn",
    )


def test_pricing_requires_an_explicit_complete_configuration() -> None:
    assert Pricing.from_strings(None, None) is None
    with pytest.raises(ValueError, match="both"):
        Pricing.from_strings("1", None)
    with pytest.raises(ValueError, match="decimal"):
        Pricing.from_strings("not-a-number", "2")
    with pytest.raises(ValueError, match="negative"):
        Pricing.from_strings("-1", "2")


def test_pricing_estimate_uses_reported_input_and_output_tokens() -> None:
    pricing = Pricing.from_strings("2", "4")
    assert pricing is not None

    estimate = pricing.estimate(ModelUsage(100, 50, 10, 20))

    assert estimate == Decimal("0.00046")


def test_report_distinguishes_visible_independent_and_verified_success(
    tmp_path: Path,
) -> None:
    pricing = Pricing.from_strings("2", "4")
    result = write_run_artifacts(
        tmp_path,
        task="Fix the bug",
        outcome=outcome(),
        original_repository=Path("/source/repo"),
        original_commit="a" * 40,
        model="claude-test",
        diff="diff --git a/app.py b/app.py\n",
        modified_files=("app.py",),
        workspace_generation=2,
        last_tested_generation=2,
        last_test_result={"exit_code": 0, "passed": True},
        pricing=pricing,
        independent_evaluation={
            "status": "passed",
            "passed": True,
            "regression_passed": True,
        },
    )

    verification = result["verification"]
    assert verification["visible_tests"]["status"] == "passed"
    assert verification["independent_evaluation"]["status"] == "passed"
    assert verification["verified_task_success"] is True
    assert result["estimated_cost_usd"] == "0.000460"
    assert result["pricing"] == {
        "input_usd_per_million": "2",
        "output_usd_per_million": "4",
        "source": "user-configured",
    }
    assert json.loads((tmp_path / "result.json").read_text(encoding="utf-8")) == result
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "Visible tests:** `passed`" in summary
    assert "Independent evaluation:** `passed`" in summary
    assert "Verified task success:** `true`" in summary
    assert "A model statement alone is never treated as verification" in summary


def test_outdated_visible_result_cannot_verify_task_and_absent_pricing_is_null(
    tmp_path: Path,
) -> None:
    result = write_run_artifacts(
        tmp_path,
        task="Fix it",
        outcome=outcome(),
        original_repository=tmp_path,
        original_commit="b" * 40,
        model="claude-test",
        diff="",
        modified_files=(),
        workspace_generation=3,
        last_tested_generation=2,
        last_test_result={"exit_code": 0, "passed": True},
        independent_evaluation={"status": "passed", "passed": True},
    )

    visible = result["verification"]["visible_tests"]
    assert visible["status"] == "outdated"
    assert visible["tested_generation"] == 2
    assert visible["current_generation"] == 3
    assert result["verification"]["verified_task_success"] is False
    assert result["estimated_cost_usd"] is None
    assert result["pricing"] is None


def test_no_test_run_is_not_confused_with_a_failed_test(tmp_path: Path) -> None:
    result = write_run_artifacts(
        tmp_path,
        task="Inspect only",
        outcome=outcome(visible_passed=False),
        original_repository=tmp_path,
        original_commit="c" * 40,
        model="claude-test",
        diff="",
        modified_files=(),
        workspace_generation=0,
        last_tested_generation=-1,
        last_test_result=None,
    )

    assert result["verification"]["visible_tests"] == {"status": "not_run"}
    assert result["verification"]["independent_evaluation"] == {"status": "not_run"}
    assert result["verification"]["verified_task_success"] is False


def test_independent_result_without_regression_evidence_is_not_verified(
    tmp_path: Path,
) -> None:
    result = write_run_artifacts(
        tmp_path,
        task="Fix it",
        outcome=outcome(),
        original_repository=tmp_path,
        original_commit="d" * 40,
        model="claude-test",
        diff="diff --git a/app.py b/app.py\n",
        modified_files=("app.py",),
        workspace_generation=1,
        last_tested_generation=1,
        last_test_result={"exit_code": 0, "passed": True},
        independent_evaluation={"status": "passed", "passed": True},
    )

    assert result["verification"]["visible_tests"]["status"] == "passed"
    assert result["verification"]["verified_task_success"] is False


def test_non_python_only_change_does_not_claim_python_syntax_validation(
    tmp_path: Path,
) -> None:
    result = write_run_artifacts(
        tmp_path,
        task="Update configuration",
        outcome=outcome(visible_passed=False),
        original_repository=tmp_path,
        original_commit="e" * 40,
        model="claude-test",
        diff="diff --git a/config.toml b/config.toml\n",
        modified_files=("config.toml",),
        workspace_generation=1,
        last_tested_generation=-1,
        last_test_result=None,
    )

    assert result["verification"]["python_syntax"] == "no_modified_python_files"


def test_trace_writer_sequences_events_and_bounds_oversized_payloads(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    writer = TraceWriter(path)
    writer("small", {"value": 1})
    writer("large", {"payload": "x" * TRACE_EVENT_LIMIT})

    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [event["sequence"] for event in events] == [1, 2]
    assert events[0]["data"] == {"value": 1}
    assert events[1]["data"]["truncated"] is True
    assert "exceeded" in events[1]["data"]["reason"]
