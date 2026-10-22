"""Durable JSONL traces and concise run artifacts."""

from __future__ import annotations

import json
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .agent import AgentOutcome
from .model import JsonObject, ModelUsage

TRACE_EVENT_LIMIT = 1_000_000


class TraceWriter:
    """Append and flush each event so partial runs remain inspectable."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self._sequence = 0
        self._lock = threading.Lock()

    def __call__(self, event_type: str, data: JsonObject) -> None:
        with self._lock:
            self._sequence += 1
            event = {
                "schema_version": 1,
                "sequence": self._sequence,
                "timestamp": datetime.now(UTC).isoformat(),
                "type": event_type,
                "data": data,
            }
            encoded = json.dumps(event, ensure_ascii=False, sort_keys=True, default=str)
            if len(encoded.encode("utf-8")) > TRACE_EVENT_LIMIT:
                event["data"] = {
                    "truncated": True,
                    "reason": f"event exceeded {TRACE_EVENT_LIMIT} encoded bytes",
                }
                encoded = json.dumps(event, ensure_ascii=False, sort_keys=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded + "\n")
                handle.flush()


@dataclass(frozen=True)
class Pricing:
    input_usd_per_million: Decimal
    output_usd_per_million: Decimal
    source: str = "user-configured"

    @classmethod
    def from_strings(cls, input_price: str | None, output_price: str | None) -> Pricing | None:
        if not input_price and not output_price:
            return None
        if not input_price or not output_price:
            raise ValueError("both input and output prices must be configured")
        try:
            parsed_input = Decimal(input_price)
            parsed_output = Decimal(output_price)
        except InvalidOperation as exc:
            raise ValueError("pricing values must be decimal numbers") from exc
        if parsed_input < 0 or parsed_output < 0:
            raise ValueError("pricing values cannot be negative")
        return cls(parsed_input, parsed_output)

    def estimate(self, usage: ModelUsage) -> Decimal:
        input_total = (
            usage.input_tokens
            + usage.cache_creation_input_tokens
            + usage.cache_read_input_tokens
        )
        return (
            Decimal(input_total) * self.input_usd_per_million
            + Decimal(usage.output_tokens) * self.output_usd_per_million
        ) / Decimal(1_000_000)


def default_output_directory(prefix: str = "run") -> Path:
    base = Path(tempfile.gettempdir()) / "coderepair-runs"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return base / f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"


def write_run_artifacts(
    output_directory: Path,
    *,
    task: str,
    outcome: AgentOutcome,
    original_repository: Path,
    original_commit: str,
    model: str,
    diff: str,
    modified_files: tuple[str, ...],
    workspace_generation: int,
    last_tested_generation: int,
    last_test_result: JsonObject | None,
    pricing: Pricing | None = None,
    independent_evaluation: JsonObject | None = None,
) -> JsonObject:
    output_directory.mkdir(parents=True, exist_ok=True)
    diff_path = output_directory / "diff.patch"
    diff_path.write_text(diff, encoding="utf-8", newline="\n")
    visible_state = _visible_test_state(
        workspace_generation,
        last_tested_generation,
        outcome.visible_tests_passed,
        last_test_result,
    )
    estimated_cost = pricing.estimate(outcome.usage) if pricing is not None else None
    verified = bool(
        independent_evaluation
        and independent_evaluation.get("passed") is True
        and visible_state["status"] == "passed"
        and independent_evaluation.get("regression_passed") is True
    )
    result: JsonObject = {
        "schema_version": 1,
        "task": task,
        "status": outcome.status.value,
        "agent_summary": outcome.final_text,
        "source": {
            "repository": str(original_repository),
            "commit": original_commit,
        },
        "model": model,
        "modified_files": list(modified_files),
        "metrics": outcome.metrics_dict(),
        "verification": {
            "python_syntax": (
                "passed_for_modified_python_files"
                if any(Path(name).suffix.casefold() == ".py" for name in modified_files)
                else "no_modified_python_files"
            ),
            "visible_tests": visible_state,
            "independent_evaluation": independent_evaluation or {"status": "not_run"},
            "verified_task_success": verified,
        },
        "estimated_cost_usd": (
            format(estimated_cost.quantize(Decimal("0.000001")), "f")
            if estimated_cost is not None
            else None
        ),
        "pricing": asdict(pricing) if pricing is not None else None,
        "artifacts": {
            "diff": "diff.patch",
            "trace": "trace.jsonl",
            "summary": "summary.md",
        },
    }
    # Decimal values need an explicit conversion for portable JSON.
    if result["pricing"] is not None:
        result["pricing"] = {
            key: str(value) if isinstance(value, Decimal) else value
            for key, value in result["pricing"].items()
        }
    (output_directory / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_directory / "summary.md").write_text(
        _render_summary(result), encoding="utf-8", newline="\n"
    )
    return result


def _visible_test_state(
    workspace_generation: int,
    last_tested_generation: int,
    currently_passing: bool,
    last_result: JsonObject | None,
) -> JsonObject:
    if last_result is None:
        return {"status": "not_run"}
    if last_tested_generation != workspace_generation:
        return {
            "status": "outdated",
            "tested_generation": last_tested_generation,
            "current_generation": workspace_generation,
            "last_result": last_result,
        }
    return {"status": "passed" if currently_passing else "failed", "result": last_result}


def _render_summary(result: JsonObject) -> str:
    metrics = result["metrics"]
    verification = result["verification"]
    visible = verification["visible_tests"]
    files = result["modified_files"]
    file_lines = "\n".join(f"- `{path}`" for path in files) if files else "- None"
    summary = result["agent_summary"] or "The model did not provide a final textual summary."
    cost = result["estimated_cost_usd"]
    return f"""# CodeRepair run

**Status:** `{result['status']}`  
**Source commit:** `{result['source']['commit']}`  
**Visible tests:** `{visible['status']}`  
**Independent evaluation:** `{verification['independent_evaluation'].get('status', 'completed')}`  
**Verified task success:** `{str(verification['verified_task_success']).lower()}`

## Agent summary

{summary}

## Modified files

{file_lines}

## Measurements

- Iterations: {metrics['iterations']}
- Model requests: {metrics['model_requests']}
- Tool calls: {metrics['tool_calls']}
- Duration: {metrics['duration_seconds']:.3f} seconds
- Input tokens: {metrics['usage']['input_tokens']}
- Output tokens: {metrics['usage']['output_tokens']}
- Estimated cost: {f'${cost}' if cost is not None else 'not configured'}

Syntax validity and visible-test success are reported separately from independent
evaluation. A model statement alone is never treated as verification.
"""
