from __future__ import annotations

import json
from typing import Any

import pytest

from coderepair.agent import Agent, AgentLimits, RunStatus, ToolExecution
from coderepair.model import FakeModel, ModelError, ModelResponse, ModelUsage


class RecordingTools:
    definitions = ({"name": "inspect", "input_schema": {"type": "object"}},)

    def __init__(self, *, passing: bool = False, raise_on: str | None = None) -> None:
        self.passing = passing
        self.raise_on = raise_on
        self.calls: list[tuple[str, Any]] = []

    @property
    def visible_tests_currently_passing(self) -> bool:
        return self.passing

    def execute(self, name: str, arguments: Any) -> ToolExecution:
        self.calls.append((name, arguments))
        if name == self.raise_on:
            raise RuntimeError("dispatcher exploded")
        return ToolExecution({"name": name, "arguments": arguments})


def tool_response(*blocks: dict[str, Any], usage: ModelUsage | None = None) -> ModelResponse:
    return ModelResponse(list(blocks), "tool_use", usage or ModelUsage())


def end_response(text: str = "done", *, usage: ModelUsage | None = None) -> ModelResponse:
    return ModelResponse([{"type": "text", "text": text}], "end_turn", usage or ModelUsage())


def test_replays_complete_assistant_content_and_orders_parallel_results() -> None:
    thinking = {
        "type": "thinking",
        "thinking": "I should inspect both files.",
        "signature": "signed-thinking",
    }
    first_content = [
        thinking,
        {"type": "text", "text": "Inspecting."},
        {"type": "tool_use", "id": "one", "name": "first", "input": {"n": 1}},
        {"type": "tool_use", "id": "two", "name": "second", "input": {"n": 2}},
    ]
    model = FakeModel(
        [ModelResponse(first_content, "tool_use", ModelUsage(5, 3)), end_response("fixed")]
    )
    tools = RecordingTools()

    outcome = Agent(model).run(task="repair", tools=tools)

    assert outcome.status is RunStatus.COMPLETED_UNVERIFIED
    assert outcome.final_text == "fixed"
    assert outcome.tool_calls == 2
    assert tools.calls == [("first", {"n": 1}), ("second", {"n": 2})]
    second_request_messages = model.calls[1]["messages"]
    assert second_request_messages[1] == {
        "role": "assistant",
        "content": first_content,
    }
    assert second_request_messages[1]["content"][0] == thinking
    result_turn = second_request_messages[2]
    assert result_turn["role"] == "user"
    assert [block["tool_use_id"] for block in result_turn["content"]] == ["one", "two"]
    assert [json.loads(block["content"])["name"] for block in result_turn["content"]] == [
        "first",
        "second",
    ]


def test_dispatcher_exception_becomes_an_error_tool_observation() -> None:
    model = FakeModel(
        [
            tool_response({"type": "tool_use", "id": "bad", "name": "explode", "input": {}}),
            end_response(),
        ]
    )

    Agent(model).run(task="repair", tools=RecordingTools(raise_on="explode"))

    result = model.calls[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True
    observation = json.loads(result["content"])
    assert "dispatcher failed (RuntimeError)" in observation["error"]


@pytest.mark.parametrize(
    ("response", "passing", "expected_status"),
    [
        (end_response(), False, RunStatus.COMPLETED_UNVERIFIED),
        (end_response(), True, RunStatus.VISIBLE_TESTS_PASSED),
        (
            ModelResponse([{"type": "text", "text": "cut off"}], "max_tokens"),
            False,
            RunStatus.LIMIT_REACHED,
        ),
        (
            ModelResponse([{"type": "text", "text": "paused"}], "pause_turn"),
            False,
            RunStatus.INCOMPLETE,
        ),
    ],
)
def test_stop_reason_and_visible_test_status(
    response: ModelResponse, passing: bool, expected_status: RunStatus
) -> None:
    outcome = Agent(FakeModel([response])).run(task="repair", tools=RecordingTools(passing=passing))

    assert outcome.status is expected_status


def test_tool_use_stop_without_a_tool_block_is_incomplete() -> None:
    outcome = Agent(FakeModel([ModelResponse([], "tool_use")])).run(
        task="repair", tools=RecordingTools()
    )

    assert outcome.status is RunStatus.INCOMPLETE
    assert "without a tool_use block" in (outcome.error or "")


def test_iteration_budget_stops_before_another_model_request() -> None:
    model = FakeModel(
        [tool_response({"type": "tool_use", "id": "one", "name": "inspect", "input": {}})]
    )
    tools = RecordingTools()

    outcome = Agent(model, limits=AgentLimits(max_iterations=1)).run(task="repair", tools=tools)

    assert outcome.status is RunStatus.LIMIT_REACHED
    assert outcome.error == "iteration budget exhausted"
    assert outcome.model_requests == 1
    assert tools.calls == [("inspect", {})]


def test_reported_token_overrun_stops_before_tool_execution() -> None:
    response = tool_response(
        {"type": "tool_use", "id": "one", "name": "inspect", "input": {}},
        usage=ModelUsage(input_tokens=8, output_tokens=3),
    )
    tools = RecordingTools()

    outcome = Agent(
        FakeModel([response]), limits=AgentLimits(max_total_tokens=10, max_output_tokens=50)
    ).run(task="repair", tools=tools)

    assert outcome.status is RunStatus.LIMIT_REACHED
    assert outcome.error == "model-reported token budget exceeded"
    assert outcome.usage.total_tokens == 11
    assert tools.calls == []


def test_tool_budget_returns_ordered_error_for_unexecuted_parallel_calls() -> None:
    model = FakeModel(
        [
            tool_response(
                {"type": "tool_use", "id": "one", "name": "first", "input": {}},
                {"type": "tool_use", "id": "two", "name": "second", "input": {}},
            )
        ]
    )
    tools = RecordingTools()

    outcome = Agent(model, limits=AgentLimits(max_tool_calls=1)).run(task="repair", tools=tools)

    assert outcome.status is RunStatus.LIMIT_REACHED
    assert outcome.error == "tool-call budget exhausted"
    assert tools.calls == [("first", {})]
    results = outcome.messages[-1]["content"]
    assert [result["tool_use_id"] for result in results] == ["one", "two"]
    assert "is_error" not in results[0]
    assert results[1]["is_error"] is True


def test_transient_model_error_retries_with_bounded_backoff() -> None:
    class FlakyModel:
        def __init__(self) -> None:
            self.calls = 0

        def create_message(self, **kwargs: Any) -> ModelResponse:
            del kwargs
            self.calls += 1
            if self.calls == 1:
                raise ModelError("try later", retriable=True)
            return end_response()

    sleeps: list[float] = []
    model = FlakyModel()
    outcome = Agent(
        model,
        limits=AgentLimits(max_api_retries=1),
        sleeper=sleeps.append,
    ).run(task="repair", tools=RecordingTools())

    assert outcome.status is RunStatus.COMPLETED_UNVERIFIED
    assert outcome.model_requests == 2
    assert sleeps == [1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_iterations": 0},
        {"max_tool_calls": 0},
        {"max_total_tokens": 0},
        {"max_seconds": 0},
        {"max_output_tokens": 0},
        {"max_api_retries": -1},
    ],
)
def test_invalid_agent_limits_are_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        AgentLimits(**kwargs)
