from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from coderepair.model import (
    AnthropicMessagesModel,
    FakeModel,
    ModelError,
    ModelResponse,
    ModelUsage,
)


class FakeMessagesAPI:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, messages: FakeMessagesAPI) -> None:
        self.messages = messages


class ParamResponse:
    def __init__(self, blocks: list[dict[str, Any]]) -> None:
        self._blocks = blocks
        self.content: list[Any] = []
        self.stop_reason = "tool_use"
        self.model = "claude-test"
        self.id = "msg_123"
        self.usage = SimpleNamespace(
            input_tokens=11,
            output_tokens=7,
            cache_creation_input_tokens=3,
            cache_read_input_tokens=5,
        )

    def to_param(self) -> dict[str, Any]:
        return {"role": "assistant", "content": self._blocks}


def test_anthropic_adapter_preserves_complete_signed_content_blocks() -> None:
    blocks = [
        {"type": "thinking", "thinking": "inspect first", "signature": "signed-value"},
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "read_file",
            "input": {"path": "src/app.py"},
        },
    ]
    api = FakeMessagesAPI(ParamResponse(blocks))
    model = AnthropicMessagesModel("claude-test", client=FakeClient(api))

    response = model.create_message(
        system="system",
        messages=[{"role": "user", "content": "fix it"}],
        tools=[{"name": "read_file"}],
        max_tokens=321,
    )

    assert response.content == blocks
    assert response.content is not blocks
    assert response.content[0] is not blocks[0]
    assert response.usage == ModelUsage(11, 7, 3, 5)
    assert response.response_id == "msg_123"
    assert api.calls == [
        {
            "model": "claude-test",
            "max_tokens": 321,
            "system": "system",
            "messages": [{"role": "user", "content": "fix it"}],
            "tools": [{"name": "read_file"}],
        }
    ]

    blocks[0]["signature"] = "mutated"
    assert response.content[0]["signature"] == "signed-value"


def test_anthropic_adapter_falls_back_to_model_dump_for_content_blocks() -> None:
    class Block:
        def model_dump(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs == {"mode": "json", "exclude_none": True}
            return {"type": "text", "text": "done"}

    response = SimpleNamespace(
        content=[Block()],
        stop_reason="end_turn",
        model="claude-test",
        id="msg_fallback",
        usage=SimpleNamespace(input_tokens=-1, output_tokens="bad"),
    )
    model = AnthropicMessagesModel("claude-test", client=FakeClient(FakeMessagesAPI(response)))

    actual = model.create_message(system="s", messages=[], tools=[], max_tokens=10)

    assert actual.content == [{"type": "text", "text": "done"}]
    assert actual.usage == ModelUsage()


@pytest.mark.parametrize(
    ("status", "retriable"),
    [(400, False), (408, True), (409, True), (429, True), (500, True), (503, True)],
)
def test_anthropic_adapter_classifies_api_failures(status: int, retriable: bool) -> None:
    class APIError(Exception):
        def __init__(self) -> None:
            super().__init__("provider failed")
            self.status_code = status

    model = AnthropicMessagesModel(
        "claude-test", client=FakeClient(FakeMessagesAPI(error=APIError()))
    )

    with pytest.raises(ModelError, match="Anthropic request failed") as caught:
        model.create_message(system="s", messages=[], tools=[], max_tokens=1)

    assert caught.value.retriable is retriable


def test_fake_model_records_deep_copies_and_factory_index() -> None:
    seen_indexes: list[int] = []

    def factory(
        index: int,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        del system, messages, tools
        seen_indexes.append(index)
        return ModelResponse([{"type": "text", "text": str(index)}], "end_turn")

    model = FakeModel(factory)
    messages = [{"role": "user", "content": "original"}]
    tools = [{"name": "read_file"}]
    first = model.create_message(system="s", messages=messages, tools=tools, max_tokens=10)
    model.create_message(system="s", messages=messages, tools=tools, max_tokens=10)
    messages[0]["content"] = "changed"
    tools[0]["name"] = "changed"

    assert first.content[0]["text"] == "0"
    assert seen_indexes == [0, 1]
    assert model.calls[0]["messages"][0]["content"] == "original"
    assert model.calls[0]["tools"][0]["name"] == "read_file"


def test_fake_model_exhaustion_is_a_model_error() -> None:
    model = FakeModel([])

    with pytest.raises(ModelError, match="ran out"):
        model.create_message(system="s", messages=[], tools=[], max_tokens=1)


def test_model_usage_adds_all_reported_token_categories() -> None:
    usage = ModelUsage(10, 20, 30, 40) + ModelUsage(1, 2, 3, 4)

    assert usage == ModelUsage(11, 22, 33, 44)
    assert usage.total_tokens == 110
