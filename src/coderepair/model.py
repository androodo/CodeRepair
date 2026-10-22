"""Small model boundary around Anthropic's Messages API.

The agent depends on this module's protocol rather than on the SDK directly. That
keeps the production path narrow and makes complete tool loops deterministic in
unit tests.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

JsonObject = dict[str, Any]
Message = dict[str, Any]


@dataclass(frozen=True)
class ModelUsage:
    """Token counts reported by the provider; missing fields remain zero."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    def __add__(self, other: ModelUsage) -> ModelUsage:
        return ModelUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
        )


@dataclass(frozen=True)
class ModelResponse:
    """Provider-neutral subset required by the custom execution loop."""

    content: list[JsonObject]
    stop_reason: str
    usage: ModelUsage = field(default_factory=ModelUsage)
    model: str | None = None
    response_id: str | None = None


class ModelError(RuntimeError):
    """A sanitized model request failure."""

    def __init__(self, message: str, *, retriable: bool = False) -> None:
        super().__init__(message)
        self.retriable = retriable


class ModelClient(Protocol):
    def create_message(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[JsonObject],
        max_tokens: int,
    ) -> ModelResponse:
        """Create one assistant message."""


class AnthropicMessagesModel:
    """Thin synchronous adapter for the current Anthropic Messages API.

    Complete content blocks are serialized and retained. This is important for
    current Claude models because signed thinking blocks must be sent back
    unchanged on the next request in a tool-use turn.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
        client: Any | None = None,
    ) -> None:
        self.model = model
        if client is not None:
            self._client = client
            return
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - packaging installs the dependency
            raise ModelError("The 'anthropic' package is not installed") from exc
        kwargs: dict[str, Any] = {
            "timeout": timeout_seconds,
            "max_retries": max_retries,
        }
        if api_key is not None:
            kwargs["api_key"] = api_key
        self._client = Anthropic(**kwargs)

    def create_message(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[JsonObject],
        max_tokens: int,
    ) -> ModelResponse:
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=list(messages),
                tools=list(tools),
            )
        except Exception as exc:  # SDK exception classes can change across compatible releases.
            status = getattr(exc, "status_code", None)
            retriable = status in {408, 409, 429} or (isinstance(status, int) and status >= 500)
            name = type(exc).__name__
            raise ModelError(
                f"Anthropic request failed ({name}): {exc}", retriable=retriable
            ) from exc

        to_param = getattr(response, "to_param", None)
        param = to_param() if callable(to_param) else None
        if isinstance(param, dict) and isinstance(param.get("content"), list):
            blocks = [copy.deepcopy(block) for block in param["content"]]
        else:
            blocks = [_serialize_block(block) for block in response.content]
        raw_usage = getattr(response, "usage", None)
        usage = ModelUsage(
            input_tokens=_integer_attr(raw_usage, "input_tokens"),
            output_tokens=_integer_attr(raw_usage, "output_tokens"),
            cache_creation_input_tokens=_integer_attr(raw_usage, "cache_creation_input_tokens"),
            cache_read_input_tokens=_integer_attr(raw_usage, "cache_read_input_tokens"),
        )
        return ModelResponse(
            content=blocks,
            stop_reason=str(getattr(response, "stop_reason", "unknown") or "unknown"),
            usage=usage,
            model=str(getattr(response, "model", self.model)),
            response_id=str(getattr(response, "id", "")) or None,
        )


ResponseFactory = Callable[[int, str, Sequence[Message], Sequence[JsonObject]], ModelResponse]


class FakeModel:
    """Deterministic model used by tests and offline demonstrations."""

    def __init__(self, responses: Iterable[ModelResponse] | ResponseFactory) -> None:
        self._factory = responses if callable(responses) else None
        self._responses = None if callable(responses) else iter(responses)
        self.calls: list[dict[str, Any]] = []

    def create_message(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[JsonObject],
        max_tokens: int,
    ) -> ModelResponse:
        call = {
            "system": system,
            "messages": copy.deepcopy(list(messages)),
            "tools": copy.deepcopy(list(tools)),
            "max_tokens": max_tokens,
        }
        self.calls.append(call)
        index = len(self.calls) - 1
        if self._factory is not None:
            return self._factory(index, system, messages, tools)
        assert self._responses is not None
        try:
            return next(self._responses)
        except StopIteration as exc:
            raise ModelError("FakeModel ran out of scripted responses") from exc


def _serialize_block(block: Any) -> JsonObject:
    if isinstance(block, dict):
        return copy.deepcopy(block)
    model_dump = getattr(block, "model_dump", None)
    if callable(model_dump):
        value = model_dump(mode="json", exclude_none=True)
        if isinstance(value, dict):
            return value
    raise ModelError(f"Unsupported Anthropic content block: {type(block).__name__}")


def _integer_attr(value: Any, name: str) -> int:
    candidate = getattr(value, name, 0) if value is not None else 0
    return candidate if isinstance(candidate, int) and candidate >= 0 else 0

