"""CodeRepair: a bounded, inspectable autonomous code-repair agent."""

from .agent import Agent, AgentLimits, AgentOutcome, RunStatus
from .model import AnthropicMessagesModel, FakeModel, ModelResponse, ModelUsage

__all__ = [
    "Agent",
    "AgentLimits",
    "AgentOutcome",
    "AnthropicMessagesModel",
    "FakeModel",
    "ModelResponse",
    "ModelUsage",
    "RunStatus",
]

__version__ = "0.1.0"

