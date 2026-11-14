"""Tool registry and built-in tools for the ReAct agent."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mcp_minion.mcp_client import ToolOutcome


@dataclass
class Tool:
    """A tool that the agent can use."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for parameters
    function: Callable[..., Any]

    def to_openai_tool(self) -> dict[str, Any]:
        """Convert to OpenAI tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, **kwargs: Any) -> ToolOutcome:
        """Execute the tool; an exception becomes an error outcome.

        A string return value is the outcome text as is; any other value is
        JSON-encoded (falling back to ``str`` for non-JSON values).
        """
        try:
            result = self.function(**kwargs)
        except Exception as e:
            return ToolOutcome(text=f"Tool '{self.name}' failed: {e}", is_error=True)
        if isinstance(result, str):
            return ToolOutcome(text=result)
        try:
            return ToolOutcome(text=json.dumps(result))
        except (TypeError, ValueError):
            return ToolOutcome(text=str(result))


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_openai_tools(self) -> list[dict[str, Any]]:
        """Get all tools in OpenAI format."""
        return [tool.to_openai_tool() for tool in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)


# Built-in tools


def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


# Tool definitions
ADD_TOOL = Tool(
    name="add",
    description="Add two numbers together. Use this when you need to calculate a sum.",
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "The first number"},
            "b": {"type": "number", "description": "The second number"},
        },
        "required": ["a", "b"],
    },
    function=add,
)

SUBTRACT_TOOL = Tool(
    name="subtract",
    description="Subtract the second number from the first.",
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "The number to subtract from"},
            "b": {"type": "number", "description": "The number to subtract"},
        },
        "required": ["a", "b"],
    },
    function=subtract,
)

MULTIPLY_TOOL = Tool(
    name="multiply",
    description="Multiply two numbers together.",
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "The first number"},
            "b": {"type": "number", "description": "The second number"},
        },
        "required": ["a", "b"],
    },
    function=multiply,
)

DIVIDE_TOOL = Tool(
    name="divide",
    description="Divide the first number by the second.",
    parameters={
        "type": "object",
        "properties": {
            "a": {
                "type": "number",
                "description": "The dividend (number to be divided)",
            },
            "b": {"type": "number", "description": "The divisor (number to divide by)"},
        },
        "required": ["a", "b"],
    },
    function=divide,
)


def create_default_registry() -> ToolRegistry:
    """Create a registry with the default tools."""
    registry = ToolRegistry()
    registry.register(ADD_TOOL)
    registry.register(SUBTRACT_TOOL)
    registry.register(MULTIPLY_TOOL)
    registry.register(DIVIDE_TOOL)
    return registry
