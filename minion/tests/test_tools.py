"""Unit tests for tools module."""

import json

import pytest

from mcp_minion.tools import (
    ADD_TOOL,
    DIVIDE_TOOL,
    MULTIPLY_TOOL,
    SUBTRACT_TOOL,
    Tool,
    ToolRegistry,
    add,
    create_default_registry,
    divide,
    multiply,
    subtract,
)


class TestMathFunctions:
    """Tests for the raw math functions."""

    def test_add(self) -> None:
        assert add(2, 3) == 5
        assert add(-1, 1) == 0
        assert add(0.5, 0.5) == 1.0

    def test_subtract(self) -> None:
        assert subtract(5, 3) == 2
        assert subtract(1, 1) == 0
        assert subtract(0, 5) == -5

    def test_multiply(self) -> None:
        assert multiply(3, 4) == 12
        assert multiply(-2, 3) == -6
        assert multiply(0, 100) == 0

    def test_divide(self) -> None:
        assert divide(10, 2) == 5
        assert divide(7, 2) == 3.5
        assert divide(-6, 3) == -2

    def test_divide_by_zero(self) -> None:
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide(5, 0)


class TestTool:
    """Tests for the Tool class."""

    def test_to_openai_tool(self) -> None:
        result = ADD_TOOL.to_openai_tool()
        assert result["type"] == "function"
        assert result["function"]["name"] == "add"
        assert "description" in result["function"]
        assert "parameters" in result["function"]

    def test_execute_success(self) -> None:
        result = ADD_TOOL.execute(a=2, b=3)
        parsed = json.loads(result)
        assert parsed == {"result": 5}

    def test_execute_error(self) -> None:
        result = DIVIDE_TOOL.execute(a=5, b=0)
        parsed = json.loads(result)
        assert "error" in parsed
        assert "zero" in parsed["error"].lower()

    def test_execute_multiply(self) -> None:
        result = MULTIPLY_TOOL.execute(a=7, b=8)
        parsed = json.loads(result)
        assert parsed == {"result": 56}

    def test_execute_subtract(self) -> None:
        result = SUBTRACT_TOOL.execute(a=10, b=4)
        parsed = json.loads(result)
        assert parsed == {"result": 6}


class TestToolRegistry:
    """Tests for the ToolRegistry class."""

    def test_register_and_get(self) -> None:
        registry = ToolRegistry()
        registry.register(ADD_TOOL)
        assert registry.get("add") is ADD_TOOL
        assert registry.get("nonexistent") is None

    def test_len(self) -> None:
        registry = ToolRegistry()
        assert len(registry) == 0
        registry.register(ADD_TOOL)
        assert len(registry) == 1
        registry.register(MULTIPLY_TOOL)
        assert len(registry) == 2

    def test_get_openai_tools(self) -> None:
        registry = ToolRegistry()
        registry.register(ADD_TOOL)
        registry.register(MULTIPLY_TOOL)
        tools = registry.get_openai_tools()
        assert len(tools) == 2
        names = {t["function"]["name"] for t in tools}
        assert names == {"add", "multiply"}

    def test_create_default_registry(self) -> None:
        registry = create_default_registry()
        assert len(registry) == 4
        assert registry.get("add") is not None
        assert registry.get("subtract") is not None
        assert registry.get("multiply") is not None
        assert registry.get("divide") is not None


class TestCustomTool:
    """Tests for creating custom tools."""

    def test_custom_tool(self) -> None:
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        tool = Tool(
            name="greet",
            description="Greet someone by name",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name to greet"},
                },
                "required": ["name"],
            },
            function=greet,
        )

        result = tool.execute(name="World")
        parsed = json.loads(result)
        assert parsed == {"result": "Hello, World!"}
