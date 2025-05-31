"""
Tool Capability Detector for MCP-Solver (EXPERIMENTAL)

This module provides model capability detection for tool calling in MCP-Solver.
It helps identify what level of tool calling support a model has, supporting
both cloud and local models.

NOTE: This detector is EXPERIMENTAL and still under development. Results should be
considered preliminary guidance rather than definitive classifications. Some models,
particularly Claude, may be incorrectly classified as JSON_ONLY despite having
native tool calling capabilities.

## Overview

MCP-Solver categorizes models based on their tool calling abilities:

1. NATIVE: Full native tool calling support (e.g., Claude, GPT-4)
2. JSON_ONLY: Can produce JSON but cannot invoke tools directly (e.g., many local models)
3. NONE: Limited capability or cannot reliably output JSON (some basic models)

## Usage

Test a model's capability using test_setup.py with the --test-tool-calling flag:

```bash
uv run python -m mcp_solver.client.test_setup --mc "LM:model@url" --test-tool-calling
```

For example, to test a local model running in LM Studio:

```bash
uv run python -m mcp_solver.client.test_setup --mc "LM:unsloth/llama-3.2-3b-instruct@http://localhost:1234/v1" --test-tool-calling
```

## Working with Different Model Types

1. Cloud Models (NATIVE): Models like Claude and GPT-4 support tool calling natively
   and will work with MCP-Solver's client.py out of the box.

2. Local Models with JSON capability (JSON_ONLY): Models like llama-3.2-3b-instruct
   can produce correctly formatted JSON and will work with MCP-Solver through
   LangGraph's adaptation layer, but may not invoke tools directly.

3. Models with Limited Capability (NONE): Some models can't reliably produce
   JSON or follow tool calling instructions. These may struggle with MCP-Solver's
   tool-intensive workflows.

## Implementation Details

The system works by:
1. Testing if the model can output JSON when instructed to
2. Testing if the model can directly call tools
3. Categorizing the model based on these tests
4. Testing if tool calls can be extracted from the model's JSON output

## Recommended Local Models

Based on our testing, these local models work well with MCP-Solver:
- unsloth/llama-3.2-3b-instruct: 3B parameter model with good JSON capabilities
- mistralai.ministral-8b-instruct-2410: 8B parameter model with good JSON capabilities
"""

import json
import re
from typing import Any


class ToolCallCapability:
    """Enum of tool calling capabilities a model might support."""

    NONE = "none"  # No tool calling support
    JSON_ONLY = "json_only"  # Model can output JSON but not invoke tools directly
    NATIVE = "native"  # Full native tool calling support


class ToolCapabilityDetector:
    """
    Detector for tool calling capabilities of different models (EXPERIMENTAL).

    This class can detect the level of tool calling support a model has,
    which is useful for adapting the client's behavior to different models.

    NOTE: This detector is EXPERIMENTAL and still being refined. Test results should be
    considered as guidance rather than definitive capabilities. Some models like
    Claude may be incorrectly classified as JSON_ONLY despite having native tool
    calling support. This is due to differences in how LangChain wrappers implement
    the tool calling API for different model providers.
    """

    def __init__(self):
        self.model_capabilities: dict[str, str] = {}  # model_id -> capability type

    def detect_capability(self, model, model_code: str) -> str:
        """
        Detect what kind of tool calling capability a model has.

        Args:
            model: LLM model instance
            model_code: Unique identifier for the model

        Returns:
            str: The detected capability level (from ToolCallCapability)
        """
        # Check if we've already detected this model's capabilities
        if model_code in self.model_capabilities:
            return self.model_capabilities[model_code]

        # First check if model can output JSON
        has_json = self._test_json_output(model)
        if not has_json:
            capability = ToolCallCapability.NONE
        else:
            # Test direct tool calling
            tool_called = self._test_direct_tool_calling(model)
            if tool_called:
                capability = ToolCallCapability.NATIVE
            else:
                # JSON output but no direct tool calling
                capability = ToolCallCapability.JSON_ONLY

        self.model_capabilities[model_code] = capability
        return capability

    def _test_json_output(self, model) -> bool:
        """Test if the model can output JSON when instructed to."""
        try:
            # System message explicitly asking for JSON
            system_prompt = """You are a helpful assistant that always responds in JSON format.
            When asked a math question, respond with a JSON object like this:
            {"answer": 42, "explanation": "This is the answer"}
            
            ONLY respond with valid JSON. Do not include any other text before or after the JSON."""

            # Simple prompt
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "What is 2 plus 2?"},
            ]

            # Invoke the model
            response = model.invoke(messages)

            # Get the response content
            content = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Look for JSON patterns
            json_pattern = r"\{.*\}"
            matches = re.findall(json_pattern, content, re.DOTALL)

            if matches:
                for match in matches:
                    try:
                        json.loads(match)
                        return True
                    except json.JSONDecodeError:
                        # Try cleaning up the JSON
                        cleaned = match.replace("\n", "").replace("\\", "").strip()
                        try:
                            json.loads(cleaned)
                            return True
                        except:
                            pass

            return False

        except Exception:
            # If we get an error (like "unordered_map::at: key not found"),
            # try a simpler test without system message
            try:
                # User message asking for JSON
                messages = [
                    {
                        "role": "user",
                        "content": 'Please respond with a simple JSON object like this: {"answer": 4}. Just return the JSON.',
                    }
                ]

                # Invoke the model
                response = model.invoke(messages)

                # Get the response content
                content = (
                    response.content if hasattr(response, "content") else str(response)
                )

                # Look for JSON patterns
                json_pattern = r"\{.*\}"
                matches = re.findall(json_pattern, content, re.DOTALL)

                if matches:
                    for match in matches:
                        try:
                            json.loads(match)
                            return True
                        except json.JSONDecodeError:
                            # Try cleaning up the JSON
                            cleaned = match.replace("\n", "").replace("\\", "").strip()
                            try:
                                json.loads(cleaned)
                                return True
                            except:
                                pass

                return False

            except Exception:
                # Both approaches failed
                return False

    def _test_direct_tool_calling(self, model) -> bool:
        """Test if the model can directly call tools."""
        try:
            # Create a flag to track if the tool was called
            tool_called = False

            # Define a simple tool
            def calculator(x: float, y: float, operation: str) -> float:
                """Calculate the result of an operation on two numbers."""
                nonlocal tool_called
                tool_called = True

                if operation == "add":
                    return x + y
                elif operation == "multiply":
                    return x * y
                else:
                    return 0

            # Create a tool definition
            calculator_tool = {
                "name": "calculator",
                "description": "Calculate the result of an operation on two numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "description": "First number"},
                        "y": {"type": "number", "description": "Second number"},
                        "operation": {
                            "type": "string",
                            "description": "Operation to perform (add, multiply)",
                        },
                    },
                    "required": ["x", "y", "operation"],
                },
            }

            # Check if model has the bind_tools method
            if not hasattr(model, "bind_tools"):
                return False

            # Try to bind the tool
            model_with_tools = model.bind_tools([calculator_tool])

            # Create test messages
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that can use tools. Use the calculator tool when asked to perform calculations.",
                },
                {
                    "role": "user",
                    "content": "What is 2 plus 3? Please use the calculator tool.",
                },
            ]

            # Call the model
            model_with_tools.invoke(messages)

            # Return whether the tool was called
            return tool_called

        except Exception:
            # Failed to invoke tool
            return False

    def extract_tool_call(
        self, response_text: str
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Extract a tool call from the model's response text.

        Args:
            response_text: The model's response text

        Returns:
            Tuple[bool, Optional[Dict]]: Whether a tool call was found and the tool call info
        """
        # Look for JSON patterns
        json_pattern = r"\{[\s\S]*\}"
        matches = re.findall(json_pattern, response_text, re.DOTALL)

        for match in matches:
            try:
                parsed = json.loads(match)

                # Check different possible schema patterns for function/tool calls
                if "function_call" in parsed:
                    return True, parsed["function_call"]

                elif "tool_calls" in parsed:
                    # OpenAI format - get the first tool call
                    if (
                        isinstance(parsed["tool_calls"], list)
                        and len(parsed["tool_calls"]) > 0
                    ):
                        return True, parsed["tool_calls"][0]

                elif "name" in parsed and "arguments" in parsed:
                    # Direct tool call format
                    return True, parsed

            except json.JSONDecodeError:
                # Try cleaning up the JSON and try again
                cleaned = match.replace("\n", "").replace("\\", "").strip()
                try:
                    parsed = json.loads(cleaned)
                    # Repeat the same checks as above
                    if "function_call" in parsed:
                        return True, parsed["function_call"]
                    elif "tool_calls" in parsed:
                        if (
                            isinstance(parsed["tool_calls"], list)
                            and len(parsed["tool_calls"]) > 0
                        ):
                            return True, parsed["tool_calls"][0]
                    elif "name" in parsed and "arguments" in parsed:
                        return True, parsed
                except:
                    pass

        # No tool call found
        return False, None

    def get_enhanced_prompt(self, capability: str) -> str:
        """
        Get an enhanced system prompt based on the model's capability.

        Args:
            capability: The model's capability type

        Returns:
            str: A system prompt tailored to the model's capability
        """
        if capability == ToolCallCapability.NATIVE:
            # For models with native tool calling support
            return """You are a helpful assistant that can use tools.
            When you need to use a tool, use the provided tool interface."""

        else:  # JSON_ONLY or NONE
            # For models with limited tool calling abilities
            return """You are a helpful assistant that can use tools.
            
            When you need to use a tool, respond with a JSON object in this format:
            
            ```json
            {
              "function_call": {
                "name": "tool_name",
                "arguments": {
                  "param1": "value1",
                  "param2": "value2"
                }
              }
            }
            ```
            
            For example, if using a calculator tool to add 2 and 3:
            
            ```json
            {
              "function_call": {
                "name": "calculator",
                "arguments": {
                  "x": 2,
                  "y": 3,
                  "operation": "add"
                }
              }
            }
            ```
            
            After the tool provides a result, continue with your response.
            """
