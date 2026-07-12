"""ReAct agent implementation using OpenRouter."""

from __future__ import annotations

import asyncio
import json
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from openai import OpenAI

from mcp_minion.logging import RunLogger
from mcp_minion.tools import ToolRegistry

if TYPE_CHECKING:
    from mcp_minion.mcp_client import MCPManager


SYSTEM_PROMPT_BASE = """You are a helpful assistant that solves problems step by step.

When given a task, think through what you need to do and use the available tools to help.

{tool_sections}
Guidelines:
- Use tools when needed to get accurate information or perform calculations
- Think step by step about what tools to use
- After getting tool results, reason about them before continuing
- Provide a clear final answer when done
"""


def build_system_prompt(
    custom_template: str | None,
    tool_sections: str,
    base_template: str = SYSTEM_PROMPT_BASE,
) -> str:
    """Assemble the system prompt.

    If ``custom_template`` is None, the built-in base template is used with
    ``{tool_sections}`` filled in via ``str.format``.

    If a custom template is supplied (e.g. a solver template loaded from
    ``files.system``), it may contain literal braces (ASP templates do), so
    ``str.format`` is unsafe. Instead:

    * if the template contains the ``{tool_sections}`` placeholder, it is
      replaced literally with ``str.replace`` (surviving any other braces);
    * otherwise the generated tool sections are appended after the template.
    """
    if custom_template is None:
        return base_template.format(tool_sections=tool_sections)

    if "{tool_sections}" in custom_template:
        return custom_template.replace("{tool_sections}", tool_sections)

    if tool_sections:
        return f"{custom_template}\n\n{tool_sections}"
    return custom_template


def _extract_kernel_id(result: str) -> str | None:
    """Pull a ``kernel_id`` out of a python_reset tool result, if present.

    Supports both a flat payload ``{"kernel_id": "k1"}`` and the
    double-encoded shape produced by :class:`MCPManager`, which wraps the
    server's text output as ``{"result": "{\\"kernel_id\\": \\"k1\\"}"}``.
    """
    try:
        parsed = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(parsed, dict):
        return None

    inner = parsed.get("result")
    if isinstance(inner, str) and inner:
        try:
            inner = json.loads(inner)
        except (json.JSONDecodeError, TypeError):
            inner = None
    if isinstance(inner, dict) and inner.get("kernel_id"):
        return inner["kernel_id"]
    if parsed.get("kernel_id"):
        return parsed["kernel_id"]
    return None


@dataclass
class AgentConfig:
    """Configuration for the agent."""

    model: str = "google/gemini-3-flash-preview"
    max_steps: int = 10
    base_url: str = "https://openrouter.ai/api/v1"
    api_params: dict[str, Any] = field(
        default_factory=dict
    )  # Extra params for API call
    # Optional system-prompt body loaded from a run folder's files.system.
    # When set it replaces the built-in base prompt (see build_system_prompt).
    system_prompt: str | None = None
    # Packages to preinstall in a python_exec kernel via an automatic
    # python_reset before the first python_exec call.
    packages: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    """Result from agent execution."""

    answer: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    tool_calls_made: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    max_steps_reached: bool = False


class Agent:
    """A ReAct agent that uses tools to solve problems."""

    # Kept as a class attribute for backward compatibility; the canonical
    # source is the module-level SYSTEM_PROMPT_BASE.
    SYSTEM_PROMPT_BASE = SYSTEM_PROMPT_BASE

    def __init__(
        self,
        api_key: str,
        tools: ToolRegistry | None,
        config: AgentConfig | None = None,
        logger: RunLogger | None = None,
        mcp_manager: MCPManager | None = None,
        on_step: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.tools = tools
        self.logger = logger
        self.mcp_manager = mcp_manager
        # Optional progress hook, called with each step's step_info dict as
        # soon as the step completes (tool results included).
        self.on_step = on_step
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.config.base_url,
        )
        # Auto-tracked kernel state for python_exec / python_reset.
        self._kernel_id: str | None = None
        self._reset_done: bool = False

    def _build_tool_sections(self) -> str:
        """Build tool documentation sections for the system prompt."""
        if not self.mcp_manager:
            return ""

        tools = self.mcp_manager.get_tools()
        if not tools:
            return ""

        # Group tools by server
        servers: dict[str, list] = {}
        for tool in tools:
            servers.setdefault(tool.server_name, []).append(tool)

        sections = ["You have access to the following tools:\n"]

        for server_name, server_tools in servers.items():
            sections.append(f"## {server_name}")
            for tool in server_tools:
                desc = (
                    tool.description.split("\n")[0]
                    if tool.description
                    else "No description"
                )
                sections.append(f"- `{tool.name}`: {desc}")
            sections.append("")

        return "\n".join(sections) + "\n"

    def run(self, prompt: str) -> AgentResult:
        """Run the agent synchronously (convenience wrapper).

        For MCP tools, use run_async() directly or ensure no event loop is running.
        """
        return asyncio.run(self.run_async(prompt))

    async def run_async(self, prompt: str) -> AgentResult:
        """Run the agent with the given prompt (async version)."""
        # Log configuration and prompt
        if self.logger:
            self.logger.log_config(asdict(self.config))
            self.logger.log_prompt(prompt)

        # Build system prompt with tool sections
        tool_sections = self._build_tool_sections()
        system_prompt = build_system_prompt(self.config.system_prompt, tool_sections)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        steps: list[dict[str, Any]] = []
        tool_calls_made = 0
        input_tokens = 0
        output_tokens = 0

        # Combine tools from registry and MCP (MCP takes precedence)
        openai_tools = []
        mcp_tool_names: set[str] = set()

        if self.mcp_manager:
            mcp_tools = self.mcp_manager.get_openai_tools()
            openai_tools.extend(mcp_tools)
            mcp_tool_names = {t["function"]["name"] for t in mcp_tools}

        if self.tools:
            # Skip built-in tools that conflict with MCP tools
            for tool in self.tools.get_openai_tools():
                if tool["function"]["name"] not in mcp_tool_names:
                    openai_tools.append(tool)

        extra_headers = {
            "HTTP-Referer": "https://github.com/szeider/mcp-solver",
            "X-Title": "mcp-minion Agent",
        }
        extra_body: dict[str, Any] = {}

        # Filter reserved keys from api_params to prevent accidental overrides
        reserved_keys = {
            "model",
            "messages",
            "tools",
            "tool_choice",
            "extra_headers",
            "extra_body",
        }
        safe_api_params = {
            k: v for k, v in self.config.api_params.items() if k not in reserved_keys
        }

        try:
            for step_num in range(self.config.max_steps):
                if self.logger:
                    self.logger.log_step_start(step_num + 1)

                request_kwargs: dict[str, Any] = {
                    "model": self.config.model,
                    "messages": messages,
                    "extra_headers": extra_headers,
                    "extra_body": extra_body,
                    **safe_api_params,
                }

                # Only include tools if available (avoid passing None)
                if openai_tools:
                    request_kwargs["tools"] = openai_tools
                    request_kwargs["tool_choice"] = "auto"

                # Log API request (without headers for security)
                if self.logger:
                    self.logger.log_api_request(request_kwargs)

                response = self.client.chat.completions.create(**request_kwargs)

                if not response.choices:
                    raise RuntimeError("API returned no choices")

                assistant_message = response.choices[0].message

                # Accumulate token usage (independent of logging).
                if response.usage:
                    input_tokens += response.usage.prompt_tokens
                    output_tokens += response.usage.completion_tokens

                # Log API response
                if self.logger:
                    response_data = {
                        "id": response.id,
                        "model": response.model,
                        "content": assistant_message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in (assistant_message.tool_calls or [])
                        ],
                        "finish_reason": response.choices[0].finish_reason,
                        "usage": {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens,
                        }
                        if response.usage
                        else None,
                    }
                    self.logger.log_api_response(response_data)
                    self.logger.log_step_content(assistant_message.content)

                    # Accumulate token usage
                    if response.usage:
                        self.logger.log_token_usage(
                            input_tokens=response.usage.prompt_tokens,
                            output_tokens=response.usage.completion_tokens,
                        )

                # Record this step
                step_info: dict[str, Any] = {
                    "step": step_num + 1,
                    "content": assistant_message.content,
                    "tool_calls": None,
                }

                # Check if the model wants to use tools
                if assistant_message.tool_calls:
                    step_info["tool_calls"] = []

                    # Add assistant message to history (must include tool_calls)
                    messages.append(
                        {
                            "role": "assistant",
                            "content": assistant_message.content or "",
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in assistant_message.tool_calls
                            ],
                        }
                    )

                    # Execute each tool call
                    for tool_call in assistant_message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args_str = tool_call.function.arguments

                        try:
                            tool_args = json.loads(tool_args_str)
                        except (json.JSONDecodeError, TypeError):
                            tool_result = json.dumps(
                                {"error": f"Invalid JSON arguments: {tool_args_str}"}
                            )
                            tool_args = {"_raw": tool_args_str}
                        else:
                            tool_result = await self._execute_tool(tool_name, tool_args)

                        tool_calls_made += 1

                        # Log tool call
                        if self.logger:
                            self.logger.log_tool_call(
                                tool_name=tool_name,
                                tool_args=tool_args,
                                tool_result=tool_result,
                                tool_call_id=tool_call.id,
                            )

                        # Record tool call info
                        step_info["tool_calls"].append(
                            {
                                "name": tool_name,
                                "arguments": tool_args,
                                "result": tool_result,
                            }
                        )

                        # Add tool result to message history
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": tool_result,
                            }
                        )

                    steps.append(step_info)
                    if self.on_step:
                        self.on_step(step_info)

                else:
                    # No tool calls - this is the final answer
                    steps.append(step_info)
                    if self.on_step:
                        self.on_step(step_info)

                    result = AgentResult(
                        answer=assistant_message.content or "",
                        steps=steps,
                        tool_calls_made=tool_calls_made,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )

                    if self.logger:
                        self.logger.log_completion(
                            {
                                "answer": result.answer,
                                "steps_count": len(result.steps),
                                "tool_calls_made": result.tool_calls_made,
                            }
                        )

                    return result

            # Reached max steps without final answer
            result = AgentResult(
                answer="[Agent reached maximum steps without providing a final answer]",
                steps=steps,
                tool_calls_made=tool_calls_made,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                max_steps_reached=True,
            )

            if self.logger:
                self.logger.log_completion(
                    {
                        "answer": result.answer,
                        "steps_count": len(result.steps),
                        "tool_calls_made": result.tool_calls_made,
                        "max_steps_reached": True,
                    }
                )

            return result

        except Exception as e:
            if self.logger:
                self.logger.log_error(str(e), traceback.format_exc())
            raise

    async def _execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        """Execute a tool (built-in or MCP).

        Handles automatic kernel setup for python_exec: on the first
        python_exec call, if the config declares packages and no python_reset
        has run yet, a python_reset (carrying those packages) is issued first
        and any returned kernel_id is remembered and injected into later
        python_exec calls.
        """
        # Auto kernel setup before the first python_exec.
        if tool_name == "python_exec" and not self._reset_done:
            self._reset_done = True  # attempt auto-reset at most once
            if (
                self.config.packages
                and self.mcp_manager
                and self.mcp_manager.has_tool("python_reset")
            ):
                reset_result = await self.mcp_manager.call_tool(
                    "python_reset", {"packages": list(self.config.packages)}
                )
                self._kernel_id = _extract_kernel_id(reset_result)

        # Inject the tracked kernel_id into python_exec if not already set.
        if (
            tool_name == "python_exec"
            and self._kernel_id
            and "kernel_id" not in tool_args
        ):
            tool_args["kernel_id"] = self._kernel_id

        # Check MCP tools first
        if self.mcp_manager and self.mcp_manager.has_tool(tool_name):
            result = await self.mcp_manager.call_tool(tool_name, tool_args)
        else:
            # Check built-in tools
            tool = self.tools.get(tool_name) if self.tools else None
            if tool is not None:
                try:
                    result = tool.execute(**tool_args)
                except Exception as e:
                    result = json.dumps({"error": f"Tool '{tool_name}' failed: {e}"})
            else:
                result = json.dumps({"error": f"Unknown tool: {tool_name}"})

        # An explicit python_reset also satisfies the "reset happened" guard
        # and updates the tracked kernel_id.
        if tool_name == "python_reset":
            self._reset_done = True
            kernel_id = _extract_kernel_id(result)
            if kernel_id:
                self._kernel_id = kernel_id

        return result
