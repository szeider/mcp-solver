"""ReAct agent implementation using OpenRouter."""

from __future__ import annotations

import asyncio
import json
import re
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI, OpenAIError

from mcp_minion.logging import RESERVED_REQUEST_KEYS, RunLogger
from mcp_minion.mcp_client import ToolOutcome
from mcp_minion.tools import ToolRegistry

if TYPE_CHECKING:
    from mcp_minion.mcp_client import MCPManager


# Retries for API error responses the OpenAI SDK parses without raising:
# OpenRouter intermittently returns HTTP 200 whose body is an error object
# ({"choices": null, "error": {...}}), or a choice with finish_reason "error".
# (Ported from agentic-python-coder's agent.py; exceptions raised by the SDK
# are retried by the SDK itself.)
API_ERROR_MAX_RETRIES = 3
API_ERROR_BACKOFF_SECONDS = 5

# Sent when a reply without tool calls was cut off at max_tokens.
TRUNCATION_NOTICE = (
    "Your previous response was truncated at the max_tokens limit."
    " Continue, and be more concise."
)


# Kernel-server tools (ipython_mcp from agentic-python-coder) that take a
# kernel_id; the tracked kernel_id is injected into calls that lack one.
KERNEL_TOOLS = frozenset(
    ("python_exec", "python_reset", "python_status", "python_interrupt")
)

# Prefix of a tool message whose outcome is an error.
ERROR_PREFIX = "ERROR: "

_MISSING = object()


class LLMResponseError(RuntimeError):
    """The API kept returning error responses after all retries."""


def _response_error(response: Any) -> str | None:
    """Return a description if the response is an error envelope, else None."""
    if not response.choices:
        error = (getattr(response, "model_extra", None) or {}).get("error")
        if error:
            return f"no choices, error: {error}"
        dump = getattr(response, "model_dump_json", None)
        body = dump() if callable(dump) else repr(response)
        return f"no choices, body: {body[:500]}"
    choice = response.choices[0]
    if choice.finish_reason == "error":
        error = (getattr(choice, "model_extra", None) or {}).get("error") or (
            getattr(response, "model_extra", None) or {}
        ).get("error")
        return f"finish_reason 'error', error: {error}"
    return None


def _one_line(text: str, limit: int = 500) -> str:
    """Collapse whitespace so a message fits on one line, then truncate."""
    line = " ".join(text.split())
    return line if len(line) <= limit else line[: limit - 1] + "…"


def _echo_arguments(raw: Any) -> str:
    """Tool-call arguments as the string sent back in the assistant history.

    A non-empty string is kept; missing, empty or blank arguments become
    "{}" (the API rejects null); any other JSON value is JSON-encoded.
    """
    if isinstance(raw, str):
        return raw if raw.strip() else "{}"
    if raw is None:
        return "{}"
    try:
        return json.dumps(raw)
    except (TypeError, ValueError):
        return json.dumps(str(raw))


def truncate_for_model(text: str, limit: int) -> str:
    """Cut ``text`` to about ``limit`` characters: head, marker line, tail.

    Roughly 3/4 of the budget goes to the head and 1/4 to the tail. Text
    within the limit, and any text when ``limit`` <= 0, is returned as is.
    """
    if limit <= 0 or len(text) <= limit:
        return text
    head = limit * 3 // 4
    tail = limit - head
    cut = len(text) - head - tail
    return f"{text[:head]}\n[... {cut} characters truncated ...]\n{text[-tail:]}"


def merge_extra_body(user: dict[str, Any] | None) -> dict[str, Any]:
    """The request's extra_body: user keys over ``{"usage": {"include": True}}``.

    User keys win, except that ``usage.include`` stays on (OpenRouter usage
    accounting, which reports the per-step cost).
    """
    body = dict(user or {})
    usage = body.get("usage")
    body["usage"] = {**(usage if isinstance(usage, dict) else {}), "include": True}
    return body


def _parse_tool_arguments(raw: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Parse a tool call's arguments into a dict.

    Returns ``(args, None)`` on success and ``(None, message)`` when the
    arguments are unusable. Empty or missing arguments count as ``{}``;
    arguments that parse to anything but a JSON object are an error.
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}, None
    if isinstance(raw, dict):
        return raw, None
    if not isinstance(raw, str | bytes | bytearray):
        return None, (
            f"Tool arguments must be a JSON object, got {type(raw).__name__}: {raw!r}"
        )
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, RecursionError):
        return None, f"Invalid JSON arguments: {raw}"
    if not isinstance(parsed, dict):
        return None, (
            f"Tool arguments must be a JSON object, got {type(parsed).__name__}: {raw}"
        )
    return parsed, None


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


# Built-in tool for following MCP resources (announced ones and resource_links
# received in tool results). Only offered when an MCP manager is attached and
# no server defines a tool of the same name.
READ_RESOURCE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_resource",
        "description": "Read an MCP resource by its URI — one listed in the"
        " Resources section of your instructions, or received as a"
        " [resource_link] in a tool result.",
        "parameters": {
            "type": "object",
            "properties": {
                "uri": {
                    "type": "string",
                    "description": "The resource URI to read.",
                }
            },
            "required": ["uri"],
        },
    },
}


def _reasoning_record(message: Any) -> dict[str, Any]:
    """The reasoning fields an assistant message carries, for the records.

    ``reasoning_details`` (a non-empty list, as OpenRouter returns it) and a
    non-empty string ``reasoning`` are included unmodified when present.
    """
    record: dict[str, Any] = {}
    details = getattr(message, "reasoning_details", None)
    if isinstance(details, list) and details:
        record["reasoning_details"] = details
    reasoning = getattr(message, "reasoning", None)
    if isinstance(reasoning, str) and reasoning:
        record["reasoning"] = reasoning
    return record


def _reasoning_replay(record: dict[str, Any]) -> dict[str, Any]:
    """Reasoning to replay on the history message: the details list as is,
    else the plain reasoning string, else nothing (no null keys)."""
    if "reasoning_details" in record:
        return {"reasoning_details": record["reasoning_details"]}
    if "reasoning" in record:
        return {"reasoning": record["reasoning"]}
    return {}


_DIST_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _package_key(entry: Any) -> str:
    """Comparison key of a package spec: the normalized distribution name.

    Extras and version specifiers are dropped and ``-``/``_``/``.`` runs are
    treated alike, case-insensitively ("Python_SAT[pb]>=1" -> "python-sat").
    Paths and URLs (anything with a slash or backslash, or starting with "."
    or "~") are compared as plain strings.
    """
    text = str(entry).strip()
    if "/" in text or "\\" in text or text.startswith((".", "~")):
        return "raw:" + text
    match = _DIST_NAME.match(text)
    if match is None:
        return "raw:" + text
    return re.sub(r"[-_.]+", "-", match.group()).lower()


def merge_packages(configured: list[Any], requested: list[Any]) -> list[Any]:
    """The configured packages, then the requested ones they do not cover."""
    merged = list(configured)
    seen = {_package_key(p) for p in configured}
    for entry in requested:
        key = _package_key(entry)
        if key not in seen:
            seen.add(key)
            merged.append(entry)
    return merged


def _json_object(text: str) -> dict[str, Any] | None:
    """``text`` parsed as a JSON object, or None."""
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError, RecursionError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_kernel_id(outcome: ToolOutcome) -> str | None:
    """The ``kernel_id`` of a successful python_reset outcome, if any."""
    payload = _json_object(outcome.text)
    if outcome.is_error or payload is None or payload.get("success") is False:
        return None
    kernel_id = payload.get("kernel_id")
    return kernel_id if isinstance(kernel_id, str) and kernel_id else None


def _reset_failed(outcome: ToolOutcome) -> bool:
    """True for an error outcome or a JSON object with ``success`` false."""
    if outcome.is_error:
        return True
    payload = _json_object(outcome.text)
    return payload is not None and payload.get("success") is False


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
    # Packages for the kernel server: when set and a server offers
    # python_reset, run_async calls python_reset with them before the first
    # LLM call and ends the run if that fails.
    packages: list[str] = field(default_factory=list)
    # Hard cap on the cumulative input+output tokens of one run; 0 (the
    # default) means unlimited. Checked after each completed step.
    max_total_tokens: int = 0
    # Provider-specific request fields (e.g. OpenRouter "reasoning",
    # "provider", "top_k"), merged over {"usage": {"include": True}}; user
    # keys win except that usage.include stays on.
    extra_body: dict[str, Any] | None = None
    # Per-request timeout in seconds for the LLM call (None: SDK default).
    request_timeout: float | None = None
    # Tool results longer than this are cut (head + tail) in the model's
    # context; step records and the run log keep the full text. 0 disables.
    max_tool_result_chars: int = 64000


@dataclass
class AgentResult:
    """Result from agent execution."""

    answer: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    tool_calls_made: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    # Actual charge in USD, summed from OpenRouter's usage accounting
    # (0.0 when the provider does not report cost).
    cost: float = 0.0
    max_steps_reached: bool = False
    # True when the run was aborted by AgentConfig.max_total_tokens.
    token_cap_reached: bool = False
    # Set (one line, with the exception type) when the run was cut short by
    # an LLM API failure; steps and totals then cover the steps completed.
    error: str | None = None


class Agent:
    """A ReAct agent that uses tools to solve problems.

    Client lifetime: if ``agent.client`` is set (injected by the caller),
    every run uses it and never closes it; otherwise each run creates its own
    ``AsyncOpenAI`` client in the running event loop and closes it when the
    run ends, so one Agent can run several times.
    """

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
        self._api_key = api_key
        # Injected LLM client (any object with an async
        # chat.completions.create); None: one AsyncOpenAI client per run.
        self.client: Any = None
        # Kernel tracked from python_reset results, injected into kernel
        # tool calls that carry no kernel_id.
        self._kernel_id: str | None = None

    def _build_tool_sections(self) -> str:
        """Build tool documentation sections for the system prompt."""
        if not self.mcp_manager:
            return ""

        tools = self.mcp_manager.get_tools()
        resources = self.mcp_manager.get_resources()
        instructions = self.mcp_manager.get_instructions()
        if not tools and not resources and not instructions:
            return ""

        # Group tools by server; a server with instructions but no tools
        # still gets a section.
        servers: dict[str, list] = {}
        for tool in tools:
            servers.setdefault(tool.server_name, []).append(tool)
        for server_name in instructions:
            servers.setdefault(server_name, [])

        sections = ["You have access to the following tools:\n"]

        for server_name, server_tools in servers.items():
            sections.append(f"## {server_name}")
            if instructions.get(server_name):
                sections.append(f"Server instructions:\n{instructions[server_name]}\n")
            for tool in server_tools:
                desc = (
                    tool.description.split("\n")[0]
                    if tool.description
                    else "No description"
                )
                sections.append(f"- `{tool.name}`: {desc}")
            sections.append("")

        # Announced resources: list them cheaply; content is fetched on
        # demand via read_resource (successive revelation).
        if resources:
            sections.append("## Resources")
            sections.append(
                "Fetch any of these — or any [resource_link] URI you receive"
                " in a tool result — with the `read_resource` tool:"
            )
            for res in resources:
                desc = res.description.split("\n")[0] if res.description else ""
                suffix = f": {desc}" if desc else ""
                sections.append(f"- `{res.uri}` ({res.name}){suffix}")
            sections.append("")

        return "\n".join(sections) + "\n"

    async def _create_with_retry(
        self, client: Any, request_kwargs: dict[str, Any]
    ) -> Any:
        """Call the completions API, retrying error responses.

        Raises LLMResponseError when retries are exhausted, so the run ends
        with an error instead of a silent early stop reported as success.
        """
        detail = ""
        for attempt in range(API_ERROR_MAX_RETRIES + 1):
            # The SDK does not wrap a malformed JSON body (JSONDecodeError)
            # and returns the raw text for a non-JSON content type (e.g. an
            # HTML error page); both are treated as error responses.
            try:
                response = await client.chat.completions.create(**request_kwargs)
            except json.JSONDecodeError as e:
                detail = f"malformed JSON response: {e}"
            else:
                choices = getattr(response, "choices", _MISSING)
                if choices is _MISSING or not (
                    choices is None or isinstance(choices, list)
                ):
                    detail = (
                        f"unexpected response of type {type(response).__name__}:"
                        f" {str(response)[:500]}"
                    )
                else:
                    detail = _response_error(response)
            if detail is None:
                return response
            if self.logger:
                self.logger.log_step_metadata(
                    api_error_response=_one_line(detail), api_attempt=attempt + 1
                )
            if attempt < API_ERROR_MAX_RETRIES:
                await asyncio.sleep(API_ERROR_BACKOFF_SECONDS * (attempt + 1))
        raise LLMResponseError(
            f"API returned error responses on {API_ERROR_MAX_RETRIES + 1}"
            f" attempts; last: {detail}"
        )

    def run(self, prompt: str) -> AgentResult:
        """Run the agent synchronously (convenience wrapper).

        For MCP tools, use run_async() directly or ensure no event loop is running.
        """
        return asyncio.run(self.run_async(prompt))

    async def run_async(self, prompt: str) -> AgentResult:
        """Run the agent with the given prompt (async version)."""
        client = self.client
        owned = client is None
        if owned:
            client = AsyncOpenAI(api_key=self._api_key, base_url=self.config.base_url)
        try:
            return await self._run(prompt, client)
        finally:
            if owned:
                await client.close()

    async def _run(self, prompt: str, client: Any) -> AgentResult:
        """One run with the given LLM client (see run_async)."""
        # Log configuration and prompt
        if self.logger:
            self.logger.log_config(asdict(self.config))
            self.logger.log_prompt(prompt)

        # Kernel setup comes before any LLM call: a failure ends the run.
        setup_error = await self._setup_kernel()
        if setup_error:
            if self.logger:
                self.logger.log_error(setup_error)
            return AgentResult(
                answer=f"[Agent stopped: {setup_error}]", error=setup_error
            )

        # Build system prompt with tool sections
        tool_sections = self._build_tool_sections()
        system_prompt = build_system_prompt(self.config.system_prompt, tool_sections)
        if self.logger:
            self.logger.log_system_prompt(system_prompt)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        steps: list[dict[str, Any]] = []
        tool_calls_made = 0
        input_tokens = 0
        output_tokens = 0
        cost = 0.0
        token_cap_reached = False

        # Combine tools from registry and MCP (MCP takes precedence)
        openai_tools = []
        mcp_tool_names: set[str] = set()

        if self.mcp_manager:
            mcp_tools = self.mcp_manager.get_openai_tools()
            openai_tools.extend(mcp_tools)
            mcp_tool_names = {t["function"]["name"] for t in mcp_tools}
            # Resource access, unless a server claims the name itself.
            if "read_resource" not in mcp_tool_names:
                openai_tools.append(READ_RESOURCE_TOOL)
                mcp_tool_names.add("read_resource")

        if self.tools:
            # Skip built-in tools that conflict with MCP tools
            for tool in self.tools.get_openai_tools():
                if tool["function"]["name"] not in mcp_tool_names:
                    openai_tools.append(tool)

        extra_headers = {
            "HTTP-Referer": "https://github.com/szeider/mcp-solver",
            "X-Title": "mcp-minion Agent",
        }

        # Filter reserved keys from api_params to prevent accidental overrides
        safe_api_params = {
            k: v
            for k, v in self.config.api_params.items()
            if k not in RESERVED_REQUEST_KEYS
        }

        try:
            for step_num in range(self.config.max_steps):
                if self.logger:
                    self.logger.log_step_start(step_num + 1)

                request_kwargs: dict[str, Any] = {
                    "model": self.config.model,
                    "messages": messages,
                    "extra_headers": extra_headers,
                    # OpenRouter usage accounting (always on): the response
                    # then carries the actual cost (USD) in usage.cost.
                    "extra_body": merge_extra_body(self.config.extra_body),
                    **safe_api_params,
                }
                if self.config.request_timeout is not None:
                    request_kwargs["timeout"] = self.config.request_timeout

                # Only include tools if available (avoid passing None)
                if openai_tools:
                    request_kwargs["tools"] = openai_tools
                    request_kwargs["tool_choice"] = "auto"

                # Log API request (without headers for security)
                if self.logger:
                    self.logger.log_api_request(request_kwargs)

                try:
                    response = await self._create_with_retry(client, request_kwargs)
                except (OpenAIError, LLMResponseError) as e:
                    # Keep the steps so far: end the run with an error
                    # result instead of raising.
                    error = _one_line(f"{type(e).__name__}: {e}")
                    if self.logger:
                        self.logger.log_error(error, traceback.format_exc())
                    return AgentResult(
                        answer=f"[Agent stopped: LLM call failed: {error}]",
                        steps=steps,
                        tool_calls_made=tool_calls_made,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost=cost,
                        error=error,
                    )

                assistant_message = response.choices[0].message
                finish_reason = response.choices[0].finish_reason
                # Reasoning state, replayed unmodified on the history message
                # (OpenRouter: reasoning blocks must match the model output).
                reasoning = _reasoning_record(assistant_message)
                replay = _reasoning_replay(reasoning)

                # Accumulate token usage and cost (independent of logging).
                step_cost = None
                if response.usage:
                    input_tokens += response.usage.prompt_tokens
                    output_tokens += response.usage.completion_tokens
                    step_cost = getattr(response.usage, "cost", None)
                    if isinstance(step_cost, int | float) and not isinstance(
                        step_cost, bool
                    ):
                        cost += float(step_cost)
                    else:
                        step_cost = None

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
                        "finish_reason": finish_reason,
                        **reasoning,
                        "usage": {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens,
                            "cost": step_cost,
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
                    **reasoning,
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
                                        "arguments": _echo_arguments(
                                            tc.function.arguments
                                        ),
                                    },
                                }
                                for tc in assistant_message.tool_calls
                            ],
                            **replay,
                        }
                    )

                    # Execute each tool call
                    for tool_call in assistant_message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args_str = tool_call.function.arguments

                        parsed_args, args_error = _parse_tool_arguments(tool_args_str)
                        if parsed_args is None:
                            outcome = ToolOutcome(text=args_error, is_error=True)
                            tool_args = {"_raw": tool_args_str}
                        else:
                            tool_args = parsed_args
                            outcome = await self._execute_tool(tool_name, tool_args)

                        tool_calls_made += 1

                        # Log tool call (full text)
                        if self.logger:
                            self.logger.log_tool_call(
                                tool_name=tool_name,
                                tool_args=tool_args,
                                tool_result=outcome.text,
                                tool_call_id=tool_call.id,
                                is_error=outcome.is_error,
                            )

                        # Record tool call info (full text)
                        step_info["tool_calls"].append(
                            {
                                "name": tool_name,
                                "arguments": tool_args,
                                "result": outcome.text,
                                "is_error": outcome.is_error,
                            }
                        )

                        # Add tool result to message history (bounded)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": self._model_content(outcome),
                            }
                        )

                elif finish_reason == "length":
                    # Reply cut off at max_tokens with no tool call: do not
                    # accept the truncated text as the final answer; ask the
                    # model to continue (as agentic-python-coder does).
                    step_info["truncated"] = True
                    messages.append(
                        {
                            "role": "assistant",
                            "content": assistant_message.content or "",
                            **replay,
                        }
                    )
                    messages.append({"role": "user", "content": TRUNCATION_NOTICE})
                    if self.logger:
                        self.logger.log_step_metadata(truncated=True)

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
                        cost=cost,
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

                # Tool-call or truncated step: record it and continue.
                steps.append(step_info)
                if self.on_step:
                    self.on_step(step_info)

                # Per-run token cap: stop looping once the cumulative
                # input+output tokens exceed the cap. Checked only after a
                # completed step, so a step is never cut in half.
                cap = self.config.max_total_tokens
                if cap > 0 and input_tokens + output_tokens > cap:
                    token_cap_reached = True
                    if self.logger:
                        self.logger.log_step_metadata(
                            token_cap_reached=True,
                            cumulative_tokens=input_tokens + output_tokens,
                            token_cap=cap,
                        )
                    break

            # Reached max steps or the token cap without a final answer
            if token_cap_reached:
                stop_msg = (
                    "[Agent stopped: per-run token cap of"
                    f" {self.config.max_total_tokens} exceeded"
                    f" (cumulative {input_tokens + output_tokens})]"
                )
            else:
                stop_msg = (
                    "[Agent reached maximum steps without providing a final answer]"
                )

            result = AgentResult(
                answer=stop_msg,
                steps=steps,
                tool_calls_made=tool_calls_made,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                max_steps_reached=not token_cap_reached,
                token_cap_reached=token_cap_reached,
            )

            if self.logger:
                self.logger.log_completion(
                    {
                        "answer": result.answer,
                        "steps_count": len(result.steps),
                        "tool_calls_made": result.tool_calls_made,
                        "max_steps_reached": result.max_steps_reached,
                        "token_cap_reached": token_cap_reached,
                        "cumulative_tokens": input_tokens + output_tokens,
                    }
                )

            return result

        except Exception as e:
            if self.logger:
                self.logger.log_error(str(e), traceback.format_exc())
            raise

    def _model_content(self, outcome: ToolOutcome) -> str:
        """The tool message the model sees: bounded text, error-prefixed."""
        text = truncate_for_model(outcome.text, self.config.max_tool_result_chars)
        return ERROR_PREFIX + text if outcome.is_error else text

    async def _setup_kernel(self) -> str | None:
        """Start the configured kernel before the first LLM call.

        When ``config.packages`` is set and a server offers python_reset,
        calls it with those packages and tracks the returned kernel_id.
        Returns an error message if the reset failed, else None.
        """
        if not (
            self.config.packages
            and self.mcp_manager
            and self.mcp_manager.has_tool("python_reset")
        ):
            return None
        packages = list(self.config.packages)
        # A kernel tracked from an earlier run of this Agent is restarted with
        # the packages rather than abandoned (the server caps live kernels).
        arguments: dict[str, Any] = {"packages": packages}
        if self._kernel_id:
            arguments["kernel_id"] = self._kernel_id
        outcome = await self.mcp_manager.call_tool("python_reset", dict(arguments))
        fallback_from = None
        if (
            "kernel_id" in arguments
            and _reset_failed(outcome)
            and "not found" in outcome.text.lower()
        ):
            # The server no longer knows the tracked kernel: start a new one.
            fallback_from = outcome.text
            self._kernel_id = None
            arguments = {"packages": packages}
            outcome = await self.mcp_manager.call_tool("python_reset", dict(arguments))
        if self.logger:
            self.logger.log_kernel_setup(
                arguments, outcome.text, outcome.is_error, fallback_from
            )
        if _reset_failed(outcome):
            return _one_line(
                f"kernel setup failed: python_reset with packages {packages}"
                f" returned an error: {outcome.text}"
            )
        self._kernel_id = _extract_kernel_id(outcome)
        return None

    async def _execute_tool(
        self, tool_name: str, tool_args: dict[str, Any]
    ) -> ToolOutcome:
        """Execute a tool (MCP, built-in read_resource, or registry tool).

        Kernel convenience: a python_reset with a non-empty ``packages`` list
        gets the configured packages first (see ``merge_packages``); a kernel
        tool call without a kernel_id gets the tracked one, so a python_reset issued by the model restarts the same
        kernel (keeping its packages) instead of creating a new one; a
        successful python_reset that returns another kernel_id switches the
        tracked kernel.
        """
        is_mcp = self.mcp_manager is not None and self.mcp_manager.has_tool(tool_name)

        if (
            is_mcp
            and tool_name in KERNEL_TOOLS
            and self._kernel_id
            and "kernel_id" not in tool_args
        ):
            tool_args["kernel_id"] = self._kernel_id

        # A model reset with its own package list must not drop the
        # configured packages (extras, local helper packages).
        if (
            is_mcp
            and tool_name == "python_reset"
            and self.config.packages
            and isinstance(tool_args.get("packages"), list)
            and tool_args["packages"]
        ):
            tool_args["packages"] = merge_packages(
                self.config.packages, tool_args["packages"]
            )

        if is_mcp:
            outcome = await self.mcp_manager.call_tool(tool_name, tool_args)
            if tool_name == "python_reset":
                kernel_id = _extract_kernel_id(outcome)
                if kernel_id:
                    self._kernel_id = kernel_id
            return outcome

        # Built-in resource access (only when no server claims the name).
        if tool_name == "read_resource" and self.mcp_manager is not None:
            uri = tool_args.get("uri")
            if not isinstance(uri, str) or not uri:
                return ToolOutcome(
                    text="read_resource requires a 'uri' string argument",
                    is_error=True,
                )
            try:
                text = await self.mcp_manager.read_resource(uri)
            except Exception as e:
                return ToolOutcome(text=f"read_resource failed: {e}", is_error=True)
            return ToolOutcome(text=text)

        tool = self.tools.get(tool_name) if self.tools else None
        if tool is None:
            return ToolOutcome(text=f"Unknown tool: {tool_name}", is_error=True)
        try:
            return tool.execute(**tool_args)
        except Exception as e:
            return ToolOutcome(text=f"Tool '{tool_name}' failed: {e}", is_error=True)
