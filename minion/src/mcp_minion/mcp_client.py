"""MCP client for connecting to MCP servers and using their tools."""

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import AnyUrl

# Client-side timeouts (seconds).
INIT_TIMEOUT = 30.0
RESOURCE_TIMEOUT = 30.0
DEFAULT_TOOL_TIMEOUT = 300.0
TIMEOUT_MARGIN = 30.0


def _content_text(content: Any) -> str:
    """Join the text of MCP content items (used for results and errors alike).

    ``resource_link`` items are rendered as a fetchable reference so the model
    can follow them with the read_resource tool (Tier 2 of a
    successive-revelation server).
    """
    parts = []
    for item in content:
        if getattr(item, "type", None) == "resource_link":
            desc = getattr(item, "description", None)
            suffix = f" — {desc}" if desc else ""
            parts.append(f"[resource_link] {item.uri} ({item.name}){suffix}")
        elif hasattr(item, "text"):
            parts.append(item.text)
        else:
            parts.append(str(item))
    return "\n".join(parts)


@dataclass
class MCPToolInfo:
    """Information about an MCP tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    server_name: str


@dataclass
class MCPResourceInfo:
    """Information about an MCP resource."""

    uri: str
    name: str
    description: str
    mime_type: str | None
    server_name: str


class MCPManager:
    """Manages MCP server connections and tool routing."""

    def __init__(
        self,
        servers_config: dict[str, dict[str, Any]],
        tool_timeout: float = DEFAULT_TOOL_TIMEOUT,
    ) -> None:
        """Initialize with server configurations.

        Args:
            servers_config: Dict mapping server names to their config.
                Each config has: command, args (optional), env (optional)
            tool_timeout: Default client-side timeout (seconds) for a single
                tool call. If a call's arguments carry a numeric ``timeout``
                (e.g. python_exec), the effective timeout is at least that
                value plus a margin, so the server-side timeout fires first.
        """
        self.servers_config = servers_config
        self.tool_timeout = tool_timeout
        self._sessions: dict[str, ClientSession] = {}
        self._tools: dict[str, MCPToolInfo] = {}  # tool_name -> MCPToolInfo
        self._resources: list[MCPResourceInfo] = []
        # uri -> server that produced it as a resource_link in a tool result.
        self._link_origins: dict[str, str] = {}
        self._exit_stack: AsyncExitStack | None = None

    async def __aenter__(self) -> "MCPManager":
        """Start all MCP servers and discover tools."""
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        for server_name, cfg in self.servers_config.items():
            try:
                await self._start_server(server_name, cfg)
            except Exception as e:
                # Log error but continue with other servers. Stderr, never
                # stdout: when the manager runs inside a stdio MCP server,
                # stdout is the protocol channel.
                print(
                    f"Warning: Failed to start MCP server '{server_name}': {e}",
                    file=sys.stderr,
                )

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Stop all MCP servers."""
        if self._exit_stack:
            await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)

    async def _start_server(self, server_name: str, cfg: dict[str, Any]) -> None:
        """Start a single MCP server and discover its tools."""
        # Merge custom env with system env (custom takes precedence)
        server_env = {**os.environ, **cfg["env"]} if cfg.get("env") else None

        params = StdioServerParameters(
            command=cfg["command"],
            args=cfg.get("args", []),
            env=server_env,
        )

        # Enter stdio_client context
        transport = await self._exit_stack.enter_async_context(stdio_client(params))
        read_stream, write_stream = transport

        # Enter ClientSession context
        session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

        # Initialize; per the MCP lifecycle, only negotiated capabilities may
        # be used afterwards, so tool/resource discovery is capability-gated.
        init_result = await asyncio.wait_for(session.initialize(), timeout=INIT_TIMEOUT)
        capabilities = init_result.capabilities

        # Stage the server's tools first (consuming cursor pagination); commit
        # session and tools together only once every collision check has
        # passed, so a failure leaves no half-registered server behind.
        staged: dict[str, MCPToolInfo] = {}
        cursor: str | None = None
        while capabilities.tools is not None:
            tools_result = await asyncio.wait_for(
                session.list_tools(cursor=cursor), timeout=INIT_TIMEOUT
            )
            for tool in tools_result.tools:
                if tool.name in self._tools or tool.name in staged:
                    owner = (
                        self._tools[tool.name].server_name
                        if tool.name in self._tools
                        else server_name
                    )
                    raise ValueError(
                        f"Tool name collision: '{tool.name}' from server "
                        f"'{server_name}' conflicts with tool from server '{owner}'"
                    )
                staged[tool.name] = MCPToolInfo(
                    name=tool.name,
                    description=tool.description or "",
                    parameters=tool.inputSchema or {"type": "object", "properties": {}},
                    server_name=server_name,
                )
            cursor = tools_result.nextCursor
            if not cursor:
                break

        self._sessions[server_name] = session
        self._tools.update(staged)

        # Discover announced resources (optional capability, paginated).
        if capabilities.resources is None:
            return
        cursor = None
        while True:
            res_result = await asyncio.wait_for(
                session.list_resources(cursor=cursor), timeout=INIT_TIMEOUT
            )
            for res in res_result.resources:
                self._resources.append(
                    MCPResourceInfo(
                        uri=str(res.uri),
                        name=res.name or str(res.uri),
                        description=res.description or "",
                        mime_type=res.mimeType,
                        server_name=server_name,
                    )
                )
            cursor = res_result.nextCursor
            if not cursor:
                break

    def get_resources(self) -> list[MCPResourceInfo]:
        """Get all announced MCP resources."""
        return list(self._resources)

    async def read_resource(self, uri: str) -> str:
        """Read a resource by URI and return its text content.

        Constrained fetch, per the MCP host guidance: only URIs a server has
        announced via resources/list, or produced itself as a resource_link
        in one of its tool results, can be read — and each read is routed to
        that origin server only (no probing of other servers).
        """
        owner = next((r.server_name for r in self._resources if r.uri == uri), None)
        if owner is None:
            owner = self._link_origins.get(uri)
        if owner is None:
            raise ValueError(
                f"unknown resource URI {uri!r}: only announced resources and"
                " resource_links received in tool results can be read"
            )
        session = self._sessions.get(owner)
        if session is None:
            raise ValueError(f"MCP server '{owner}' not connected")

        result = await asyncio.wait_for(
            session.read_resource(AnyUrl(uri)), timeout=RESOURCE_TIMEOUT
        )
        parts = []
        for item in result.contents:
            if hasattr(item, "text"):
                parts.append(item.text)
            else:
                parts.append(str(item))
        return "\n".join(parts)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call a tool and return the result as a JSON string.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            JSON string with result or error
        """
        tool_info = self._tools.get(name)
        if not tool_info:
            return json.dumps({"error": f"Unknown MCP tool: {name}"})

        session = self._sessions.get(tool_info.server_name)
        if not session:
            return json.dumps(
                {"error": f"MCP server '{tool_info.server_name}' not connected"}
            )

        try:
            result = await asyncio.wait_for(
                session.call_tool(name, arguments),
                timeout=self._effective_timeout(arguments),
            )

            if result.isError:
                return json.dumps({"error": _content_text(result.content)})

            # Remember which server produced each resource_link so later
            # read_resource calls route to the origin server only. Error
            # results are excluded: a link echoed inside an error payload
            # must not widen the readable set.
            for item in result.content:
                if getattr(item, "type", None) == "resource_link":
                    self._link_origins[str(item.uri)] = tool_info.server_name

            return json.dumps({"result": _content_text(result.content)})

        except TimeoutError:
            return json.dumps({"error": f"Tool '{name}' timed out"})
        except Exception as e:
            return json.dumps({"error": f"Tool '{name}' failed: {e}"})

    def _effective_timeout(self, arguments: dict[str, Any]) -> float:
        """Client-side timeout for one tool call.

        At least ``tool_timeout``; if the call itself carries a numeric
        ``timeout`` argument (e.g. python_exec), allow that plus a margin so
        the server-side timeout always fires first.
        """
        timeout = self.tool_timeout
        arg_timeout = arguments.get("timeout")
        if isinstance(arg_timeout, int | float) and not isinstance(arg_timeout, bool):
            timeout = max(timeout, float(arg_timeout) + TIMEOUT_MARGIN)
        return timeout

    def get_tools(self) -> list[MCPToolInfo]:
        """Get all discovered MCP tools."""
        return list(self._tools.values())

    def get_openai_tools(self) -> list[dict[str, Any]]:
        """Get all MCP tools in OpenAI format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": self._sanitize_schema(tool.parameters),
                },
            }
            for tool in self._tools.values()
        ]

    def _sanitize_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Sanitize JSON Schema for OpenAI compatibility."""
        # Ensure required fields
        result = dict(schema)
        if "type" not in result:
            result["type"] = "object"
        if "properties" not in result:
            result["properties"] = {}
        return result

    def has_tool(self, name: str) -> bool:
        """Check if a tool exists."""
        return name in self._tools
