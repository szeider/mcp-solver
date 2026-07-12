"""MCP client for connecting to MCP servers and using their tools."""

import asyncio
import json
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@dataclass
class MCPToolInfo:
    """Information about an MCP tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    server_name: str


class MCPManager:
    """Manages MCP server connections and tool routing."""

    def __init__(self, servers_config: dict[str, dict[str, Any]]) -> None:
        """Initialize with server configurations.

        Args:
            servers_config: Dict mapping server names to their config.
                Each config has: command, args (optional), env (optional)
        """
        self.servers_config = servers_config
        self._sessions: dict[str, ClientSession] = {}
        self._tools: dict[str, MCPToolInfo] = {}  # tool_name -> MCPToolInfo
        self._exit_stack: AsyncExitStack | None = None

    async def __aenter__(self) -> "MCPManager":
        """Start all MCP servers and discover tools."""
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        for server_name, cfg in self.servers_config.items():
            try:
                await self._start_server(server_name, cfg)
            except Exception as e:
                # Log error but continue with other servers
                print(f"Warning: Failed to start MCP server '{server_name}': {e}")

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

        # Initialize and discover tools with timeout
        await asyncio.wait_for(session.initialize(), timeout=30.0)
        self._sessions[server_name] = session

        # Discover tools
        tools_result = await asyncio.wait_for(session.list_tools(), timeout=30.0)

        for tool in tools_result.tools:
            if tool.name in self._tools:
                raise ValueError(
                    f"Tool name collision: '{tool.name}' from server '{server_name}' "
                    f"conflicts with tool from server '{self._tools[tool.name].server_name}'"
                )

            self._tools[tool.name] = MCPToolInfo(
                name=tool.name,
                description=tool.description or "",
                parameters=tool.inputSchema or {"type": "object", "properties": {}},
                server_name=server_name,
            )

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
                session.call_tool(name, arguments), timeout=60.0
            )

            if result.isError:
                return json.dumps({"error": str(result.content)})

            # Extract text from content items
            text_parts = []
            for content in result.content:
                if hasattr(content, "text"):
                    text_parts.append(content.text)
                else:
                    text_parts.append(str(content))

            return json.dumps({"result": "\n".join(text_parts)})

        except TimeoutError:
            return json.dumps({"error": f"Tool '{name}' timed out"})
        except Exception as e:
            return json.dumps({"error": f"Tool '{name}' failed: {e}"})

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
