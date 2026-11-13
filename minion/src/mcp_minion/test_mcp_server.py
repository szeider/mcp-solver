"""Minimal MCP server for testing - provides echo/add tools and a resource."""

import asyncio
import importlib.metadata

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    ListResourcesResult,
    ListToolsResult,
    ReadResourceResult,
    Resource,
    TextContent,
    TextResourceContents,
    Tool,
)

_MCP_V2 = int(importlib.metadata.version("mcp").split(".")[0]) >= 2

GREETING_URI = "test://greeting"
GREETING_TEXT = "Hello from the test resource."


async def list_resources() -> list[Resource]:
    """Return available resources."""
    return [
        Resource(
            uri=GREETING_URI,
            name="greeting",
            description="A test greeting resource",
            mimeType="text/plain",
        )
    ]


async def read_resource(uri) -> str:
    """Serve a resource by URI."""
    if str(uri) == GREETING_URI:
        return GREETING_TEXT
    raise ValueError(f"Unknown resource: {uri}")


async def list_tools() -> list[Tool]:
    """Return available tools."""
    return [
        Tool(
            name="echo",
            description="Echo back the input message",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Message to echo"},
                },
                "required": ["message"],
            },
        ),
        Tool(
            name="add",
            description="Add two numbers",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        ),
    ]


async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute a tool."""
    if name == "echo":
        message = arguments.get("message", "")
        return [TextContent(type="text", text=f"Echo: {message}")]
    elif name == "add":
        a = arguments.get("a", 0)
        b = arguments.get("b", 0)
        result = a + b
        return [TextContent(type="text", text=str(result))]
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _on_list_resources(ctx, params) -> ListResourcesResult:
    return ListResourcesResult(resources=await list_resources())


async def _on_read_resource(ctx, params) -> ReadResourceResult:
    text = await read_resource(params.uri)
    return ReadResourceResult(
        contents=[
            TextResourceContents(uri=params.uri, text=text, mime_type="text/plain")
        ]
    )


async def _on_list_tools(ctx, params) -> ListToolsResult:
    return ListToolsResult(tools=await list_tools())


async def _on_call_tool(ctx, params) -> CallToolResult:
    return CallToolResult(content=await call_tool(params.name, params.arguments or {}))


if _MCP_V2:
    server = Server(
        "test-server",
        on_list_resources=_on_list_resources,
        on_read_resource=_on_read_resource,
        on_list_tools=_on_list_tools,
        on_call_tool=_on_call_tool,
    )
else:
    server = Server("test-server")
    server.list_resources()(list_resources)
    server.read_resource()(read_resource)
    server.list_tools()(list_tools)
    server.call_tool()(call_tool)


async def main() -> None:
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
