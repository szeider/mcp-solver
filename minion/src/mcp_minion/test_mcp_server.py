"""Minimal MCP server for testing - provides echo/add tools and a resource."""

import asyncio

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, TextContent, Tool
from pydantic import AnyUrl

server = Server("test-server")

GREETING_URI = "test://greeting"
GREETING_TEXT = "Hello from the test resource."


@server.list_resources()
async def list_resources() -> list[Resource]:
    """Return available resources."""
    return [
        Resource(
            uri=AnyUrl(GREETING_URI),
            name="greeting",
            description="A test greeting resource",
            mimeType="text/plain",
        )
    ]


@server.read_resource()
async def read_resource(uri: AnyUrl) -> str:
    """Serve a resource by URI."""
    if str(uri) == GREETING_URI:
        return GREETING_TEXT
    raise ValueError(f"Unknown resource: {uri}")


@server.list_tools()
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


@server.call_tool()
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


async def main() -> None:
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
