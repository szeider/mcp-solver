"""Minimal MCP server for testing: tools, a resource and server instructions.

Tools: ``echo``, ``add``, ``lines`` (multi-line text), ``fail`` (an
``isError`` result), ``image`` (an image block) and ``link`` (a
resource_link to the greeting resource).
"""

import asyncio

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    ImageContent,
    ListResourcesResult,
    ListToolsResult,
    ReadResourceResult,
    Resource,
    ResourceLink,
    TextContent,
    TextResourceContents,
    Tool,
)

GREETING_URI = "test://greeting"
GREETING_TEXT = "Hello from the test resource."
INSTRUCTIONS = "Test server: call echo before add."
# 4 bytes of PNG signature, base64-encoded.
IMAGE_DATA = "iVBORw=="

_NO_ARGS = {"type": "object", "properties": {}}

TOOLS = [
    Tool(
        name="echo",
        description="Echo back the input message",
        input_schema={
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
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["a", "b"],
        },
    ),
    Tool(name="lines", description="Return multi-line text", input_schema=_NO_ARGS),
    Tool(name="fail", description="Return an error result", input_schema=_NO_ARGS),
    Tool(name="image", description="Return an image", input_schema=_NO_ARGS),
    Tool(name="link", description="Return a resource link", input_schema=_NO_ARGS),
]


def call_tool(name: str, arguments: dict) -> CallToolResult:
    """Execute a tool."""
    if name == "echo":
        text = f"Echo: {arguments.get('message', '')}"
    elif name == "add":
        text = str(arguments.get("a", 0) + arguments.get("b", 0))
    elif name == "lines":
        text = 'line "one"\nline two'
    elif name == "fail":
        return CallToolResult(
            content=[TextContent(type="text", text="it failed")], is_error=True
        )
    elif name == "image":
        return CallToolResult(
            content=[ImageContent(type="image", data=IMAGE_DATA, mime_type="image/png")]
        )
    elif name == "link":
        return CallToolResult(
            content=[
                ResourceLink(
                    type="resource_link",
                    uri=GREETING_URI,
                    name="greeting",
                    description="the greeting",
                )
            ]
        )
    else:
        text = f"Unknown tool: {name}"
    return CallToolResult(content=[TextContent(type="text", text=text)])


async def _on_list_resources(ctx, params) -> ListResourcesResult:
    return ListResourcesResult(
        resources=[
            Resource(
                uri=GREETING_URI,
                name="greeting",
                description="A test greeting resource",
                mime_type="text/plain",
            )
        ]
    )


async def _on_read_resource(ctx, params) -> ReadResourceResult:
    if str(params.uri) != GREETING_URI:
        raise ValueError(f"Unknown resource: {params.uri}")
    return ReadResourceResult(
        contents=[
            TextResourceContents(
                uri=params.uri, text=GREETING_TEXT, mime_type="text/plain"
            )
        ]
    )


async def _on_list_tools(ctx, params) -> ListToolsResult:
    return ListToolsResult(tools=TOOLS)


async def _on_call_tool(ctx, params) -> CallToolResult:
    return call_tool(params.name, params.arguments or {})


server = Server(
    "test-server",
    instructions=INSTRUCTIONS,
    on_list_resources=_on_list_resources,
    on_read_resource=_on_read_resource,
    on_list_tools=_on_list_tools,
    on_call_tool=_on_call_tool,
)


async def main() -> None:
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
