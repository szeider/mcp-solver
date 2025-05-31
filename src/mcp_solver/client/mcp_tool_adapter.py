"""
MCP Tool Adapter for MCP Solver

This module provides functionality to convert MCP tools to LangChain tools.
The code is based on langchain_mcp_adapters.tools but has been modified to
ensure compatibility with mcp>=1.5.
"""

from typing import Any, Union

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from mcp import ClientSession
from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
    Tool as MCPTool,
)


NonTextContent = Union[ImageContent, EmbeddedResource]


def _convert_call_tool_result(
    call_tool_result: CallToolResult,
) -> tuple[str | list[str], list[NonTextContent] | None]:
    """
    Convert a CallToolResult to a format suitable for LangChain tools.

    Args:
        call_tool_result: The result from an MCP tool call

    Returns:
        A tuple of (text_content, non_text_contents)
    """
    text_contents: list[TextContent] = []
    non_text_contents = []
    for content in call_tool_result.content:
        if isinstance(content, TextContent):
            text_contents.append(content)
        else:
            non_text_contents.append(content)

    tool_content: str | list[str] = [content.text for content in text_contents]
    if len(text_contents) == 1:
        tool_content = tool_content[0]

    if call_tool_result.isError:
        raise ToolException(tool_content)

    return tool_content, non_text_contents or None


def convert_mcp_tool_to_langchain_tool(
    session: ClientSession,
    tool: MCPTool,
) -> BaseTool:
    """
    Convert an MCP tool to a LangChain tool.

    NOTE: This tool can be executed only in a context of an active MCP client session.

    Args:
        session: MCP client session
        tool: MCP tool to convert

    Returns:
        A LangChain tool
    """

    async def call_tool(
        **arguments: Any,
    ) -> tuple[str | list[str], list[NonTextContent] | None]:
        call_tool_result = await session.call_tool(tool.name, arguments)
        return _convert_call_tool_result(call_tool_result)

    return StructuredTool(
        name=tool.name,
        description=tool.description or "",
        args_schema=tool.inputSchema,
        coroutine=call_tool,
        response_format="content_and_artifact",
    )


async def load_mcp_tools(session: ClientSession) -> list[BaseTool]:
    """
    Load all available MCP tools and convert them to LangChain tools.

    Args:
        session: The MCP client session

    Returns:
        A list of LangChain tools
    """
    tools_response = await session.list_tools()
    return [
        convert_mcp_tool_to_langchain_tool(session, tool)
        for tool in tools_response.tools
    ]
