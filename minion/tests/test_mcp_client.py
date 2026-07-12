"""Integration tests for MCPManager against the bundled test MCP server.

These exercise the full stdio MCP round-trip (spawn, initialize, list_tools,
call_tool) with no LLM involved.
"""

import json
import sys

from mcp_minion.mcp_client import MCPManager

BUNDLED_SERVER = {
    "test": {
        "command": sys.executable,
        "args": ["-m", "mcp_minion.test_mcp_server"],
    }
}


async def test_discovers_bundled_tools() -> None:
    async with MCPManager(BUNDLED_SERVER) as mgr:
        names = {t.name for t in mgr.get_tools()}
        assert {"echo", "add"} <= names
        assert mgr.has_tool("echo")
        assert not mgr.has_tool("nonexistent")


async def test_call_echo() -> None:
    async with MCPManager(BUNDLED_SERVER) as mgr:
        raw = await mgr.call_tool("echo", {"message": "hello"})
        payload = json.loads(raw)
        assert payload["result"] == "Echo: hello"


async def test_call_add() -> None:
    async with MCPManager(BUNDLED_SERVER) as mgr:
        raw = await mgr.call_tool("add", {"a": 2, "b": 3})
        payload = json.loads(raw)
        assert payload["result"] == "5"


async def test_openai_tool_schema() -> None:
    async with MCPManager(BUNDLED_SERVER) as mgr:
        tools = mgr.get_openai_tools()
        by_name = {t["function"]["name"]: t for t in tools}
        assert "echo" in by_name
        params = by_name["echo"]["function"]["parameters"]
        assert params["type"] == "object"
        assert "message" in params["properties"]
