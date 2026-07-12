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


# --- client-side tool-call timeout -----------------------------------------


def test_effective_timeout_default() -> None:
    manager = MCPManager({})
    assert manager._effective_timeout({}) == 300.0


def test_effective_timeout_respects_arg_timeout() -> None:
    manager = MCPManager({})
    # A long python_exec timeout extends the client-side bound (plus margin).
    assert manager._effective_timeout({"timeout": 400}) == 430.0
    # A short one does not shrink it below tool_timeout.
    assert manager._effective_timeout({"timeout": 30}) == 300.0


def test_effective_timeout_custom_and_bad_arg() -> None:
    manager = MCPManager({}, tool_timeout=60.0)
    assert manager._effective_timeout({}) == 60.0
    assert manager._effective_timeout({"timeout": "soon"}) == 60.0
    assert manager._effective_timeout({"timeout": 90}) == 120.0


# --- resource discovery and reading ----------------------------------------


async def test_discovers_and_reads_bundled_resource() -> None:
    async with MCPManager(BUNDLED_SERVER) as manager:
        resources = manager.get_resources()
        assert [r.uri for r in resources] == ["test://greeting"]
        assert resources[0].name == "greeting"
        assert resources[0].server_name == "test"
        text = await manager.read_resource("test://greeting")
        assert text == "Hello from the test resource."


async def test_read_unknown_uri_is_rejected() -> None:
    # Constrained fetch: neither announced nor received as a resource_link.
    async with MCPManager(BUNDLED_SERVER) as manager:
        import pytest

        with pytest.raises(ValueError, match="unknown resource URI"):
            await manager.read_resource("test://absent")


async def test_read_resource_without_servers_raises() -> None:
    import pytest

    manager = MCPManager({})
    with pytest.raises(ValueError, match="unknown resource URI"):
        await manager.read_resource("test://greeting")


async def test_resource_link_origin_allows_read() -> None:
    # A URI learned from a resource_link (not announced) is readable and
    # routed to its origin server.
    async with MCPManager(BUNDLED_SERVER) as manager:
        manager._resources.clear()
        manager._link_origins["test://greeting"] = "test"
        text = await manager.read_resource("test://greeting")
        assert text == "Hello from the test resource."


def test_content_text_renders_resource_links() -> None:
    from mcp.types import ResourceLink, TextContent
    from pydantic import AnyUrl

    from mcp_minion.mcp_client import _content_text

    content = [
        TextContent(type="text", text="solution"),
        ResourceLink(
            type="resource_link",
            uri=AnyUrl("app://runs/1/code.py"),
            name="code.py",
            description="the program",
        ),
    ]
    text = _content_text(content)
    assert "solution" in text
    assert "[resource_link] app://runs/1/code.py (code.py) — the program" in text
