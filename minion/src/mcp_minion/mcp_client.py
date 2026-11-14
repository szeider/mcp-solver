"""MCP client for connecting to MCP servers and using their tools."""

import asyncio
import contextlib
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import PaginatedRequestParams

# Client-side timeouts (seconds).
INIT_TIMEOUT = 30.0
RESOURCE_TIMEOUT = 30.0
DEFAULT_TOOL_TIMEOUT = 300.0
TIMEOUT_MARGIN = 30.0


def _b64_size(data: Any) -> int:
    """Decoded size in bytes of a base64 string (without decoding it)."""
    if not isinstance(data, str):
        return 0
    data = data.strip()
    return max(0, len(data) * 3 // 4 - (len(data) - len(data.rstrip("="))))


def _resource_contents_text(contents: Any) -> str:
    """Text of one resource-contents item; a blob becomes a one-line note."""
    text = getattr(contents, "text", None)
    if isinstance(text, str):
        return text
    uri = getattr(contents, "uri", "?")
    mime = getattr(contents, "mime_type", None) or "?"
    return f"[blob uri={uri} mime={mime} bytes={_b64_size(getattr(contents, 'blob', None))}]"


def _content_text(content: Any) -> str:
    """Render MCP content blocks as text (results and errors alike).

    Text blocks pass through unchanged. ``resource_link`` blocks become a
    fetchable reference line the model can follow with read_resource (Tier 2
    of a successive-revelation server). Binary blocks (image, audio, blob)
    become one short line each, never base64 or an object repr. An embedded
    text resource is a header line followed by its text.
    """
    parts = []
    for item in content:
        kind = getattr(item, "type", None)
        if kind == "text":
            parts.append(item.text)
        elif kind == "resource_link":
            desc = getattr(item, "description", None)
            suffix = f" — {desc}" if desc else ""
            parts.append(f"[resource_link] {item.uri} ({item.name}){suffix}")
        elif kind in ("image", "audio"):
            mime = getattr(item, "mime_type", None) or "?"
            size = _b64_size(getattr(item, "data", None))
            parts.append(f"[{kind} mime={mime} bytes={size}]")
        elif kind == "resource":
            res = getattr(item, "resource", None)
            if isinstance(getattr(res, "text", None), str):
                mime = getattr(res, "mime_type", None) or "?"
                uri = getattr(res, "uri", "?")
                parts.append(f"[resource uri={uri} mime={mime}]\n{res.text}")
            else:
                parts.append(_resource_contents_text(res))
        else:
            parts.append(f"[{kind or type(item).__name__} content]")
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
    """Information about an MCP resource (announced, or a resource_link)."""

    uri: str
    name: str
    description: str
    mime_type: str | None
    server_name: str


@dataclass(frozen=True)
class ToolOutcome:
    """The result of one tool call, as text for a text-only model interface.

    ``text`` is the rendered content (see ``_content_text``); ``is_error`` is
    set for MCP ``isError`` results and for client-side failures (unknown
    tool, bad arguments, timeout, transport error); ``links`` lists the
    resource_links of a successful MCP result, which read_resource accepts.
    """

    text: str
    is_error: bool = False
    links: tuple[MCPResourceInfo, ...] = ()


def _error(text: str) -> ToolOutcome:
    return ToolOutcome(text=text, is_error=True)


class MCPManager:
    """Manages MCP server connections and tool routing."""

    def __init__(
        self,
        servers_config: dict[str, dict[str, Any]],
        tool_timeout: float = DEFAULT_TOOL_TIMEOUT,
        errlog: Any = None,
    ) -> None:
        """Initialize with server configurations.

        Args:
            servers_config: Dict mapping server names to their config.
                Each config has ``command`` and optionally ``args``, ``env``
                (merged by the mcp library over its default environment
                whitelist) and ``cwd`` (the server's working directory).
            tool_timeout: Default client-side timeout (seconds) for a single
                tool call. If a call's arguments carry a numeric ``timeout``
                (e.g. python_exec), the effective timeout is at least that
                value plus a margin, so the server-side timeout fires first.
            errlog: Optional text file object receiving the spawned servers'
                stderr (default: inherit this process's stderr). Lets a CLI
                keep its console clean while preserving server logs on disk.
        """
        self.servers_config = servers_config
        self.tool_timeout = tool_timeout
        self._errlog = errlog
        self._sessions: dict[str, ClientSession] = {}
        self._tools: dict[str, MCPToolInfo] = {}  # tool_name -> MCPToolInfo
        self._resources: list[MCPResourceInfo] = []
        # server name -> non-empty instructions from its initialize result.
        self._instructions: dict[str, str] = {}
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
                # Fail loudly: a configured server that does not start would
                # leave the agent running with a silently reduced tool set,
                # producing plausible-looking but unusable runs. Close the
                # servers already started, then abort.
                failure = RuntimeError(
                    f"MCP server '{server_name}' failed to start: {e}"
                )
                await self._close_started_servers()
                raise failure from e

        return self

    async def _close_started_servers(self) -> None:
        """Tear down whatever the exit stack has already entered.

        Cleanup errors are suppressed so the original start-up failure is the
        one that reaches the caller.
        """
        stack, self._exit_stack = self._exit_stack, None
        if stack is not None:
            with contextlib.suppress(Exception):
                await stack.aclose()
        self._sessions.clear()
        self._tools.clear()
        self._resources.clear()
        self._instructions.clear()
        self._link_origins.clear()

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
        # A configured env is forwarded unchanged; the mcp library merges it
        # over its default environment whitelist (PATH, HOME, ...).
        params = StdioServerParameters(
            command=cfg["command"],
            args=cfg.get("args") or [],
            env=cfg.get("env") or None,
            cwd=cfg.get("cwd") or None,
        )

        # Enter stdio_client context; errlog (when set) receives the whole
        # child process tree's stderr instead of this process's console.
        client = (
            stdio_client(params, errlog=self._errlog)
            if self._errlog is not None
            else stdio_client(params)
        )
        transport = await self._exit_stack.enter_async_context(client)
        read_stream, write_stream = transport

        # Enter ClientSession context
        session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

        # Initialize; per the MCP lifecycle, only negotiated capabilities may
        # be used afterwards, so tool/resource discovery is capability-gated.
        init_result = await asyncio.wait_for(session.initialize(), timeout=INIT_TIMEOUT)
        capabilities = init_result.capabilities
        instructions = (init_result.instructions or "").strip()

        # Stage the server's tools first (consuming cursor pagination); commit
        # session and tools together only once every collision check has
        # passed, so a failure leaves no half-registered server behind.
        staged: dict[str, MCPToolInfo] = {}
        cursor: str | None = None
        while capabilities.tools is not None:
            tools_result = await asyncio.wait_for(
                session.list_tools(params=PaginatedRequestParams(cursor=cursor)),
                timeout=INIT_TIMEOUT,
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
                    parameters=tool.input_schema
                    or {"type": "object", "properties": {}},
                    server_name=server_name,
                )
            cursor = tools_result.next_cursor
            if not cursor:
                break

        self._sessions[server_name] = session
        self._tools.update(staged)
        if instructions:
            self._instructions[server_name] = instructions

        # Discover announced resources (optional capability, paginated).
        if capabilities.resources is None:
            return
        cursor = None
        while True:
            res_result = await asyncio.wait_for(
                session.list_resources(params=PaginatedRequestParams(cursor=cursor)),
                timeout=INIT_TIMEOUT,
            )
            for res in res_result.resources:
                self._resources.append(
                    MCPResourceInfo(
                        uri=str(res.uri),
                        name=res.name or str(res.uri),
                        description=res.description or "",
                        mime_type=res.mime_type,
                        server_name=server_name,
                    )
                )
            cursor = res_result.next_cursor
            if not cursor:
                break

    def get_resources(self) -> list[MCPResourceInfo]:
        """Get all announced MCP resources."""
        return list(self._resources)

    def get_instructions(self) -> dict[str, str]:
        """Server name -> instructions from its initialize result.

        Servers that sent no (or only blank) instructions are absent.
        """
        return dict(self._instructions)

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
            session.read_resource(uri), timeout=RESOURCE_TIMEOUT
        )
        return "\n".join(_resource_contents_text(item) for item in result.contents)

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None
    ) -> ToolOutcome:
        """Call a tool and return its outcome.

        Never raises for tool-level problems: an MCP ``isError`` result and
        client-side failures (unknown tool, non-object arguments, server not
        connected, timeout, transport error) come back with ``is_error`` set.

        Args:
            name: Tool name
            arguments: Tool arguments (None counts as no arguments)
        """
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return _error(
                f"Tool '{name}' arguments must be a JSON object,"
                f" got {type(arguments).__name__}"
            )

        tool_info = self._tools.get(name)
        if not tool_info:
            return _error(f"Unknown MCP tool: {name}")

        session = self._sessions.get(tool_info.server_name)
        if not session:
            return _error(f"MCP server '{tool_info.server_name}' not connected")

        try:
            result = await asyncio.wait_for(
                session.call_tool(name, arguments),
                timeout=self._effective_timeout(arguments),
            )
        except TimeoutError:
            return _error(f"Tool '{name}' timed out")
        except Exception as e:
            return _error(f"Tool '{name}' failed: {e}")

        text = _content_text(result.content)
        if result.is_error:
            return _error(text)

        # Remember which server produced each resource_link so later
        # read_resource calls route to the origin server only. Error results
        # are excluded: a link echoed inside an error payload must not widen
        # the readable set.
        links = []
        for item in result.content:
            if getattr(item, "type", None) == "resource_link":
                uri = str(item.uri)
                self._link_origins[uri] = tool_info.server_name
                links.append(
                    MCPResourceInfo(
                        uri=uri,
                        name=getattr(item, "name", None) or uri,
                        description=getattr(item, "description", None) or "",
                        mime_type=getattr(item, "mime_type", None),
                        server_name=tool_info.server_name,
                    )
                )
        return ToolOutcome(text=text, links=tuple(links))

    def _effective_timeout(self, arguments: dict[str, Any] | None) -> float:
        """Client-side timeout for one tool call.

        At least ``tool_timeout``; if the call itself carries a numeric
        ``timeout`` argument (e.g. python_exec), allow that plus a margin so
        the server-side timeout always fires first.
        """
        timeout = self.tool_timeout
        if not isinstance(arguments, dict):
            return timeout
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
