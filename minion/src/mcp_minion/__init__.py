"""mcp-minion: a minimal ReAct agent over MCP servers via OpenRouter."""

from importlib.metadata import PackageNotFoundError, version

from mcp_minion.agent import Agent, AgentConfig, AgentResult
from mcp_minion.artifacts import extract_last_submission
from mcp_minion.env import find_api_key
from mcp_minion.logging import RunLogger
from mcp_minion.mcp_client import MCPManager, MCPResourceInfo, MCPToolInfo, ToolOutcome
from mcp_minion.tools import Tool, ToolRegistry, create_default_registry

try:
    __version__ = version("mcp-minion")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    # A sentinel, not a release number, so it cannot go stale.
    __version__ = "0+unknown"

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentResult",
    "MCPManager",
    "MCPResourceInfo",
    "MCPToolInfo",
    "RunLogger",
    "Tool",
    "ToolOutcome",
    "ToolRegistry",
    "__version__",
    "create_default_registry",
    "extract_last_submission",
    "find_api_key",
]
