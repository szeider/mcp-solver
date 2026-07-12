"""mcp-minion: a minimal ReAct agent over MCP servers via OpenRouter."""

from importlib.metadata import PackageNotFoundError, version

from mcp_minion.agent import Agent, AgentConfig, AgentResult
from mcp_minion.artifacts import extract_last_submission
from mcp_minion.logging import RunLogger
from mcp_minion.mcp_client import MCPManager, MCPResourceInfo, MCPToolInfo
from mcp_minion.tools import Tool, ToolRegistry, create_default_registry

try:
    __version__ = version("mcp-minion")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentResult",
    "MCPManager",
    "MCPResourceInfo",
    "MCPToolInfo",
    "RunLogger",
    "Tool",
    "ToolRegistry",
    "__version__",
    "create_default_registry",
    "extract_last_submission",
]
