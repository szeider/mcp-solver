"""
MCP Solver Client

This package offers clients for interacting with MCP solvers.

Both a standard client and an agent based client are available.
"""

from mcp_solver.client.client import main_cli


# The custom react_agent has been removed, now using built-in LangGraph implementation

__all__ = [
    "main_cli",
]
