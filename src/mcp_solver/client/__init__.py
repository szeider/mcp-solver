"""
MCP Solver Client

This package offers clients for interacting with MCP solvers.

Both a standard client and an agent based client are available.
"""

from mcp_solver.client.client import main_cli

from mcp_solver.client.react_agent import (
    create_react_agent,
)

__all__ = [
    "main_cli",
    "create_react_agent",
] 