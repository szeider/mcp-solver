"""
MCP Solver Client.

This package provides clients for interacting with MCP solvers, with options
for standard and ReAct agent implementations.
"""

from .client import main_cli as run_standard_client
from .test_client import main as run_test_client

__all__ = ["run_standard_client", "run_test_client"] 