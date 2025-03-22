"""
MCP Solver Client.

This package provides clients for interacting with MCP solvers, with options
for standard and agent implementations.
"""

from .client import main_cli as run_standard_client

__all__ = ["run_standard_client"] 