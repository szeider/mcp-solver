"""
MCP Solver Client module for direct interaction with mcp-solver via LLMs.
"""

from .client import main_cli as run_client
from .llm_factory import LLMFactory, ModelInfo

__all__ = ["run_client", "LLMFactory", "ModelInfo"] 