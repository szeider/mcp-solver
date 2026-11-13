"""MCP Solver: LLM-driven constraint solving on top of agentic-python-coder.

The base package is the solver helper library (``mcp_solver.helpers``),
importable inside solve-time kernels. The product layer (CLI, templates,
MCP server) lives in ``mcp_solver.agent`` and requires the ``[agent]``
extra. Nothing is imported eagerly here so that the base package works
without any solver library installed.
"""

from importlib.metadata import PackageNotFoundError, version

# PyPI distribution name. The `mcp-solver` name is held by an unrelated
# upload (PEP 541 request pending), so releases go out as `agentic-mcp-solver`
# for now. Solve kernels fetch the helper library under this name.
DIST_NAME = "agentic-mcp-solver"

try:
    __version__ = version(DIST_NAME)
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"
