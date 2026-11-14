"""MCP Solver: LLM-driven constraint solving on top of agentic-python-coder.

The product layer (CLI, benchmark harness, MCP server) lives in
``mcp_solver.agent`` and requires the ``[agent]`` extra; the solver templates
are package data in ``mcp_solver.templates``. Solve-time kernels do not
install this distribution, only the solver libraries of their backend.
"""

from importlib.metadata import PackageNotFoundError, version

# PyPI distribution name. The `mcp-solver` name is held by an unrelated
# upload (PEP 541 request pending), so releases go out as `agentic-mcp-solver`
# for now. Used for the version lookup and the install hint of the CLI.
DIST_NAME = "agentic-mcp-solver"

try:
    __version__ = version(DIST_NAME)
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"
