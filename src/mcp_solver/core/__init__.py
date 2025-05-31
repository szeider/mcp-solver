"""
MCP Solver - A constraint solving server for MCP.
"""

# Define the version
__version__ = "2.3.0"

# Import core components
from .base_manager import SolverManager

# Import server functionality
from .server import main, serve


__all__ = ["SolverManager", "__version__", "main", "serve"]
