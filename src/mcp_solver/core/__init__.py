"""
MCP Solver - A constraint solving server for MCP.
"""

# Define the version
__version__ = "2.3.0"

# Import core components
from .base_manager import SolverManager
from .base_model_manager import BaseModelManager

# Import server functionality
from .server import main, serve


__all__ = ["SolverManager", "BaseModelManager", "__version__", "main", "serve"]
