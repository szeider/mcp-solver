"""
MCP Solver - A constraint solving server for MCP.
"""

# Define the version
__version__ = "2.3.0"

# Import core components
from .base_manager import SolverManager
from .memo import MemoManager

# Import solver implementations
from .mzn.model_manager import MiniZincModelManager

# Import server functionality 
from .server import serve, main

__all__ = ['serve', 'main', 'SolverManager', 'MiniZincModelManager', 'MemoManager', '__version__']
