"""
Pytest configuration and shared fixtures.
"""
import pytest
import os
import sys
from pathlib import Path

# Ensure the src directory is in the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

@pytest.fixture
def mzn_manager():
    """Create a MiniZinc model manager for testing."""
    from src.mcp_solver.mzn.model_manager import MiniZincModelManager
    manager = MiniZincModelManager(lite_mode=True)
    yield manager
    # Clean up after tests

@pytest.fixture
def pysat_manager():
    """Create a PySAT model manager for testing."""
    from src.mcp_solver.pysat.model_manager import PySATModelManager
    manager = PySATModelManager(lite_mode=True)
    yield manager
    # Clean up after tests

@pytest.fixture
def z3_manager():
    """Create a Z3 model manager for testing."""
    from src.mcp_solver.z3.model_manager import Z3ModelManager
    manager = Z3ModelManager(lite_mode=True)
    yield manager
    # Clean up after tests 