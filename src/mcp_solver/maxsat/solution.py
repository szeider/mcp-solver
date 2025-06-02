"""
MaxSAT solution module for extracting and formatting solutions from MaxSAT solvers.

This module provides a simplified solution export function for MaxSAT problems.
"""

import logging
import os
import sys
from typing import Any, Union

# IMPORTANT: Properly import the PySAT library (not our local package)
# First, remove the current directory from the path to avoid importing ourselves
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if current_dir in sys.path:
    sys.path.remove(current_dir)
if parent_dir in sys.path:
    sys.path.remove(parent_dir)

# Add site-packages to the front of the path
import site

site_packages = site.getsitepackages()
for p in reversed(site_packages):
    if p not in sys.path:
        sys.path.insert(0, p)

# Now try to import PySAT
try:
    from pysat.examples.rc2 import RC2
    from pysat.formula import WCNF
except ImportError:
    print("PySAT solver not found. Install with: pip install python-sat")
    sys.exit(1)

# Set up logger
logger = logging.getLogger(__name__)

# Track the last solution - make it global and accessible
_LAST_SOLUTION = None


def export_solution(data: dict[str, Any]) -> dict[str, Any]:
    """
    Export solution data from a MaxSAT optimization problem.
    
    This is a simplified version that matches PySAT's export pattern.
    Just pass a dictionary with your solution data.
    
    Args:
        data: Dictionary with solution data
        
    Returns:
        The same dictionary (for consistency with PySAT)
        
    Example:
        ```python
        export_solution({
            "satisfiable": True,
            "cost": solver.cost,
            "assignment": {...}
        })
        ```
    """
    global _LAST_SOLUTION
    
    # Ensure basic fields exist
    if "satisfiable" not in data:
        data["satisfiable"] = True
        
    # Store and return
    _LAST_SOLUTION = data
    return data


# Keep the old function name as an alias during transition
export_maxsat_solution = export_solution