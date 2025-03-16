"""
PySAT solution module for extracting and formatting solutions from PySAT solvers.

This module provides functions for extracting solution data from PySAT solvers
and converting it to a standardized format.
"""

import sys
import os
from typing import Dict, Any, Optional, Union, List
import logging

# IMPORTANT: Properly import the PySAT library (not our local package)
# First, remove the current directory from the path to avoid importing ourselves
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
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
    from pysat.formula import CNF, WCNF
    from pysat.solvers import Solver
    from pysat.examples.rc2 import RC2
except ImportError:
    print("PySAT solver not found. Install with: pip install python-sat")
    sys.exit(1)

# Track the last solution
_LAST_SOLUTION = None

def export_solution(solver=None, variables=None, objective=None) -> Dict[str, Any]:
    """
    Extract and format solutions from a PySAT solver.
    
    This function collects and standardizes solution data from PySAT solvers.
    It should be called at the end of PySAT code to export the solution.
    
    Args:
        solver: PySAT Solver object or dictionary containing solution data
        variables: Dictionary mapping variable names to their values or IDs
        objective: Optional objective value for optimization problems
        
    Returns:
        Dictionary with solution data
    """
    global _LAST_SOLUTION
    
    # Initialize solution data
    solution_data = {}
    
    # Case 1: Direct dictionary data
    if isinstance(solver, dict):
        solution_data = solver
        
    # Case 2: PySAT Solver object
    elif hasattr(solver, 'get_model') and callable(getattr(solver, 'get_model')):
        # Get the model if solver reports satisfiable
        is_sat = getattr(solver, 'solve', lambda: None)()
        solution_data["satisfiable"] = bool(is_sat)
        
        if is_sat:
            model = solver.get_model()
            solution_data["model"] = model
            
            # Extract variable assignments if variables dictionary is provided
            if variables:
                solution_data["assignment"] = {
                    name: (var_id in model) if var_id > 0 else ((-var_id) not in model)
                    for name, var_id in variables.items()
                }
    
    # Case 3: RC2 MaxSAT Solver - REMOVED
    
    # Ensure the satisfiable flag matches the actual model status
    if "model" in solution_data and solution_data.get("model") is not None:
        solution_data["satisfiable"] = True
    elif "model" in solution_data and solution_data.get("model") is None:
        solution_data["satisfiable"] = False
    
    # Set the status field to match satisfiability
    solution_data["status"] = "sat" if solution_data.get("satisfiable", False) else "unsat"
    
    # Include objective value if provided
    if objective is not None:
        solution_data["objective"] = objective
    
    # Store the solution for debugging/inspection
    _LAST_SOLUTION = solution_data
    
    return solution_data 