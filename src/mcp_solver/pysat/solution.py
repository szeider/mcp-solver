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

# Track the last solution - make it global and accessible
_LAST_SOLUTION = None

def export_solution(solver=None, variables=None, objective=None) -> Dict[str, Any]:
    """
    Extract and format solutions from a PySAT solver.
    
    This function collects and standardizes solution data from PySAT solvers.
    It should be called at the end of PySAT code to export the solution.
    All values in custom dictionaries are automatically extracted and made 
    available in the "values" dictionary.
    
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
        solution_data = solver.copy()
        
    # Case 2: PySAT Solver object
    elif hasattr(solver, 'get_model') and callable(getattr(solver, 'get_model')):
        # Get the model directly - solver.solve() should have been called in a direct conditional
        model = solver.get_model()
        if model is not None:
            # Solver has already been run and is satisfiable
            solution_data["satisfiable"] = True
            solution_data["model"] = model
        else:
            # If no model, it's likely unsatisfiable
            solution_data["satisfiable"] = False
        
        # Extract variable assignments if variables dictionary is provided
        if solution_data.get("satisfiable", False) and variables and "model" in solution_data:
            solution_data["assignment"] = {
                name: (var_id in solution_data["model"]) if var_id > 0 else ((-var_id) not in solution_data["model"])
                for name, var_id in variables.items()
            }
    
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
    
    # Remove any existing values dictionary as we'll create a fresh one
    if "values" in solution_data:
        del solution_data["values"]
    
    # Create a new values dictionary
    solution_data["values"] = {}
    
    # Define reserved keys that shouldn't be processed as custom dictionaries
    reserved_keys = {"satisfiable", "model", "values", "status", "objective"}
    
    # First pass: collect all keys to detect potential collisions
    all_keys = set()
    for key, value in solution_data.items():
        if key not in reserved_keys and isinstance(value, dict):
            all_keys.update(value.keys())
    
    # Count occurrences of each key to identify collisions
    key_counts = {}
    for key, value in solution_data.items():
        if key not in reserved_keys and isinstance(value, dict):
            for subkey in value.keys():
                key_counts[subkey] = key_counts.get(subkey, 0) + 1
    
    # Second pass: extract values and handle collisions
    for key, value in solution_data.items():
        if key not in reserved_keys and isinstance(value, dict):
            for subkey, subvalue in value.items():
                # Only extract leaf nodes (not nested dictionaries)
                if not isinstance(subvalue, dict):
                    if key_counts[subkey] > 1:
                        # This is a collision - prefix with parent dictionary name
                        solution_data["values"][f"{key}.{subkey}"] = subvalue
                    else:
                        # No collision - use the key directly
                        solution_data["values"][subkey] = subvalue
    
    # Add debug marker that can be used by the model manager to extract data
    # This is helpful because sometimes the _LAST_SOLUTION variable isn't shared properly
    # between modules
    logging.getLogger(__name__).debug(f"Setting _LAST_SOLUTION: {solution_data}")
    print(f"DEBUG - _LAST_SOLUTION set to: {solution_data}")
    
    # Store the solution for debugging/inspection and make it available
    # to other modules
    _LAST_SOLUTION = solution_data
    
    # Return the solution dictionary to make it accessible to the caller
    return solution_data 