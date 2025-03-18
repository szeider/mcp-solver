"""
Z3 solution module for extracting and formatting solutions from Z3 solvers.

This module provides functions for extracting solution data from Z3 solvers
and converting it to a standardized format.
"""

import sys
import os
from typing import Dict, Any, Optional, Union, List

# IMPORTANT: Properly import the Z3 library (not our local package)
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

# Now try to import Z3
try:
    import z3
except ImportError:
    print("Z3 solver not found. Install with: pip install z3-solver>=4.12.1")
    sys.exit(1)

# Track the last solution
_LAST_SOLUTION = None

def export_solution(solver=None, variables=None, objective=None) -> Dict[str, Any]:
    """
    Extract and format solutions from a Z3 solver.
    
    This function collects and standardizes solution data from Z3 solvers.
    It should be called at the end of Z3 code to export the solution.
    
    Args:
        solver: Z3 Solver or Optimize object
        variables: Dictionary mapping variable names to Z3 variables
        objective: Z3 expression being optimized (optional)
        
    Returns:
        Dictionary containing the solution details
    """
    # Initialize result dictionary
    result = {
        "satisfiable": False,
        "values": {},
        "objective": None,
        "status": "unknown"
    }
    
    # Validate inputs
    if not solver or not variables:
        print("Missing solver or variables")
        return result
    
    # Get the solver status
    if isinstance(solver, z3.Solver) or isinstance(solver, z3.Optimize):
        status = solver.check()
        result["status"] = str(status)
    else:
        print(f"Unknown solver type: {type(solver)}")
        return result
    
    # Extract solution if satisfiable
    if status == z3.sat:
        result["satisfiable"] = True
        model = solver.model()
        
        # Extract variable values
        for name, var in variables.items():
            try:
                # Try basic model evaluation
                val = model.eval(var, model_completion=True)
                if val is not None:
                    # Convert to appropriate Python type
                    if z3.is_int(val):
                        result["values"][name] = val.as_long()
                    elif z3.is_real(val):
                        # Convert rational to float
                        result["values"][name] = float(val.as_decimal(10))
                    elif z3.is_bool(val):
                        result["values"][name] = z3.is_true(val)
                    elif hasattr(z3, 'is_string') and z3.is_string(val):
                        result["values"][name] = str(val)
                    else:
                        # For other types, convert to string
                        result["values"][name] = str(val)
            except Exception as e:
                print(f"Error extracting value for {name}: {e}")
                result["values"][name] = f"Error: {str(e)}"
        
        # Extract objective value if present
        if objective is not None and isinstance(solver, z3.Optimize):
            try:
                obj_val = model.eval(objective, model_completion=True)
                if z3.is_int(obj_val):
                    result["objective"] = obj_val.as_long()
                elif z3.is_real(obj_val):
                    result["objective"] = float(obj_val.as_decimal(10))
                else:
                    result["objective"] = str(obj_val)
            except Exception as e:
                print(f"Error extracting objective value: {e}")
                result["objective"] = f"Error: {str(e)}"
    
    # Store result in global variable for the environment to find
    global _LAST_SOLUTION
    _LAST_SOLUTION = result
    
    # Don't try to modify caller's scope, as it doesn't work reliably with nested functions
    # Instead, the environment will retrieve the result from _LAST_SOLUTION
    
    return result 