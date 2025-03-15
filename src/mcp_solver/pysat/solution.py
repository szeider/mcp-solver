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
    from pysat.formula import CNF
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
        objective: Value of the optimization objective (for MaxSAT problems)
        
    Returns:
        Dictionary containing the solution details
    """
    # Declare global variables first
    global _LAST_SOLUTION
    
    # Initialize result dictionary
    result = {
        "satisfiable": False,
        "values": {},
        "objective": None,
        "status": "unknown"
    }
    
    # Log what we're exporting for debugging
    logging.getLogger(__name__).debug(f"Exporting solution: solver={type(solver)}, variables={variables is not None}")
    
    # Case 1: Direct variable dictionary (no solver)
    if variables is not None and solver is None:
        result["satisfiable"] = True
        result["status"] = "sat"
        result["values"] = variables
        if objective is not None:
            result["objective"] = objective
        
        # Store result and return
        _LAST_SOLUTION = result
        logging.getLogger(__name__).debug(f"Exported solution (case 1): {result}")
        return result
    
    # Case 2: Solver instance (PySAT Solver or RC2)
    if solver is not None:
        # For standard SAT solvers
        if isinstance(solver, Solver):
            # Check if the problem was solved
            if hasattr(solver, '_solved') and solver._solved:
                # BUG FIX: Get the satisfiability result directly
                is_sat = solver.get_model() is not None
                
                # Log the actual solver satisfiability result for debugging
                logging.getLogger(__name__).debug(f"PySAT solver result - is_sat: {is_sat}, model: {solver.get_model() is not None}")
                
                model = solver.get_model()
                result["satisfiable"] = is_sat
                result["status"] = "sat" if is_sat else "unsat"
                
                # Extract variable values if we have a mapping
                if variables is not None and model is not None:
                    for var_name, var_id in variables.items():
                        # Handle both positive and negative variable IDs
                        var_id_abs = abs(var_id)
                        # Find variable value in model (if within range)
                        if var_id_abs <= len(model):
                            # Model contains a list of literals, where:
                            # Positive numbers mean variable is True
                            # Negative numbers mean variable is False
                            # We find the variable in the model by its absolute ID
                            # and check its sign to determine True/False
                            var_value = var_id_abs in model
                            if var_id < 0:  # If variable ID is negative, negate the value
                                var_value = not var_value
                            result["values"][var_name] = var_value
                
                # Store objective value if provided
                if objective is not None:
                    result["objective"] = objective
        
        # For MaxSAT solvers (RC2)
        elif isinstance(solver, RC2):
            model = solver.model
            cost = solver.cost if hasattr(solver, 'cost') else None
            
            # BUG FIX: Get the satisfiability result directly
            is_sat = model is not None
            
            # Log the actual solver satisfiability result for debugging
            logging.getLogger(__name__).debug(f"RC2 solver result - is_sat: {is_sat}, model: {model is not None}")
            
            result["satisfiable"] = is_sat
            result["status"] = "sat" if is_sat else "unsat"
            
            # Extract variable values if we have a mapping
            if variables is not None and model is not None:
                for var_name, var_id in variables.items():
                    var_id_abs = abs(var_id)
                    # Find variable value in model (if within range)
                    if var_id_abs <= len(model):
                        var_value = var_id_abs in model
                        if var_id < 0:  # If variable ID is negative, negate the value
                            var_value = not var_value
                        result["values"][var_name] = var_value
            
            # Store objective/cost value
            if cost is not None:
                result["objective"] = cost
            elif objective is not None:
                result["objective"] = objective
    
    # Store result for retrieval by environment module
    _LAST_SOLUTION = result
    
    # Set solution in calling namespace to make it available to exec
    import inspect
    caller_globals = inspect.currentframe().f_back.f_globals
    caller_locals = inspect.currentframe().f_back.f_locals
    caller_locals["solution"] = result
    
    # Log the final exported solution for debugging
    logging.getLogger(__name__).debug(f"Exported solution (final): {result}")
    
    return result 