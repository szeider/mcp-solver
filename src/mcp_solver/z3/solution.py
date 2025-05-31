"""
Z3 solution module for extracting and formatting solutions from Z3 solvers.

This module provides functions for extracting solution data from Z3 solvers
and converting it to a standardized format.
"""

import os
import sys
from typing import Any


# IMPORTANT: Properly import the Z3 library (not our local package)
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

# Now try to import Z3
try:
    import z3
except ImportError:
    print("Z3 solver not found. Install with: pip install z3-solver>=4.12.1")
    sys.exit(1)

# Track the last solution
_LAST_SOLUTION = None


def _extract_variable_values(model, variables: dict[str, Any]) -> dict[str, Any]:
    """
    Helper function to extract variable values from a Z3 model.

    Args:
        model: Z3 model
        variables: Dictionary mapping variable names to Z3 variables

    Returns:
        Dictionary with variable names and their values
    """
    result = {}

    if not model or not variables:
        return result

    for name, var in variables.items():
        try:
            # Try basic model evaluation
            val = model.eval(var, model_completion=True)
            if val is not None:
                # Convert to appropriate Python type
                if z3.is_int(val):
                    result[name] = val.as_long()
                elif z3.is_real(val):
                    # Convert rational to float
                    result[name] = float(val.as_decimal(10))
                elif z3.is_bool(val):
                    result[name] = z3.is_true(val)
                elif hasattr(z3, "is_string") and z3.is_string(val):
                    result[name] = str(val)
                else:
                    # For other types, convert to string
                    result[name] = str(val)
        except Exception as e:
            print(f"Error extracting value for {name}: {e}")
            result[name] = f"Error: {e!s}"

    return result


def _extract_objective_value(model, objective):
    """
    Helper function to extract objective value from a Z3 model.

    Args:
        model: Z3 model
        objective: Z3 expression being optimized

    Returns:
        Objective value in appropriate Python type or None if error
    """
    if not model or objective is None:
        return None

    try:
        obj_val = model.eval(objective, model_completion=True)
        if z3.is_int(obj_val):
            return obj_val.as_long()
        elif z3.is_real(obj_val):
            return float(obj_val.as_decimal(10))
        else:
            return str(obj_val)
    except Exception as e:
        print(f"Error extracting objective value: {e}")
        return f"Error: {e!s}"


def export_solution(
    solver=None,
    variables=None,
    objective=None,
    satisfiable=None,
    solution_dict=None,
    is_property_verification=False,
    property_verified=None,
) -> dict[str, Any]:
    """
    Extract and format solutions from a Z3 solver.

    This function collects and standardizes solution data from Z3 solvers.
    It should be called at the end of Z3 code to export the solution.

    Args:
        solver: Z3 Solver or Optimize object
        variables: Dictionary mapping variable names to Z3 variables
        objective: Z3 expression being optimized (optional)
        satisfiable: Explicitly override the satisfiability status (optional)
        solution_dict: Directly provide a solution dictionary (optional)
        is_property_verification: Flag indicating if this is a property verification problem (optional)
        property_verified: Boolean indicating if the property was verified (optional)

    Returns:
        Dictionary containing the solution details
    """
    global _LAST_SOLUTION

    # Initialize result dictionary
    result = {
        "satisfiable": False,
        "values": {},
        "objective": None,
        "status": "unknown",
        "output": [],
    }

    # If solution_dict is provided, use it as the base
    if solution_dict is not None and isinstance(solution_dict, dict):
        result.update(solution_dict)
        # Ensure result has all required fields
        result.setdefault("satisfiable", False)
        result.setdefault("values", {})
        result.setdefault("objective", None)
        result.setdefault("status", "unknown")
        result.setdefault("output", [])

        # Update status if satisfiable flag has been set
        if "satisfiable" in solution_dict:
            result["status"] = "sat" if solution_dict["satisfiable"] else "unsat"

        # Store result and return
        _LAST_SOLUTION = result
        return result

    # Process the solver if provided
    if solver is not None:
        # Get the solver status
        if isinstance(solver, z3.Solver) or isinstance(solver, z3.Optimize):
            status = solver.check()
            result["status"] = str(status)

            # Extract solution if satisfiable
            if status == z3.sat:
                result["satisfiable"] = True

                if variables is not None:
                    model = solver.model()
                    # Extract variable values
                    result["values"] = _extract_variable_values(model, variables)

                    # Extract objective value if present
                    if objective is not None and isinstance(solver, z3.Optimize):
                        result["objective"] = _extract_objective_value(model, objective)
        else:
            print(f"Warning: Unknown solver type: {type(solver)}")
    # Handle case with variables but no solver
    elif variables is not None:
        print(
            "Warning: Variables provided but no solver. Only variable names will be included."
        )
        result["values"] = {name: None for name in variables}

    # Override satisfiability if explicitly provided
    if satisfiable is not None:
        result["satisfiable"] = bool(satisfiable)
        # Update status to match satisfiability flag
        result["status"] = "sat" if result["satisfiable"] else "unsat"

    # Handle property verification cases
    if is_property_verification:
        # If property_verified is explicitly provided, use it
        if property_verified is not None:
            # Store the property verification result explicitly
            result["values"]["property_verified"] = bool(property_verified)

            # For property verification:
            # - If we found a counterexample (property not verified), the solver is satisfiable
            # - If the property is verified for all cases, there's no counterexample, so the negation is unsatisfiable
            if property_verified:
                result["output"].append("Property verified successfully.")
            else:
                result["output"].append(
                    "Property verification failed. Counterexample found."
                )
                # Ensure satisfiability is set correctly for counterexample
                result["satisfiable"] = True
                result["status"] = "sat"
        else:
            # Infer property verification status from solver result
            # If solver returned unsat, property is verified (no counterexample)
            # If solver returned sat, property is not verified (counterexample found)
            property_verified = result["status"] == "unsat"
            result["values"]["property_verified"] = property_verified

            if property_verified:
                result["output"].append("Property verified successfully.")
            else:
                result["output"].append(
                    "Property verification failed. Counterexample found."
                )
                # Ensure satisfiability is set correctly for counterexample
                result["satisfiable"] = True
                result["status"] = "sat"
    else:
        # For regular constraint solving, add appropriate output messages
        if result["satisfiable"]:
            result["output"].append("Solution found.")
        else:
            result["output"].append(
                "No solution exists that satisfies all constraints."
            )

    # Store result in global variable for the environment to find
    _LAST_SOLUTION = result

    return result
