"""
Z3 model manager implementation using BaseModelManager.

This module provides the Z3ModelManager class, which implements the SolverManager
interface for the Z3 SMT solver.
"""

import logging
from datetime import timedelta
from typing import Any

# Import z3 for status constant comparison
import z3

from ..core.base_model_manager import BaseModelManager
from ..core.constants import MAX_SOLVE_TIMEOUT, MIN_SOLVE_TIMEOUT
from ..core.validation import (
    ValidationError,
    get_standardized_response,
    validate_content,
    validate_python_code_safety,
    validate_timeout,
)
from .environment import execute_z3_code


class Z3ModelManager(BaseModelManager):
    """
    Z3 model manager implementation.

    This class manages Z3 models, including adding, removing, and modifying
    model items, as well as solving models and extracting solutions.
    """

    def __init__(self):
        """
        Initialize a new Z3 model manager.
        """
        super().__init__()
        self.initialized = True
        # Add a registry to store variables and solver across different scopes
        self._registry = {"variables": {}, "solver": None}
        self.logger = logging.getLogger(__name__)
        self.logger.info("Z3 model manager initialized")

    async def clear_model(self) -> dict[str, Any]:
        """
        Clear the current model.

        Returns:
            A dictionary with a message indicating the model was cleared
        """
        result = await super().clear_model()

        # Clear the registry when model is cleared
        self._registry = {"variables": {}, "solver": None}
        self.logger.info("Model cleared")

        return get_standardized_response(success=True, message="Model cleared")

    async def add_item(self, index: int, content: str) -> dict[str, Any]:
        """
        Add an item to the model at the specified index.

        Args:
            index: The index at which to add the item (0-based)
            content: The content to add

        Returns:
            A dictionary with the result of the operation
        """
        try:
            # Validate content and code safety
            validate_content(content)
            validate_python_code_safety(content)

            # First call parent's add_item to handle list operations
            result = await super().add_item(index, content)

            if not result.get("success"):
                return result

            # Get current model for response
            model = self.get_model()

            self.logger.info(f"Added item at index {index}")
            return get_standardized_response(
                success=True,
                message=f"Added item at index {index}",
                model=model,
            )

        except ValidationError as e:
            error_msg = str(e)
            self.logger.error(f"Validation error in add_item: {error_msg}")
            return get_standardized_response(
                success=False,
                message=f"Failed to add item: {error_msg}",
                error=error_msg,
            )
        except Exception as e:
            error_msg = f"Unexpected error in add_item: {e!s}"
            self.logger.error(error_msg, exc_info=True)
            return get_standardized_response(
                success=False,
                message="Failed to add item due to an internal error",
                error=error_msg,
            )

    async def delete_item(self, index: int) -> dict[str, Any]:
        """
        Delete an item from the model at the specified index.

        Args:
            index: The index of the item to delete (0-based)

        Returns:
            A dictionary with the result of the operation
        """
        # First call parent's delete_item
        result = await super().delete_item(index)

        if not result.get("success"):
            return result

        # Get current model for response
        model = self.get_model()

        self.logger.info(f"Deleted item at index {index}")
        return get_standardized_response(
            success=True,
            message=f"Deleted item at index {index}",
            model=model,
        )

    async def replace_item(self, index: int, content: str) -> dict[str, Any]:
        """
        Replace an item in the model at the specified index.

        Args:
            index: The index of the item to replace (0-based)
            content: The new content

        Returns:
            A dictionary with the result of the operation
        """
        try:
            # Validate content and code safety
            validate_content(content)
            validate_python_code_safety(content)

            # First call parent's replace_item
            result = await super().replace_item(index, content)

            if not result.get("success"):
                return result

            # Get current model for response
            model = self.get_model()

            self.logger.info(f"Replaced item at index {index}")
            return get_standardized_response(
                success=True,
                message=f"Replaced item at index {index}",
                model=model,
            )

        except ValidationError as e:
            error_msg = str(e)
            self.logger.error(f"Validation error in replace_item: {error_msg}")

            # Get current model for error response
            model = self.get_model()

            return get_standardized_response(
                success=False,
                message=f"Failed to replace item: {error_msg}",
                error=error_msg,
                model=model,
            )
        except Exception as e:
            error_msg = f"Unexpected error in replace_item: {e!s}"
            self.logger.error(error_msg, exc_info=True)

            # Get current model for error response
            model = self.get_model()

            return get_standardized_response(
                success=False,
                message="Failed to replace item due to an internal error",
                error=error_msg,
                model=model,
            )

    async def solve_model(self, timeout: timedelta) -> dict[str, Any]:
        """
        Solve the current model.

        Args:
            timeout: Timeout for the solve operation

        Returns:
            A dictionary with the result of the solve operation
        """
        try:
            # Check if model is empty
            if not self.code_items:
                return get_standardized_response(
                    success=False, message="Model is empty", error="Empty model"
                )

            # Validate timeout
            validate_timeout(timeout, MIN_SOLVE_TIMEOUT, MAX_SOLVE_TIMEOUT)

            # Get full code using parent's method
            combined_code = self._get_full_code()

            # Add wrapper for export_solution to capture variables and solver
            combined_code = (
                """
# Import the export_solution function if not already imported
try:
    from mcp_solver.z3 import export_solution as original_export_solution
except ImportError:
    # Use the existing export_solution if import fails
    original_export_solution = export_solution

def wrapped_export_solution(solver=None, variables=None, objective=None, satisfiable=None, 
                           solution_dict=None, is_property_verification=False, property_verified=None):
    # Store variables and solver in global variables for later access
    import sys
    module = sys.modules[__name__]
    if not hasattr(module, '_z3_registry'):
        module._z3_registry = {}
    
    # Store solver and variables in registry if provided
    if variables is not None:
        module._z3_registry["variables"] = variables
    if solver is not None:
        module._z3_registry["solver"] = solver
    
    # Pass through all parameters to the original function
    return original_export_solution(
        solver=solver, 
        variables=variables, 
        objective=objective,
        satisfiable=satisfiable,
        solution_dict=solution_dict,
        is_property_verification=is_property_verification,
        property_verified=property_verified
    )

export_solution = wrapped_export_solution
"""
                + combined_code
            )

            # Set timeout in seconds
            timeout_seconds = timeout.total_seconds()

            # Execute the code
            result = execute_z3_code(combined_code, timeout=timeout_seconds)

            # Store the result for later retrieval
            self.last_result = result
            self.last_solve_time = result.get("execution_time")

            # Check for error indicating missing export_solution
            if result.get("status") == "no_solution" or (
                result.get("error")
                and "No solution was exported" in result.get("error")
            ):
                # Check if there's a stored solver from previous export_solution calls
                if "_z3_registry" in globals() and globals()["_z3_registry"].get(
                    "solver"
                ):
                    # Use the stored solver and variables to generate a solution
                    stored_solver = globals()["_z3_registry"].get("solver")
                    stored_variables = globals()["_z3_registry"].get("variables", {})

                    # Execute solve with the stored solver
                    status = stored_solver.check()

                    if status == z3.sat:
                        model = stored_solver.model()
                        solution = {}

                        # Extract values for variables
                        for var_name, var in stored_variables.items():
                            try:
                                solution[var_name] = model.eval(var).as_long()
                            except:
                                try:
                                    solution[var_name] = model.eval(var).as_fraction()
                                except:
                                    try:
                                        solution[var_name] = bool(model.eval(var))
                                    except:
                                        solution[var_name] = str(model.eval(var))

                        # Update the result
                        result["status"] = "success"
                        result["solution"] = {
                            "satisfiable": True,
                            "values": solution,
                            "status": "sat",
                        }

                        # Store the solution for later retrieval
                        self.last_solution = result.get("solution")

            # For success case, also update our registry from the solution
            if result.get("solution") and result.get("solution").get("values"):
                self._registry["variables"] = result.get("solution").get("values", {})

            # Store the solution if available
            if result.get("solution"):
                self.last_solution = result.get("solution")

            # Format the result for the client
            formatted_result = {
                "success": True,
                "message": "Model solved",
                "status": result.get("status", "unknown"),
                "output": result.get("output", []),
                "execution_time": result.get("execution_time", 0),
            }

            # Add error information if present
            if result.get("error"):
                formatted_result["error"] = result.get("error")
                formatted_result["success"] = False
                formatted_result["message"] = "Failed to solve model"

            # Add solution information if present
            if result.get("solution"):
                solution = result.get("solution", {})
                formatted_result["satisfiable"] = solution.get("satisfiable", False)
                formatted_result["values"] = solution.get("values", {})

                # Add other solution fields if present
                if solution.get("objective") is not None:
                    formatted_result["objective"] = solution.get("objective")

                # Include output field from solution if present (for property verification messages)
                if solution.get("output") and isinstance(solution.get("output"), list):
                    # Append solution output to existing output
                    formatted_result["output"].extend(solution.get("output"))

                # Add property_verified field if present (for property verification)
                if "property_verified" in solution.get("values", {}):
                    property_verified = solution["values"]["property_verified"]
                    formatted_result["property_verified"] = property_verified

                    # Add appropriate message based on property verification
                    if property_verified:
                        if not any(
                            "verified" in line.lower()
                            for line in formatted_result["output"]
                        ):
                            formatted_result["output"].append(
                                "Property verified successfully."
                            )
                    else:
                        if not any(
                            "counterexample" in line.lower()
                            for line in formatted_result["output"]
                        ):
                            formatted_result["output"].append(
                                "Property verification failed. Counterexample found."
                            )

            return formatted_result

        except ValidationError as e:
            error_msg = str(e)
            self.logger.error(f"Validation error in solve_model: {error_msg}")
            return get_standardized_response(
                success=False,
                message=f"Failed to solve model: {error_msg}",
                error=error_msg,
            )
        except Exception as e:
            error_msg = f"Unexpected error in solve_model: {e!s}"
            self.logger.error(error_msg, exc_info=True)
            return get_standardized_response(
                success=False,
                message="Failed to solve model due to an internal error",
                error=error_msg,
            )

    def get_solution(self) -> dict[str, Any]:
        """
        Get the current solution.

        Returns:
            A dictionary with the current solution
        """
        if not self.last_solution:
            return get_standardized_response(
                success=False, message="No solution available", error="No solution"
            )

        return get_standardized_response(
            success=True,
            message="Solution retrieved",
            satisfiable=self.last_solution.get("satisfiable", False),
            values=self.last_solution.get("values", {}),
            objective=self.last_solution.get("objective"),
            status=self.last_solution.get("status", "unknown"),
        )

    def get_variable_value(self, variable_name: str) -> dict[str, Any]:
        """
        Get the value of a variable from the current solution.

        Args:
            variable_name: The name of the variable

        Returns:
            A dictionary with the value of the variable
        """
        if not self.last_solution:
            return get_standardized_response(
                success=False, message="No solution available", error="No solution"
            )

        values = self.last_solution.get("values", {})

        if variable_name not in values:
            return get_standardized_response(
                success=False,
                message=f"Variable '{variable_name}' not found in solution",
                error="Variable not found",
            )

        return get_standardized_response(
            success=True,
            message=f"Value of variable '{variable_name}'",
            value=values.get(variable_name),
        )

    def get_solve_time(self) -> dict[str, Any]:
        """
        Get the time taken for the last solve operation.

        Returns:
            A dictionary with the solve time information
        """
        if self.last_solve_time is None:
            return get_standardized_response(
                success=False,
                message="No solve operation has been performed",
                error="No solve time available",
            )

        return get_standardized_response(
            success=True,
            message="Solve time retrieved",
            solve_time=self.last_solve_time,
        )
