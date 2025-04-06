"""
Z3 model manager implementation.

This module provides the Z3ModelManager class, which implements the SolverManager
interface for the Z3 SMT solver.
"""

import asyncio
import logging
from typing import Dict, Optional, Any, List, Tuple
from datetime import timedelta

# Import z3 for status constant comparison
import z3

from ..core.base_manager import SolverManager
from ..core.constants import MIN_SOLVE_TIMEOUT, MAX_SOLVE_TIMEOUT
from .environment import execute_z3_code
from ..core.validation import (
    validate_index, validate_content, validate_python_code_safety, 
    ValidationError, get_standardized_response, validate_timeout,
    DictionaryMisuseValidator
)

class Z3ModelManager(SolverManager):
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
        self.code_items = []  # List of (index, content) tuples
        self.last_result = None
        self.last_solution = None
        self.initialized = True
        # Add a registry to store variables and solver across different scopes
        self._registry = {
            "variables": {},
            "solver": None
        }
        self.logger = logging.getLogger(__name__)
        self.dict_validator = DictionaryMisuseValidator()  # Initialize the dictionary misuse validator
        self.logger.info("Z3 model manager initialized")
    
    async def clear_model(self) -> Dict[str, Any]:
        """
        Clear the current model.
        
        Returns:
            A dictionary with a message indicating the model was cleared
        """
        self.code_items = []
        self.last_result = None
        self.last_solution = None
        # Reset the dictionary misuse validator
        self.dict_validator = DictionaryMisuseValidator()
        # Clear the registry when model is cleared
        self._registry = {
            "variables": {},
            "solver": None
        }
        self.logger.info("Model cleared")
        return get_standardized_response(
            success=True,
            message="Model cleared"
        )
    
    def get_model(self) -> List[Tuple[int, str]]:
        """
        Get the current model content with indexes.
        
        Returns:
            A list of (index, content) tuples
        """
        return [(i+1, content) for i, content in enumerate(self.code_items)]
    
    async def add_item(self, index: int, content: str) -> Dict[str, Any]:
        """
        Add an item to the model at the specified index.
        
        Args:
            index: The index at which to add the item (1-based)
            content: The content to add
            
        Returns:
            A dictionary with the result of the operation
        """
        try:
            # Validate inputs - Z3 uses 1-based indexing in API, but 0-based internally
            validate_index(index, [(i+1, c) for i, c in enumerate(self.code_items)], one_based=True)
            validate_content(content)
            validate_python_code_safety(content)
            
            # Add the code fragment to the dictionary misuse validator
            self.dict_validator.add_fragment(content, index)
            self.logger.debug(f"Added fragment to dictionary validator: item {index}")
            
            # Adjust index to 0-based for internal storage
            index_0 = max(0, min(len(self.code_items), index - 1))
            
            # Insert the item
            self.code_items.insert(index_0, content)
            
            self.logger.info(f"Added item at index {index}")
            return get_standardized_response(
                success=True,
                message=f"Added item at index {index}",
                model=self.get_model()
            )
            
        except ValidationError as e:
            error_msg = str(e)
            self.logger.error(f"Validation error in add_item: {error_msg}")
            return get_standardized_response(
                success=False,
                message=f"Failed to add item: {error_msg}",
                error=error_msg,
                model=self.get_model()
            )
        except Exception as e:
            error_msg = f"Unexpected error in add_item: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return get_standardized_response(
                success=False,
                message="Failed to add item due to an internal error",
                error=error_msg,
                model=self.get_model()
            )
    
    async def delete_item(self, index: int) -> Dict[str, Any]:
        """
        Delete an item from the model at the specified index.
        
        Args:
            index: The index of the item to delete (1-based)
            
        Returns:
            A dictionary with the result of the operation
        """
        try:
            # Validate index
            validate_index(index, one_based=True)
            
            # Adjust index to 0-based for internal storage
            index_0 = index - 1
            
            # Check if index is valid for the array
            if index_0 < 0 or index_0 >= len(self.code_items):
                return get_standardized_response(
                    success=False,
                    message=f"Invalid index: {index}",
                    error="Index out of range",
                    model=self.get_model()
                )
            
            # Delete the item
            del self.code_items[index_0]
            
            self.logger.info(f"Deleted item at index {index}")
            return get_standardized_response(
                success=True,
                message=f"Deleted item at index {index}",
                model=self.get_model()
            )
            
        except ValidationError as e:
            error_msg = str(e)
            self.logger.error(f"Validation error in delete_item: {error_msg}")
            return get_standardized_response(
                success=False,
                message=f"Failed to delete item: {error_msg}",
                error=error_msg,
                model=self.get_model()
            )
        except Exception as e:
            error_msg = f"Unexpected error in delete_item: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return get_standardized_response(
                success=False,
                message="Failed to delete item due to an internal error",
                error=error_msg,
                model=self.get_model()
            )
    
    async def replace_item(self, index: int, content: str) -> Dict[str, Any]:
        """
        Replace an item in the model at the specified index.
        
        Args:
            index: The index of the item to replace (1-based)
            content: The new content
            
        Returns:
            A dictionary with the result of the operation
        """
        try:
            # Validate inputs
            validate_index(index, one_based=True)
            validate_content(content)
            validate_python_code_safety(content)
            
            # Add the code fragment to the dictionary misuse validator
            self.dict_validator.add_fragment(content, index)
            self.logger.debug(f"Added fragment to dictionary validator: item {index}")
            
            # Adjust index to 0-based for internal storage
            index_0 = index - 1
            
            # Check if index is valid for the array
            if index_0 < 0 or index_0 >= len(self.code_items):
                return get_standardized_response(
                    success=False,
                    message=f"Invalid index: {index}",
                    error="Index out of range",
                    model=self.get_model()
                )
            
            # Replace the item
            self.code_items[index_0] = content
            
            self.logger.info(f"Replaced item at index {index}")
            return get_standardized_response(
                success=True,
                message=f"Replaced item at index {index}",
                model=self.get_model()
            )
            
        except ValidationError as e:
            error_msg = str(e)
            self.logger.error(f"Validation error in replace_item: {error_msg}")
            return get_standardized_response(
                success=False,
                message=f"Failed to replace item: {error_msg}",
                error=error_msg,
                model=self.get_model()
            )
        except Exception as e:
            error_msg = f"Unexpected error in replace_item: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return get_standardized_response(
                success=False,
                message="Failed to replace item due to an internal error",
                error=error_msg,
                model=self.get_model()
            )
    
    async def solve_model(self, timeout: timedelta) -> Dict[str, Any]:
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
                    success=False,
                    message="Model is empty",
                    error="Empty model"
                )
            
            # Validate timeout
            validate_timeout(timeout, MIN_SOLVE_TIMEOUT, MAX_SOLVE_TIMEOUT)
            
            # Perform dictionary misuse validation before executing the code
            dict_validation_result = self.dict_validator.validate()
            if dict_validation_result["has_errors"]:
                error_messages = []
                for error in dict_validation_result["errors"]:
                    item_index = error.get("item", "?")
                    line_num = error.get("fragment_line", error.get("line", "?"))
                    error_messages.append(f"Item {item_index}, Line {line_num}: {error['message']} {error['suggestion']}")
                
                self.logger.warning(f"Dictionary misuse detected: {error_messages}")
                
                return get_standardized_response(
                    success=False,
                    message="Your code contains a common dictionary usage error that will cause unexpected behavior.",
                    error="Dictionary misuse detected",
                    error_details={
                        "errors": error_messages,
                        "suggestion": "Check how you're updating dictionary variables. Use var_dict[key] = value instead of var_dict = value."
                    }
                )
            
            # Combine code items into a single string
            combined_code = "\n".join(self.code_items)
            
            # Add wrapper for export_solution to capture variables and solver
            combined_code = """
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
""" + combined_code
            
            # Set timeout in seconds
            timeout_seconds = timeout.total_seconds()
            
            # Execute the code
            result = execute_z3_code(combined_code, timeout=timeout_seconds)
            
            # Store the result for later retrieval
            self.last_result = result
            self.last_solve_time = result.get("execution_time")
            
            # Check for error indicating missing export_solution
            if (result.get("status") == "no_solution" or 
                (result.get("error") and "No solution was exported" in result.get("error"))):
                
                # Check if there's a stored solver from previous export_solution calls
                if "_z3_registry" in globals() and globals()["_z3_registry"].get("solver"):
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
                            # Evaluate the variable in the model
                            val = model.eval(var)
                            # Try to convert to appropriate type (int, fraction, bool, or string)
                            try:
                                solution[var_name] = val.as_long()
                            except (AttributeError, z3.Z3Exception):
                                try:
                                    solution[var_name] = val.as_fraction()
                                except (AttributeError, z3.Z3Exception):
                                    try:
                                        solution[var_name] = bool(val)
                                    except (ValueError, TypeError):
                                        # Fall back to string representation
                                        solution[var_name] = str(val)
                        
                        # Update the result
                        result["status"] = "success"
                        result["solution"] = {
                            "satisfiable": True,
                            "values": solution,
                            "status": "sat"
                        }
                        
                        # Store the solution for later retrieval
                        self.last_solution = result.get("solution")
            
            # For success case, also update our registry from the solution
            if result.get("solution"):
                # Check if solution is a dictionary with a 'values' field
                if isinstance(result.get("solution"), dict) and result.get("solution").get("values"):
                    self._registry["variables"] = result.get("solution").get("values", {})
            
            # Store the solution if available
            if result.get("solution"):
                self.last_solution = result.get("solution")
            
            # Prepare base result information
            success = not bool(result.get("error"))
            message = "Model solved" if success else "Failed to solve model"
            
            # Start with a standardized response
            response_data = {
                "status": result.get("status", "unknown"),
                "output": result.get("output", []),
                "execution_time": result.get("execution_time", 0),
            }
            
            # Add error information if present
            if result.get("error"):
                response_data["error"] = result.get("error")
            
            # Add solution information if present
            if result.get("solution"):
                solution = result.get("solution", {})
                response_data["satisfiable"] = solution.get("satisfiable", False)
                response_data["values"] = solution.get("values", {})
                
                # Add other solution fields if present
                if solution.get("objective") is not None:
                    response_data["objective"] = solution.get("objective")
                
                # Include output field from solution if present (for property verification messages)
                if solution.get("output") and isinstance(solution.get("output"), list):
                    # Append solution output to existing output
                    response_data["output"].extend(solution.get("output"))
                
                # Add property_verified field if present (for property verification)
                if "property_verified" in solution.get("values", {}):
                    property_verified = solution["values"]["property_verified"]
                    response_data["property_verified"] = property_verified
                    
                    # Add appropriate message based on property verification
                    if property_verified:
                        if not any("verified" in line.lower() for line in response_data["output"]):
                            response_data["output"].append("Property verified successfully.")
                    else:
                        if not any("counterexample" in line.lower() for line in response_data["output"]):
                            response_data["output"].append("Property verification failed. Counterexample found.")
            
            # Use standardized response format
            formatted_result = get_standardized_response(
                success=success,
                message=message,
                **response_data
            )
            
            return formatted_result
            
        except ValidationError as e:
            error_msg = str(e)
            self.logger.error(f"Validation error in solve_model: {error_msg}")
            return get_standardized_response(
                success=False,
                message=f"Failed to solve model: {error_msg}",
                error=error_msg
            )
        except Exception as e:
            error_msg = f"Unexpected error in solve_model: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return get_standardized_response(
                success=False,
                message="Failed to solve model due to an internal error",
                error=error_msg
            )
    
    def get_solution(self) -> Dict[str, Any]:
        """
        Get the current solution.
        
        Returns:
            A dictionary with the current solution
        """
        if not self.last_solution:
            return get_standardized_response(
                success=False,
                message="No solution available",
                error="No solution"
            )
        
        return get_standardized_response(
            success=True,
            message="Solution retrieved",
            satisfiable=self.last_solution.get("satisfiable", False),
            values=self.last_solution.get("values", {}),
            objective=self.last_solution.get("objective"),
            status=self.last_solution.get("status", "unknown")
        )
    
    def get_variable_value(self, variable_name: str) -> Dict[str, Any]:
        """
        Get the value of a variable from the current solution.
        
        Args:
            variable_name: The name of the variable
            
        Returns:
            A dictionary with the value of the variable
        """
        if not self.last_solution:
            return get_standardized_response(
                success=False,
                message="No solution available",
                error="No solution"
            )
        
        values = self.last_solution.get("values", {})
        
        if variable_name not in values:
            return get_standardized_response(
                success=False,
                message=f"Variable '{variable_name}' not found in solution",
                error="Variable not found"
            )
        
        return get_standardized_response(
            success=True,
            message=f"Value of variable '{variable_name}'",
            value=values.get(variable_name)
        )
    
    def get_solve_time(self) -> Dict[str, Any]:
        """
        Get the time taken for the last solve operation.
        
        Returns:
            A dictionary with the solve time information
        """
        if self.last_solve_time is None:
            return get_standardized_response(
                success=False,
                message="No solve operation has been performed",
                error="No solve time available"
            )
        
        return get_standardized_response(
            success=True,
            message="Solve time retrieved",
            solve_time=self.last_solve_time
        ) 