"""
Z3 model manager implementation.

This module provides the Z3ModelManager class, which implements the SolverManager
interface for the Z3 SMT solver.
"""

import asyncio
import logging
from typing import Dict, Optional, Any, List, Tuple
from datetime import timedelta

from ..base_manager import SolverManager
from ..constants import MIN_SOLVE_TIMEOUT, MAX_SOLVE_TIMEOUT
from .environment import execute_z3_code
from ..validation import validate_index, validate_content, validate_python_code_safety, validate_timeout, ValidationError, get_standardized_response

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
        self.logger = logging.getLogger(__name__)
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
            
            # Combine code items into a single string
            combined_code = "\n".join(self.code_items)
            
            # Set timeout in seconds
            timeout_seconds = timeout.total_seconds()
            
            # Execute the code
            result = execute_z3_code(combined_code, timeout=timeout_seconds)
            
            # Store the result for later retrieval
            self.last_result = result
            self.last_solve_time = result.get("execution_time")
            
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
                if solution.get("objective") is not None:
                    formatted_result["objective"] = solution.get("objective")
            
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