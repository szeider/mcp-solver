"""
Z3 model manager implementation.

This module provides the Z3ModelManager class, which implements the SolverManager
interface for the Z3 SMT solver.
"""

import asyncio
from typing import Dict, Optional, Any, List, Tuple
from datetime import timedelta

from ..base_manager import SolverManager
from .environment import execute_z3_code

class Z3ModelManager(SolverManager):
    """
    Z3 model manager implementation.
    
    This class manages Z3 models, including adding, removing, and modifying
    model items, as well as solving models and extracting solutions.
    """
    
    def __init__(self, lite_mode: bool = False):
        """
        Initialize a new Z3 model manager.
        
        Args:
            lite_mode: Whether the solver is running in lite mode
        """
        super().__init__(lite_mode)
        self.code_items = []  # List of (index, content) tuples
        self.last_result = None
        self.last_solution = None
        self.initialized = True
    
    async def clear_model(self) -> Dict[str, Any]:
        """
        Clear the current model.
        
        Returns:
            A dictionary with a message indicating the model was cleared
        """
        self.code_items = []
        self.last_result = None
        self.last_solution = None
        return {"message": "Model cleared"}
    
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
        # Adjust index to 0-based for internal storage
        index_0 = max(0, min(len(self.code_items), index - 1))
        
        # Insert the item
        self.code_items.insert(index_0, content)
        
        # Return the updated model
        return {
            "message": f"Added item at index {index}",
            "model": self.get_model()
        }
    
    async def delete_item(self, index: int) -> Dict[str, Any]:
        """
        Delete an item from the model at the specified index.
        
        Args:
            index: The index of the item to delete (1-based)
            
        Returns:
            A dictionary with the result of the operation
        """
        # Adjust index to 0-based for internal storage
        index_0 = index - 1
        
        # Check if index is valid
        if index_0 < 0 or index_0 >= len(self.code_items):
            return {
                "error": f"Invalid index: {index}",
                "model": self.get_model()
            }
        
        # Delete the item
        del self.code_items[index_0]
        
        # Return the updated model
        return {
            "message": f"Deleted item at index {index}",
            "model": self.get_model()
        }
    
    async def replace_item(self, index: int, content: str) -> Dict[str, Any]:
        """
        Replace an item in the model at the specified index.
        
        Args:
            index: The index of the item to replace (1-based)
            content: The new content
            
        Returns:
            A dictionary with the result of the operation
        """
        # Adjust index to 0-based for internal storage
        index_0 = index - 1
        
        # Check if index is valid
        if index_0 < 0 or index_0 >= len(self.code_items):
            return {
                "error": f"Invalid index: {index}",
                "model": self.get_model()
            }
        
        # Replace the item
        self.code_items[index_0] = content
        
        # Return the updated model
        return {
            "message": f"Replaced item at index {index}",
            "model": self.get_model()
        }
    
    async def solve_model(self, timeout: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Solve the current model.
        
        Args:
            timeout: Optional timeout for the solve operation
            
        Returns:
            A dictionary with the result of the solve operation
        """
        # Check if model is empty
        if not self.code_items:
            return {
                "error": "Model is empty",
                "status": "error"
            }
        
        # Combine code items into a single string
        combined_code = "\n".join(self.code_items)
        
        # Set timeout in seconds
        timeout_seconds = timeout.total_seconds() if timeout else 10.0
        
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
            "status": result.get("status", "unknown"),
            "output": result.get("output", []),
            "execution_time": result.get("execution_time", 0),
        }
        
        # Add error information if present
        if result.get("error"):
            formatted_result["error"] = result.get("error")
        
        # Add solution information if present
        if result.get("solution"):
            solution = result.get("solution", {})
            formatted_result["satisfiable"] = solution.get("satisfiable", False)
            formatted_result["values"] = solution.get("values", {})
            if solution.get("objective") is not None:
                formatted_result["objective"] = solution.get("objective")
        
        return formatted_result
    
    def get_solution(self) -> Dict[str, Any]:
        """
        Get the current solution.
        
        Returns:
            A dictionary with the current solution
        """
        if not self.last_solution:
            return {
                "error": "No solution available",
                "status": "error"
            }
        
        return {
            "satisfiable": self.last_solution.get("satisfiable", False),
            "values": self.last_solution.get("values", {}),
            "objective": self.last_solution.get("objective"),
            "status": self.last_solution.get("status", "unknown")
        }
    
    def get_variable_value(self, variable_name: str) -> Dict[str, Any]:
        """
        Get the value of a variable from the current solution.
        
        Args:
            variable_name: The name of the variable
            
        Returns:
            A dictionary with the value of the variable
        """
        if not self.last_solution:
            return {
                "error": "No solution available",
                "status": "error"
            }
        
        values = self.last_solution.get("values", {})
        
        if variable_name not in values:
            return {
                "error": f"Variable '{variable_name}' not found in solution",
                "status": "error"
            }
        
        return {
            "value": values.get(variable_name),
            "status": "success"
        }
    
    def get_solve_time(self) -> Dict[str, Any]:
        """
        Get the time taken for the last solve operation.
        
        Returns:
            A dictionary with the solve time information
        """
        if self.last_solve_time is None:
            return {
                "error": "No solve operation has been performed",
                "status": "error"
            }
        
        return {
            "time": self.last_solve_time,
            "status": "success"
        } 