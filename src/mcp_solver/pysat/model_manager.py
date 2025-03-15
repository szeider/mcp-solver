"""
PySAT model manager implementation.

This module implements the SolverManager abstract base class for PySAT,
providing methods for managing PySAT models.
"""

import asyncio
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import timedelta
import time

from ..base_manager import SolverManager
from .environment import execute_pysat_code

class PySATModelManager(SolverManager):
    """
    PySAT model manager implementation.
    
    This class manages PySAT models, including adding, removing, and modifying
    model items, as well as solving models and extracting solutions.
    """
    
    def __init__(self, lite_mode: bool = False):
        """
        Initialize a new PySAT model manager.
        
        Args:
            lite_mode: Whether the solver is running in lite mode
        """
        super().__init__(lite_mode)
        self.code_items: List[Tuple[int, str]] = []
        self.last_result: Optional[Dict[str, Any]] = None
        self.last_solution: Optional[Dict[str, Any]] = None
        self.initialized = True
        logging.getLogger(__name__).info("PySAT model manager initialized")
    
    async def clear_model(self) -> Dict[str, Any]:
        """
        Clear the current model.
        
        Returns:
            A dictionary with a message indicating the model was cleared
        """
        self.code_items = []
        self.last_result = None
        self.last_solution = None
        logging.getLogger(__name__).info("Model cleared")
        return {"message": "Model cleared"}
    
    def get_model(self) -> List[Tuple[int, str]]:
        """
        Get the current model content with indices.
        
        Returns:
            A list of (index, content) tuples
        """
        return self.code_items
    
    async def add_item(self, index: int, content: str) -> Dict[str, Any]:
        """
        Add an item to the model at the specified index.
        
        Args:
            index: The index at which to add the item
            content: The content to add
            
        Returns:
            A dictionary with the result of the operation
        """
        # If index is -1, append to the end
        if index == -1:
            self.code_items.append((len(self.code_items), content))
            logging.getLogger(__name__).info(f"Added item at index {len(self.code_items) - 1}")
            return {"message": f"Added item at index {len(self.code_items) - 1}"}
        
        # Insert at specific index
        for i, (idx, _) in enumerate(self.code_items):
            if idx == index:
                # Replace existing item
                self.code_items[i] = (index, content)
                logging.getLogger(__name__).info(f"Replaced item at index {index}")
                return {"message": f"Replaced item at index {index}"}
        
        # If index doesn't exist, append with the specified index
        self.code_items.append((index, content))
        logging.getLogger(__name__).info(f"Added item at index {index}")
        return {"message": f"Added item at index {index}"}
    
    async def delete_item(self, index: int) -> Dict[str, Any]:
        """
        Delete an item from the model at the specified index.
        
        Args:
            index: The index of the item to delete
            
        Returns:
            A dictionary with the result of the operation
        """
        for i, (idx, _) in enumerate(self.code_items):
            if idx == index:
                del self.code_items[i]
                logging.getLogger(__name__).info(f"Deleted item at index {index}")
                return {"message": f"Deleted item at index {index}"}
        
        logging.getLogger(__name__).warning(f"Item at index {index} not found")
        return {"message": f"Item at index {index} not found"}
    
    async def replace_item(self, index: int, content: str) -> Dict[str, Any]:
        """
        Replace an item in the model at the specified index.
        
        Args:
            index: The index of the item to replace
            content: The new content
            
        Returns:
            A dictionary with the result of the operation
        """
        for i, (idx, _) in enumerate(self.code_items):
            if idx == index:
                self.code_items[i] = (index, content)
                logging.getLogger(__name__).info(f"Replaced item at index {index}")
                return {"message": f"Replaced item at index {index}"}
        
        # If index doesn't exist, add it
        logging.getLogger(__name__).warning(f"Item at index {index} not found, adding new item")
        return await self.add_item(index, content)
    
    async def solve_model(self, timeout: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Solve the current model.
        
        Args:
            timeout: Optional timeout for the solve operation
            
        Returns:
            A dictionary with the result of the solve operation
        """
        if not self.code_items:
            logging.getLogger(__name__).warning("Attempted to solve empty model")
            return {"message": "Model is empty, nothing to solve", "success": False}
        
        # Sort code items by index
        sorted_items = sorted(self.code_items, key=lambda x: x[0])
        
        # Join code items into a single string
        code_string = "\n".join(content for _, content in sorted_items)
        
        # Set timeout (default to 30 seconds if not specified)
        timeout_seconds = 30.0
        if timeout:
            timeout_seconds = timeout.total_seconds()
        
        # Execute code with timeout
        start_time = time.time()
        self.last_result = execute_pysat_code(code_string, timeout=timeout_seconds)
        self.last_solve_time = time.time() - start_time
        
        # Extract solution if available
        if self.last_result.get("solution"):
            self.last_solution = self.last_result["solution"]
        
        # Determine success/failure message
        if self.last_result.get("success", False):
            message = "Model solved successfully"
            if self.last_solution:
                status = self.last_solution.get("status", "unknown")
                if status == "sat":
                    message += " (satisfiable)"
                elif status == "unsat":
                    message += " (unsatisfiable)"
                else:
                    message += f" (status: {status})"
        else:
            message = "Failed to solve model"
            if self.last_result.get("error"):
                message += f": {self.last_result['error']}"
        
        # Build result dictionary
        result = {
            "message": message,
            "success": self.last_result.get("success", False),
            "solve_time": f"{self.last_solve_time:.6f} seconds",
            "output": self.last_result.get("output", ""),
        }
        
        # Add solution information if available
        if self.last_solution:
            result["satisfiable"] = self.last_solution.get("satisfiable", False)
        
        logging.getLogger(__name__).info(f"Model solved: {result['success']}")
        return result
    
    def get_solution(self) -> Dict[str, Any]:
        """
        Get the current solution.
        
        Returns:
            A dictionary with the current solution
        """
        if not self.last_solution:
            return {"message": "No solution available", "success": False}
        
        return {
            "message": "Solution retrieved",
            "success": True,
            "solution": self.last_solution,
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
            return {"message": "No solution available", "success": False}
        
        if "values" not in self.last_solution:
            return {"message": "Solution does not contain variable values", "success": False}
        
        values = self.last_solution["values"]
        if variable_name not in values:
            return {"message": f"Variable '{variable_name}' not found in solution", "success": False}
        
        return {
            "message": f"Value of variable '{variable_name}'",
            "success": True,
            "value": values[variable_name]
        }
    
    def get_solve_time(self) -> Dict[str, Any]:
        """
        Get the time taken for the last solve operation.
        
        Returns:
            A dictionary with the solve time information
        """
        if self.last_solve_time is None:
            return {"message": "No solve time available", "success": False}
        
        return {
            "message": "Solve time retrieved",
            "success": True,
            "solve_time": f"{self.last_solve_time:.6f} seconds"
        } 