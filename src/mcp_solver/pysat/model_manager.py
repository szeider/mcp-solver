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
from ..constants import MIN_SOLVE_TIMEOUT, MAX_SOLVE_TIMEOUT
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
            timeout: Timeout for the solve operation. Defaults to 5 seconds if None.
            
        Returns:
            A dictionary with the result of the solve operation
        """
        if not self.code_items:
            logging.getLogger(__name__).warning("Attempted to solve empty model")
            return {"message": "Model is empty, nothing to solve", "success": False}
        
        # Set default timeout if not provided
        if timeout is None:
            timeout = timedelta(seconds=5.0)
        
        # Sort code items by index
        sorted_items = sorted(self.code_items, key=lambda x: x[0])
        
        # Join code items into a single string
        code_string = "\n".join(content for _, content in sorted_items)
        
        # Modify the code to ensure it prints the satisfiability result
        # We'll add a simple print statement that we can parse later
        # Look for solver.solve() patterns and add appropriate debug prints
        modified_code = ""
        import re
        
        # Track if we've already found and handled a solve() call
        found_solve_call = False
        
        for line in code_string.split("\n"):
            modified_code += line + "\n"
            
            # Case 1: Direct conditional - if solver.solve():
            if re.search(r'if\s+\w+\.solve\(\)', line) and not found_solve_call:
                # Extract solver name
                solver_name = re.search(r'if\s+(\w+)\.solve\(\)', line).group(1)
                # Add debug print inside the conditional branch
                modified_code += f"    print(f\"PYSAT_DEBUG_OUTPUT: model_is_satisfiable=True\")\n"
                found_solve_call = True
            
            # Case 2: Variable assignment - is_sat = solver.solve() or any variable name
            elif re.search(r'(\w+)\s*=\s*\w+\.solve\(\)', line) and not found_solve_call:
                # Extract variable name and add debug print
                var_name = re.search(r'(\w+)\s*=\s*\w+\.solve\(\)', line).group(1)
                modified_code += f"print(f\"PYSAT_DEBUG_OUTPUT: model_is_satisfiable={{{var_name}}}\")\n"
                found_solve_call = True
        
        # Set timeout
        timeout_seconds = timeout.total_seconds()
        
        # Execute code with timeout
        start_time = time.time()
        self.last_result = execute_pysat_code(modified_code, timeout=timeout_seconds)
        self.last_solve_time = time.time() - start_time
        
        # Extract solver output to check for satisfiability
        output = self.last_result.get("output", "")
        satisfiable = False
        
        # Parse output for explicit satisfiability result
        import re
        sat_match = re.search(r"PYSAT_DEBUG_OUTPUT: model_is_satisfiable=(\w+)", output)
        if sat_match:
            satisfiable = sat_match.group(1).lower() == "true"
            logging.getLogger(__name__).debug(f"Found explicit satisfiability result: {satisfiable}")
        else:
            # Also try to find a standard output message
            if "Is satisfiable: True" in output:
                satisfiable = True
                logging.getLogger(__name__).debug("Found 'Is satisfiable: True' in output")
        
        # Extract solution if available
        if self.last_result.get("solution"):
            self.last_solution = self.last_result["solution"]
            
            # Check if the solution contains satisfiability info
            if "satisfiable" in self.last_solution:
                # Use our parsed result if available, otherwise use solution's value
                if sat_match:
                    self.last_solution["satisfiable"] = satisfiable
        else:
            # Create a minimal solution with just the satisfiability flag
            self.last_solution = {
                "satisfiable": satisfiable,
                "status": "sat" if satisfiable else "unsat",
                "values": {}
            }
        
        # Determine success/failure message
        if self.last_result.get("success", False):
            message = "Model solved successfully"
            status = self.last_solution.get("status", "unknown")
            if satisfiable:
                message += " (satisfiable)"
                # Ensure status is consistent with satisfiability
                self.last_solution["status"] = "sat"
            elif satisfiable is False:  # Explicitly False, not just falsy
                message += " (unsatisfiable)"
                # Ensure status is consistent with satisfiability
                self.last_solution["status"] = "unsat"
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
            "output": output,
            "satisfiable": satisfiable  # Always include the satisfiability flag
        }
        
        # Add solution information if available
        if self.last_solution:
            # Include status and values from solution
            if "status" in self.last_solution:
                result["status"] = self.last_solution["status"]
            if "values" in self.last_solution:
                result["values"] = self.last_solution["values"]
        
        logging.getLogger(__name__).info(f"Model solved: {result['success']}, satisfiable: {satisfiable}")
        
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