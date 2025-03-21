"""
PySAT model manager implementation.

This module implements the SolverManager abstract base class for PySAT,
providing methods for managing PySAT models.
"""

import asyncio
import logging
from typing import Dict, Any, List, Tuple, Optional, Union, cast
from datetime import timedelta
import time
import re

from ..base_manager import SolverManager
from ..constants import MIN_SOLVE_TIMEOUT, MAX_SOLVE_TIMEOUT
from .environment import execute_pysat_code
from .error_handling import PySATError, format_solution_error
from ..validation import validate_index, validate_content, validate_python_code_safety, ValidationError, get_standardized_response, validate_timeout

# Validation constants are now imported from validation module

class PySATModelManager(SolverManager):
    """
    PySAT model manager implementation.
    
    This class manages PySAT models, including adding, removing, and modifying
    model items, as well as solving models and extracting solutions.
    """
    
    def __init__(self):
        """
        Initialize a new PySAT model manager.
        """
        super().__init__()
        self.code_items: List[Tuple[int, str]] = []
        self.last_result: Optional[Dict[str, Any]] = None
        self.last_solution: Optional[Dict[str, Any]] = None
        self.last_solve_time: float = 0.0
        self.initialized = True
        self.logger = logging.getLogger(__name__)
        self.logger.info("PySAT model manager initialized")
    
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
        return {"message": "Model cleared successfully"}
    
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
            content: The content of the item
            
        Returns:
            A dictionary with the result of the operation
            
        Raises:
            ValidationError: If the input is invalid
        """
        try:
            # Validate inputs
            validate_index(index, self.code_items, one_based=True)
            validate_content(content)
            validate_python_code_safety(content)
            
            # Check if an item with the same index already exists
            for i, (idx, _) in enumerate(self.code_items):
                if idx == index:
                    # Replace existing item
                    self.code_items[i] = (index, content)
                    self.logger.info(f"Replaced item at index {index}")
                    return get_standardized_response(
                        success=True,
                        message=f"Replaced item at index {index}"
                    )
            
            # Add new item
            self.code_items.append((index, content))
            self.logger.info(f"Added item at index {index}")
            return get_standardized_response(
                success=True,
                message=f"Added item at index {index}"
            )
            
        except ValidationError as e:
            error_msg = str(e)
            self.logger.error(f"Validation error in add_item: {error_msg}")
            return get_standardized_response(
                success=False,
                message=f"Failed to add item: {error_msg}",
                error=error_msg
            )
        except Exception as e:
            error_msg = f"Unexpected error in add_item: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return get_standardized_response(
                success=False,
                message="Failed to add item due to an internal error",
                error=error_msg
            )
    
    async def delete_item(self, index: int) -> Dict[str, Any]:
        """
        Delete an item from the model at the specified index.
        
        Args:
            index: The index of the item to delete
            
        Returns:
            A dictionary with the result of the operation
        """
        try:
            # Basic index validation - only check if it's a valid integer
            validate_index(index, one_based=True)
            
            for i, (idx, _) in enumerate(self.code_items):
                if idx == index:
                    del self.code_items[i]
                    self.logger.info(f"Deleted item at index {index}")
                    return get_standardized_response(
                        success=True,
                        message=f"Deleted item at index {index}"
                    )
            
            self.logger.warning(f"Item at index {index} not found")
            return get_standardized_response(
                success=False,
                message=f"Item at index {index} not found",
                error="Item not found"
            )
        
        except ValidationError as e:
            error_msg = str(e)
            self.logger.error(f"Validation error in delete_item: {error_msg}")
            return get_standardized_response(
                success=False,
                message=f"Failed to delete item: {error_msg}",
                error=error_msg
            )
        except Exception as e:
            error_msg = f"Unexpected error in delete_item: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return get_standardized_response(
                success=False,
                message="Failed to delete item due to an internal error",
                error=error_msg
            )
    
    async def replace_item(self, index: int, content: str) -> Dict[str, Any]:
        """
        Replace an item in the model at the specified index.
        
        Args:
            index: The index of the item to replace
            content: The new content of the item
            
        Returns:
            A dictionary with the result of the operation
        """
        try:
            # Validate inputs
            validate_index(index, self.code_items, one_based=True)
            validate_content(content)
            validate_python_code_safety(content)
            
            # Check if the item exists
            for i, (idx, _) in enumerate(self.code_items):
                if idx == index:
                    # Replace existing item
                    self.code_items[i] = (index, content)
                    self.logger.info(f"Replaced item at index {index}")
                    return get_standardized_response(
                        success=True,
                        message=f"Replaced item at index {index}"
                    )
            
            # Item not found, add as new
            self.logger.warning(f"Item at index {index} not found, adding as new")
            return await self.add_item(index, content)
            
        except ValidationError as e:
            error_msg = str(e)
            self.logger.error(f"Validation error in replace_item: {error_msg}")
            return get_standardized_response(
                success=False,
                message=f"Failed to replace item: {error_msg}",
                error=error_msg
            )
        except Exception as e:
            error_msg = f"Unexpected error in replace_item: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return get_standardized_response(
                success=False,
                message="Failed to replace item due to an internal error",
                error=error_msg
            )
    
    async def solve_model(self, timeout: timedelta) -> Dict[str, Any]:
        """
        Solve the current model with a timeout.
        
        Args:
            timeout: Maximum time to spend on solving
            
        Returns:
            A dictionary with the solving result
        """
        try:
            if not self.initialized:
                return get_standardized_response(
                    success=True,  # Always return success=True to maintain connection
                    message="Model manager not initialized",
                    status="error",
                    error="Not initialized"
                )
            
            if not self.code_items:
                return get_standardized_response(
                    success=True,  # Always return success=True to maintain connection
                    message="No model items to solve",
                    status="error",
                    error="Empty model"
                )
            
            # Validate timeout
            validate_timeout(timeout, MIN_SOLVE_TIMEOUT, MAX_SOLVE_TIMEOUT)
            
            # Sort code items by index
            sorted_items = sorted(self.code_items, key=lambda x: x[0])
            
            # Join code items into a single string
            code_string = "\n".join(content for _, content in sorted_items)
            
            # Perform static analysis on the code before executing it
            try:
                import ast
                ast_tree = ast.parse(code_string)
                
                # Check for solver.solve() patterns
                solve_calls = 0
                for node in ast.walk(ast_tree):
                    if isinstance(node, ast.Call):
                        if (isinstance(node.func, ast.Attribute) and 
                            node.func.attr == 'solve' and
                            isinstance(node.func.value, ast.Name)):
                            solve_calls += 1
                
                if solve_calls == 0:
                    self.logger.warning("No solver.solve() call found in the code")
                    return get_standardized_response(
                        success=True,  # Always return success=True to maintain connection
                        message="No solver.solve() call found in the code. Make sure to create a solver and call its solve() method.",
                        status="error",
                        error="Missing solve call",
                        code_analysis="Missing solver.solve() call"
                    )
                
                # Check for export_solution calls
                export_calls = 0
                for node in ast.walk(ast_tree):
                    if isinstance(node, ast.Call):
                        if (isinstance(node.func, ast.Name) and 
                            node.func.id == 'export_solution'):
                            export_calls += 1
                
                if export_calls == 0:
                    self.logger.warning("No export_solution() call found in the code")
                    return get_standardized_response(
                        success=True,  # Always return success=True to maintain connection
                        message="No export_solution() call found in the code. Make sure to call export_solution() with your result.",
                        status="error",
                        error="Missing export_solution call",
                        code_analysis="Missing export_solution() call"
                    )
                
                # Look for common mistake patterns
                code_lines = code_string.split('\n')
                line_issues = []
                
                for i, line in enumerate(code_lines):
                    # Check for variables = var_id pattern (common mistake)
                    if re.search(r'variables\s*=\s*var_id', line):
                        line_issues.append({
                            "line": i + 1,
                            "code": line.strip(),
                            "issue": "Incorrect assignment to variables dictionary. Use variables[key] = var_id instead of variables = var_id"
                        })
                    
                    # Check for variables in list comprehension without indexing
                    if re.search(r'\[\s*variables\s+for', line):
                        line_issues.append({
                            "line": i + 1,
                            "code": line.strip(),
                            "issue": "Incorrect use of variables in list comprehension. Use variables[key] instead of just variables"
                        })
                
                if line_issues:
                    issues_text = "\n".join([f"Line {issue['line']}: {issue['code']} - {issue['issue']}" for issue in line_issues])
                    self.logger.warning(f"Found potential code issues:\n{issues_text}")
                    # We don't return here, just warn - give the code a chance to run
            
            except SyntaxError as e:
                line_num = e.lineno if hasattr(e, 'lineno') else '?'
                col_num = e.offset if hasattr(e, 'offset') else '?'
                error_text = e.text.strip() if hasattr(e, 'text') and e.text else 'unknown'
                
                self.logger.error(f"Syntax error in code at line {line_num}, column {col_num}: {str(e)}")
                return get_standardized_response(
                    success=True,  # Always return success=True to maintain connection
                    message=f"Syntax error at line {line_num}, column {col_num}: {str(e)}",
                    status="error",
                    error="Syntax error",
                    error_details={
                        "line": line_num,
                        "column": col_num,
                        "code": error_text,
                        "message": str(e)
                    }
                )
            except Exception as e:
                self.logger.error(f"Error analyzing code: {str(e)}")
                # Continue despite analysis error
            
            # Modify the code to enhance debugging
            modified_code = self._enhance_code_for_debugging(code_string)
            
            # Set timeout
            timeout_seconds = timeout.total_seconds()
            self.last_solve_time = 0.0
            
            # Execute code with timeout using asyncio to avoid blocking the event loop
            start_time = time.time()
            
            # Use run_in_executor to execute the synchronous execute_pysat_code in a separate thread
            # This prevents it from blocking the asyncio event loop
            loop = asyncio.get_running_loop()
            try:
                # Execute the code in a thread pool to not block the event loop
                self.last_result = await loop.run_in_executor(
                    None,  # Use default executor
                    lambda: execute_pysat_code(modified_code, timeout=timeout_seconds)
                )
                self.last_solve_time = time.time() - start_time
                
            except asyncio.CancelledError:
                # Handle cancellation during execution - create a timeout response
                self.last_solve_time = time.time() - start_time
                self.logger.warning(f"Execution was cancelled after {self.last_solve_time:.2f} seconds")
                
                # Create a timeout-like response that is still a success
                self.last_result = {
                    'success': True,  # Mark as successful to prevent disconnection
                    'error': None,
                    'status': 'timeout',
                    'output': f"Execution was cancelled after {self.last_solve_time:.2f} seconds",
                    'solution': None,
                    'timeout': True
                }
                
            except Exception as e:
                # Handle other exceptions during execution
                self.last_solve_time = time.time() - start_time
                self.logger.error(f"Error executing code: {str(e)}", exc_info=True)
                
                # Create an error response that is still a success
                self.last_result = {
                    'success': True,  # Mark as successful to prevent disconnection
                    'error': f"Execution error: {str(e)}",
                    'status': 'error',
                    'output': f"Error during execution: {str(e)}",
                    'solution': None
                }
            
            # If solve time exceeds timeout (shouldn't happen but just in case)
            if self.last_solve_time > timeout_seconds:
                self.logger.warning(f"Actual solving time ({self.last_solve_time:.2f}s) exceeds timeout ({timeout_seconds}s)")
            
            # Check if there was a timeout
            if self.last_result and self.last_result.get('timeout') is True:
                # Handle timeout as a special case
                self.logger.warning(f"PySAT execution timed out after {timeout_seconds} seconds")
                
                # Create a standardized timeout response
                self.last_solution = {
                    "satisfiable": None,
                    "status": "timeout",
                    "values": {},
                    "solve_time": f"{self.last_solve_time:.6f} seconds",
                    "timeout": True
                }
                
                # Return a timeout response that's successful but indicates timeout
                timeout_response = {
                    "message": f"Model execution timed out after {timeout_seconds} seconds",
                    "success": True,  # We're treating timeout as a successful execution with a timeout result
                    "status": "timeout",
                    "solve_time": self.last_solve_time,
                    "timeout": True
                }
                
                # Log that we're returning a controlled timeout response
                self.logger.info("Returning controlled timeout response to client")
                return timeout_response
            
            # Check if there were other execution errors but still maintain connection
            elif self.last_result and self.last_result.get("error"):
                error_msg = self.last_result["error"]
                self.logger.error(f"Error executing code: {error_msg}")
                
                # Return a response that indicates an error but is still a success
                error_response = {
                    "message": f"Error executing code: {error_msg}",
                    "success": True,  # Mark as successful to prevent disconnection
                    "status": "error",
                    "error": error_msg,
                    "solve_time": self.last_solve_time
                }
                
                # If we captured line issues during analysis, include them in error details
                if line_issues:
                    error_response["error_details"] = {
                        "execution_error": error_msg,
                        "code_issues": line_issues
                    }
                
                return error_response
            
            # Extract solver output to check for satisfiability
            output = self.last_result.get("output", "")
            satisfiable = False
            
            # Parse output for explicit satisfiability result
            sat_match = re.search(r"PYSAT_DEBUG_OUTPUT: model_is_satisfiable=(\w+)", output)
            if sat_match:
                satisfiable = sat_match.group(1).lower() == "true"
                self.logger.debug(f"Found explicit satisfiability result: {satisfiable}")
            else:
                # Also try to find standard output messages
                if "Is satisfiable: True" in output:
                    satisfiable = True
                    self.logger.debug("Found 'Is satisfiable: True' in output")
            
            # Extract solution if available
            if self.last_result.get("solution"):
                self.last_solution = self.last_result["solution"]
                
                # Use our parsed satisfiability result to override the solution
                # This ensures consistency between satisfiable flag and status
                if sat_match or "Is satisfiable: True" in output:
                    self.last_solution["satisfiable"] = satisfiable
            else:
                # Create a minimal solution with just the satisfiability flag
                self.last_solution = {
                    "satisfiable": satisfiable,
                    "status": "sat" if satisfiable else "unsat",
                    "values": {}
                }
            
            # Ensure there's a 'values' dictionary for standardized access
            if "values" not in self.last_solution:
                self.last_solution["values"] = {}
            
            # Add solve time to solution
            self.last_solution["solve_time"] = f"{self.last_solve_time:.6f} seconds"
            
            # Add any warnings from static analysis
            if line_issues:
                self.last_solution["warnings"] = [
                    f"Line {issue['line']}: {issue['issue']}" for issue in line_issues
                ]
            
            # Log the successful solve
            self.logger.info(f"Model solved: {bool(self.last_result.get('success'))}, satisfiable: {satisfiable}")
            
            # Prepare final response
            response = {
                "message": "Model solved successfully" + (" (satisfiable)" if satisfiable else " (unsatisfiable)"),
                "success": True,
                "solve_time": self.last_solve_time,
                "output": output,
                "satisfiable": satisfiable
            }
            
            # Include status and values
            if self.last_solution.get("status"):
                response["status"] = self.last_solution["status"]
            if self.last_solution.get("values"):
                response["values"] = self.last_solution["values"]
            
            # Include warnings if any
            if self.last_solution.get("warnings"):
                response["warnings"] = self.last_solution["warnings"]
            
            return response
            
        except Exception as e:
            # Log the error
            self.logger.error(f"Error in solve_model: {str(e)}", exc_info=True)
            
            # Return a structured error response that is still a success
            return {
                "message": f"Error solving model: {str(e)}",
                "success": True,  # Mark as successful to prevent disconnection
                "status": "error",
                "error": str(e),
                "solve_time": self.last_solve_time if hasattr(self, 'last_solve_time') else 0.0
            }
            
    def _enhance_code_for_debugging(self, code_string: str) -> str:
        """
        Enhances the code with debug statements to aid in debugging.
        
        Args:
            code_string: The original code string
            
        Returns:
            Enhanced code string with debug information
        """
        # Add debug headers and imports if not present
        debug_header = (
            "# === DEBUG INSTRUMENTATION ===\n"
            "import traceback\n"
            "# === END DEBUG HEADER ===\n\n"
        )
        
        modified_code = debug_header + code_string
        
        # Modify the code to add debug info around solver.solve() calls
        modified_lines = []
        lines = modified_code.split('\n')
        
        for i, line in enumerate(lines):
            modified_lines.append(line)
            
            # Add debugging for if solver.solve():
            if re.search(r'if\s+\w+\.solve\(\)', line):
                # Capture the solver variable name
                solver_var = re.search(r'if\s+(\w+)\.solve\(\)', line)
                if solver_var:
                    solver_name = solver_var.group(1)
                    # Indent level of the original line
                    indent = re.match(r'^(\s*)', line).group(1)
                    next_indent = indent + "    "  # Assume 4-space indentation
                    
                    # Add debug prints after the conditional
                    modified_lines.append(f"{next_indent}print(f\"PYSAT_DEBUG_OUTPUT: model_is_satisfiable=True\")")
                    modified_lines.append(f"{next_indent}print(f\"PYSAT_DEBUG_OUTPUT: solver={solver_name!r}\")")
            
            # Add exception handling around export_solution calls
            if 'export_solution(' in line:
                # Find indentation
                indent = re.match(r'^(\s*)', line).group(1)
                
                # Find the start of the function call
                call_start = line.find('export_solution(')
                
                # If the call is not at the beginning of the line, we need to preserve what comes before it
                prefix = line[:call_start]
                
                # Extract the call and arguments
                call_match = re.search(r'export_solution\((.*)\)', line)
                if call_match:
                    args = call_match.group(1)
                    
                    # Replace the line with a try-except block
                    modified_lines[-1] = f"{indent}try:"
                    modified_lines.append(f"{indent}    {prefix}export_solution({args})")
                    modified_lines.append(f"{indent}except Exception as e:")
                    modified_lines.append(f"{indent}    print(f\"PYSAT_DEBUG_OUTPUT: export_solution_error={{str(e)}}\")")
                    modified_lines.append(f"{indent}    print(f\"PYSAT_DEBUG_OUTPUT: traceback={{traceback.format_exc()}}\")")
                    modified_lines.append(f"{indent}    # Re-raise to ensure proper error handling")
                    modified_lines.append(f"{indent}    raise")
        
        return '\n'.join(modified_lines)
    
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
        
        # First, check if the variable is directly available in the solution
        # This handles custom dictionaries like 'casting'
        if variable_name in self.last_solution:
            return {
                "message": f"Value of dictionary '{variable_name}'",
                "success": True,
                "value": self.last_solution[variable_name]
            }
        
        # Then check in the values dictionary for individual variables
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