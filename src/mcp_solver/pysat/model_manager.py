"""
PySAT model manager implementation using BaseModelManager.

This module implements the SolverManager abstract base class for PySAT,
providing methods for managing PySAT models.
"""

import logging
import re
import time
from datetime import timedelta
from typing import Any

from ..core.base_model_manager import BaseModelManager
from ..core.constants import MAX_SOLVE_TIMEOUT, MIN_SOLVE_TIMEOUT
from ..core.validation import (
    DictionaryMisuseValidator,
    ValidationError,
    get_standardized_response,
    validate_content,
    validate_python_code_safety,
    validate_timeout,
)
from .environment import execute_pysat_code


class PySATModelManager(BaseModelManager):
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
        self.initialized = True
        self.logger = logging.getLogger(__name__)
        self.dict_validator = DictionaryMisuseValidator()
        self.logger.info("PySAT model manager initialized")

    async def clear_model(self) -> dict[str, Any]:
        """
        Clear the current model.

        Returns:
            A dictionary with a message indicating the model was cleared
        """
        result = await super().clear_model()

        # Reset the dictionary misuse validator
        self.dict_validator = DictionaryMisuseValidator()

        self.logger.info("Model cleared")
        return {"message": "Model cleared successfully"}

    async def add_item(self, index: int, content: str) -> dict[str, Any]:
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
            # Validate content and code safety
            validate_content(content)
            validate_python_code_safety(content)

            # First call parent's add_item to handle list operations
            result = await super().add_item(index, content)

            if not result.get("success"):
                return result

            # Add the code fragment to the dictionary misuse validator
            self.dict_validator.add_fragment(content, index)
            self.logger.debug(f"Added fragment to dictionary validator: item {index}")

            return result

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

    async def replace_item(self, index: int, content: str) -> dict[str, Any]:
        """
        Replace an item in the model at the specified index.

        Args:
            index: The index of the item to replace
            content: The new content of the item

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

            # Add the code fragment to the dictionary misuse validator
            self.dict_validator.add_fragment(content, index)
            self.logger.debug(f"Added fragment to dictionary validator: item {index}")

            return result

        except ValidationError as e:
            error_msg = str(e)
            self.logger.error(f"Validation error in replace_item: {error_msg}")
            return get_standardized_response(
                success=False,
                message=f"Failed to replace item: {error_msg}",
                error=error_msg,
            )
        except Exception as e:
            error_msg = f"Unexpected error in replace_item: {e!s}"
            self.logger.error(error_msg, exc_info=True)
            return get_standardized_response(
                success=False,
                message="Failed to replace item due to an internal error",
                error=error_msg,
            )

    async def solve_model(self, timeout: timedelta) -> dict[str, Any]:
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
                    success=False,
                    message="Model manager not initialized",
                    error="Not initialized",
                )

            if not self.code_items:
                return get_standardized_response(
                    success=False,
                    message="No model items to solve",
                    error="Empty model",
                )

            # Validate timeout
            validate_timeout(timeout, MIN_SOLVE_TIMEOUT, MAX_SOLVE_TIMEOUT)

            # Validate code fragments for dictionary misuse
            dict_validation_result = self.dict_validator.validate()
            if dict_validation_result["has_errors"]:
                # Dictionary misuse detected, format errors for user
                error_messages = []
                for error in dict_validation_result["errors"]:
                    item_index = error.get("item", "?")
                    line_num = error.get("fragment_line", error.get("line", "?"))
                    error_messages.append(
                        f"Item {item_index}, Line {line_num}: {error['message']} {error['suggestion']}"
                    )

                self.logger.warning(f"Dictionary misuse detected: {error_messages}")

                return get_standardized_response(
                    success=False,
                    message="Your code contains a common dictionary usage error that will cause unexpected behavior.",
                    error="Dictionary misuse detected",
                    error_details={
                        "errors": error_messages,
                        "suggestion": "Check how you're updating dictionary variables. Use var_dict[key] = value instead of var_dict = value.",
                    },
                )

            # Get full code using parent's method
            code_string = self._get_full_code()

            # Perform static analysis on the code before executing it
            try:
                import ast

                ast_tree = ast.parse(code_string)

                # Check for solver.solve() patterns
                solve_calls = 0
                for node in ast.walk(ast_tree):
                    if isinstance(node, ast.Call):
                        if (
                            isinstance(node.func, ast.Attribute)
                            and node.func.attr == "solve"
                            and isinstance(node.func.value, ast.Name)
                        ):
                            solve_calls += 1

                if solve_calls == 0:
                    self.logger.warning("No solver.solve() call found in the code")
                    return get_standardized_response(
                        success=False,
                        message="No solver.solve() call found in the code. Make sure to create a solver and call its solve() method.",
                        error="Missing solve call",
                        code_analysis="Missing solver.solve() call",
                    )

                # Check for export_solution calls
                export_calls = 0
                for node in ast.walk(ast_tree):
                    if isinstance(node, ast.Call):
                        if (
                            isinstance(node.func, ast.Name)
                            and node.func.id == "export_solution"
                        ):
                            export_calls += 1

                if export_calls == 0:
                    self.logger.warning("No export_solution() call found in the code")
                    return get_standardized_response(
                        success=False,
                        message="No export_solution() call found in the code. Make sure to call export_solution() with your result.",
                        error="Missing export_solution call",
                        code_analysis="Missing export_solution() call",
                    )

                # AST-based validation has replaced these regex checks
                line_issues = []

            except SyntaxError as e:
                line_num = e.lineno if hasattr(e, "lineno") else "?"
                col_num = e.offset if hasattr(e, "offset") else "?"
                error_text = (
                    e.text.strip() if hasattr(e, "text") and e.text else "unknown"
                )

                self.logger.error(
                    f"Syntax error in code at line {line_num}, column {col_num}: {e!s}"
                )
                return get_standardized_response(
                    success=False,
                    message=f"Syntax error at line {line_num}, column {col_num}: {e!s}",
                    error="Syntax error",
                    error_details={
                        "line": line_num,
                        "column": col_num,
                        "code": error_text,
                        "message": str(e),
                    },
                )
            except Exception as e:
                self.logger.error(f"Error analyzing code: {e!s}")
                # Continue despite analysis error

            # Modify the code to enhance debugging
            modified_code = self._enhance_code_for_debugging(code_string)

            # Set timeout
            timeout_seconds = timeout.total_seconds()

            # Execute code with timeout
            start_time = time.time()
            self.last_result = execute_pysat_code(
                modified_code, timeout=timeout_seconds
            )
            self.last_solve_time = time.time() - start_time

            # Check if there were execution errors
            if self.last_result.get("error"):
                error_msg = self.last_result["error"]
                self.logger.error(f"Error executing code: {error_msg}")

                # If we captured line issues during analysis, include them in error details
                if line_issues:
                    return get_standardized_response(
                        success=False,
                        message=f"Error executing code: {error_msg}",
                        error="Execution error",
                        error_details={
                            "execution_error": error_msg,
                            "code_issues": line_issues,
                        },
                    )

                return get_standardized_response(
                    success=False,
                    message=f"Error executing code: {error_msg}",
                    error="Execution error",
                )

            # Extract solver output to check for satisfiability
            output = self.last_result.get("output", "")
            satisfiable = False

            # Parse output for explicit satisfiability result
            sat_match = re.search(
                r"PYSAT_DEBUG_OUTPUT: model_is_satisfiable=(\w+)", output
            )
            if sat_match:
                satisfiable = sat_match.group(1).lower() == "true"
                self.logger.debug(
                    f"Found explicit satisfiability result: {satisfiable}"
                )
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
                    "values": {},
                }

            # Ensure there's a 'values' dictionary for standardized access
            if "values" not in self.last_solution:
                self.last_solution["values"] = {}

            # Extract solution data from the debug output if available
            # Look for the _LAST_SOLUTION debug output which contains the complete solution data
            last_solution_pattern = re.compile(r"DEBUG - _LAST_SOLUTION set to: (.*)")
            last_solution_match = last_solution_pattern.search(output)
            if last_solution_match:
                try:
                    last_solution_str = last_solution_match.group(1)
                    # Preprocess string to improve JSON compatibility
                    last_solution_str = self._prepare_solution_string_for_json(
                        last_solution_str
                    )

                    # Try to parse as JSON
                    import json

                    try:
                        last_solution_data = json.loads(last_solution_str)
                        if isinstance(last_solution_data, dict):
                            # Enhanced copy mechanism for all solution data
                            self._merge_solution_data(last_solution_data)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Error parsing solution JSON: {e!s}")
                        self.logger.debug(
                            f"Problematic JSON string: {last_solution_str[:100]}..."
                        )

                        # Attempt alternative parsing using ast.literal_eval which is more forgiving
                        self._try_alternative_parsing(last_solution_str)

                        # Even with parsing error, we'll keep the solution data we have
                        self.last_solution["warning"] = (
                            f"Solution parsing warning: {e!s}"
                        )
                except Exception as e:
                    self.logger.error(f"Error extracting solution: {e!s}")
                    self.last_solution["warning"] = f"Solution extraction error: {e!s}"

            # Add solve time to solution
            self.last_solution["solve_time"] = f"{self.last_solve_time:.6f} seconds"

            # Add any warnings from static analysis
            if line_issues:
                self.last_solution["warnings"] = [
                    f"Line {issue['line']}: {issue['issue']}" for issue in line_issues
                ]

            # Standard message for SAT problems
            message = "Model solved successfully" + (
                " (satisfiable)" if satisfiable else " (unsatisfiable)"
            )

            response = {
                "message": message,
                "success": True,
                "solve_time": f"{self.last_solve_time:.6f} seconds",
                "output": output,
                "satisfiable": satisfiable,
            }

            # Include status and values - ensure consistency
            if satisfiable:
                response["status"] = "sat"
                response["satisfiable"] = True  # Ensure this is explicitly set
            else:
                response["status"] = "unsat"
                response["satisfiable"] = False  # Ensure this is explicitly set

            # Override with solution-specific status if available
            if self.last_solution.get("status"):
                response["status"] = self.last_solution["status"]

            # Include values dictionary
            if self.last_solution.get("values"):
                response["values"] = self.last_solution["values"]

            # Include warnings if any
            if self.last_solution.get("warnings"):
                response["warnings"] = self.last_solution["warnings"]

            return response

        except Exception as e:
            # Log the error
            self.logger.error(f"Error in solve_model: {e!s}", exc_info=True)

            # Return a structured error response
            return get_standardized_response(
                success=False,
                message=f"Error solving model: {e!s}",
                error="Internal error",
            )

    def _prepare_solution_string_for_json(self, solution_str):
        """
        Prepare a solution string for JSON parsing by handling common formatting issues.

        Args:
            solution_str: The solution string extracted from output

        Returns:
            A cleaned string ready for JSON parsing
        """
        # Replace Python syntax with JSON syntax
        clean_str = solution_str.replace("'", '"')
        clean_str = clean_str.replace("True", "true").replace("False", "false")
        clean_str = clean_str.replace("None", "null")

        # Fix common tuple formatting issues (convert Python tuples to JSON arrays)
        # Handle simple tuples with two numbers
        clean_str = re.sub(r"\((\d+),\s*(\d+)\)", r"[\1, \2]", clean_str)

        # Handle tuples with three numbers (e.g., in cut_edges)
        clean_str = re.sub(r"\((\d+),\s*(\d+),\s*(\d+)\)", r"[\1, \2, \3]", clean_str)

        # Handle nested tuples in lists
        clean_str = re.sub(r"\[\(", "[[", clean_str)
        clean_str = re.sub(r"\)\]", "]]", clean_str)
        clean_str = re.sub(r"\),\s*\(", "], [", clean_str)

        # Remove trailing commas which are valid in Python but not in JSON
        clean_str = re.sub(r",\s*([}\]])", r"\1", clean_str)

        # Replace NaN and Infinity with null (which is JSON compatible)
        clean_str = re.sub(r"NaN", "null", clean_str)
        clean_str = re.sub(r"Infinity", "null", clean_str)

        return clean_str

    def _merge_solution_data(self, solution_data):
        """
        Merge solution data from extracted output into the last_solution dictionary.

        Args:
            solution_data: Dictionary containing solution data
        """
        # Copy all fields, not just dictionaries
        for key, value in solution_data.items():
            # Special handling for dictionaries
            if isinstance(value, dict):
                if key not in self.last_solution or not isinstance(
                    self.last_solution[key], dict
                ):
                    self.last_solution[key] = {}

                # Merge dictionaries rather than replace
                for inner_key, inner_value in value.items():
                    self.last_solution[key][inner_key] = inner_value

                self.logger.debug(f"Merged dictionary '{key}' into solution")
            # Special handling for list fields (like 'queens' or 'knights')
            elif isinstance(value, list):
                self.last_solution[key] = value
                self.logger.debug(f"Copied list '{key}' to solution")
            # For primitive values
            else:
                self.last_solution[key] = value
                self.logger.debug(f"Copied value '{key}' to solution")

        # Ensure values are properly formatted
        if "values" in solution_data:
            self.logger.debug(
                f"Found values in solution_data: {solution_data['values']}"
            )
            # Convert JSON booleans back to Python booleans
            for key, value in solution_data["values"].items():
                if value is True or value == "true":
                    self.last_solution["values"][key] = True
                elif value is False or value == "false":
                    self.last_solution["values"][key] = False
                else:
                    self.last_solution["values"][key] = value

    def _try_alternative_parsing(self, solution_str):
        """
        Try alternative parsing methods when JSON parsing fails.

        Args:
            solution_str: The solution string that failed JSON parsing

        Returns:
            Boolean indicating whether parsing succeeded
        """
        try:
            # Try using Python's literal_eval which can handle more Python-like syntax
            import ast

            solution_data = ast.literal_eval(solution_str)

            if isinstance(solution_data, dict):
                self.logger.info(
                    "Successfully parsed solution using ast.literal_eval fallback"
                )
                self._merge_solution_data(solution_data)
                return True
        except Exception as e:
            self.logger.debug(f"Alternative parsing also failed: {e!s}")

            # Even if both parsing methods fail, try to extract any useful data using regex
            if self._extract_data_with_regex(solution_str):
                return True

        return False

    def _extract_data_with_regex(self, solution_str):
        """
        Extract critical data using regex patterns when all parsing fails.

        Args:
            solution_str: The solution string that failed parsing

        Returns:
            Boolean indicating whether any useful data was extracted
        """
        extracted_something = False

        # Try to extract satisfiability
        sat_match = re.search(
            r"'satisfiable':\s*(true|false)", solution_str, re.IGNORECASE
        )
        if sat_match:
            is_sat = sat_match.group(1).lower() == "true"
            self.last_solution["satisfiable"] = is_sat
            self.last_solution["status"] = "sat" if is_sat else "unsat"
            extracted_something = True
            self.logger.debug(f"Extracted satisfiability from regex: {is_sat}")

        # Try to extract lists like 'queens' or 'knights' positions
        for list_type in [
            "queens",
            "knights",
            "board_representation",
            "set_s",
            "set_complement",
            "cut_edges",
        ]:
            list_match = re.search(
                f"'{list_type}':\\s*(\\[.*?\\])", solution_str, re.DOTALL
            )
            if list_match:
                try:
                    import ast

                    # Clean up the list string and convert Python to JSON syntax
                    list_str = list_match.group(1).replace("'", '"')
                    list_data = ast.literal_eval(list_str)
                    self.last_solution[list_type] = list_data
                    self.logger.debug(f"Extracted {list_type} list using regex")
                    extracted_something = True
                except Exception:
                    pass  # If this fails, we still continue with other extractions

        return extracted_something

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
        lines = modified_code.split("\n")

        for _, line in enumerate(lines):
            modified_lines.append(line)

            # Add debugging for if solver.solve():
            if re.search(r"if\s+\w+\.solve\(\)", line):
                # Capture the solver variable name
                solver_var = re.search(r"if\s+(\w+)\.solve\(\)", line)
                if solver_var:
                    solver_name = solver_var.group(1)
                    # Indent level of the original line
                    indent = re.match(r"^(\s*)", line).group(1)
                    next_indent = indent + "    "  # Assume 4-space indentation

                    # Add debug prints after the conditional
                    modified_lines.append(
                        f'{next_indent}print(f"PYSAT_DEBUG_OUTPUT: model_is_satisfiable=True")'
                    )
                    modified_lines.append(
                        f'{next_indent}print(f"PYSAT_DEBUG_OUTPUT: solver={solver_name!r}")'
                    )

            # Add exception handling around export_solution calls
            if "export_solution(" in line:
                # Find indentation
                indent = re.match(r"^(\s*)", line).group(1)

                # Find the start of the function call
                call_start = line.find("export_solution(")

                # If the call is not at the beginning of the line, we need to preserve what comes before it
                prefix = line[:call_start]

                # Extract the call and arguments
                call_match = re.search(r"export_solution\((.*)\)", line)
                if call_match:
                    args = call_match.group(1)

                    # Replace the line with a try-except block
                    modified_lines[-1] = f"{indent}try:"
                    modified_lines.append(
                        f"{indent}    {prefix}export_solution({args})"
                    )
                    modified_lines.append(f"{indent}except Exception as e:")
                    modified_lines.append(
                        f'{indent}    print(f"PYSAT_DEBUG_OUTPUT: export_solution_error={{str(e)}}")'
                    )
                    modified_lines.append(
                        f'{indent}    print(f"PYSAT_DEBUG_OUTPUT: traceback={{traceback.format_exc()}}")'
                    )
                    modified_lines.append(
                        f"{indent}    # Re-raise to ensure proper error handling"
                    )
                    modified_lines.append(f"{indent}    raise")

        return "\n".join(modified_lines)

    def get_solution(self) -> dict[str, Any]:
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

    def get_variable_value(self, variable_name: str) -> dict[str, Any]:
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
                "value": self.last_solution[variable_name],
            }

        # Then check in the values dictionary for individual variables
        if "values" not in self.last_solution:
            return {
                "message": "Solution does not contain variable values",
                "success": False,
            }

        values = self.last_solution["values"]
        if variable_name not in values:
            return {
                "message": f"Variable '{variable_name}' not found in solution",
                "success": False,
            }

        return {
            "message": f"Value of variable '{variable_name}'",
            "success": True,
            "value": values[variable_name],
        }

    def get_solve_time(self) -> dict[str, Any]:
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
            "solve_time": f"{self.last_solve_time:.6f} seconds",
        }
