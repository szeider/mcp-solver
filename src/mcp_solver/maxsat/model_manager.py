"""
MaxSAT model manager implementation using BaseModelManager.

This module implements the SolverManager abstract base class for MaxSAT,
providing methods for managing MaxSAT optimization models.
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


class MaxSATModelManager(BaseModelManager):
    """
    MaxSAT model manager implementation.

    This class manages MaxSAT optimization models, including adding, removing, and modifying
    model items, as well as solving models and extracting optimization solutions.
    """

    def __init__(self):
        """
        Initialize a new MaxSAT model manager.
        """
        super().__init__()
        self.initialized = True
        self.logger = logging.getLogger(__name__)
        self.dict_validator = DictionaryMisuseValidator()
        self.logger.info("MaxSAT model manager initialized")

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
        Solve the current MaxSAT optimization model with a timeout.

        Args:
            timeout: Maximum time to spend on solving

        Returns:
            A dictionary with the solving result, including optimization details
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

                error_response = get_standardized_response(
                    success=False,  # This needs to be false to indicate an error
                    message="Your code contains a common dictionary usage error that will cause unexpected behavior.",
                    error="Dictionary misuse detected",
                    error_details={
                        "errors": error_messages,
                        "suggestion": "Check how you're updating dictionary variables. Use var_dict[key] = value instead of var_dict = value.",
                    },
                )
                # Return the error response with consistent error status flag
                return error_response

            # Get full code using parent's method
            code_string = self._get_full_code()

            # Initialize line_issues before the try block
            line_issues = []

            # Perform static analysis on the code before executing it
            try:
                import ast

                ast_tree = ast.parse(code_string)

                # Check for RC2 usage or other MaxSAT solvers
                maxsat_solver_found = False
                wcnf_found = False
                for node in ast.walk(ast_tree):
                    # Check for RC2 instantiation
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "RC2"
                    ):
                        maxsat_solver_found = True
                        self.logger.debug("Found RC2 solver in code")
                    # Check for with RC2(wcnf) pattern
                    elif (
                        isinstance(node, ast.withitem)
                        and isinstance(node.context_expr, ast.Call)
                        and isinstance(node.context_expr.func, ast.Name)
                        and node.context_expr.func.id == "RC2"
                    ):
                        maxsat_solver_found = True
                        self.logger.debug("Found RC2 context manager in code")
                    # Check for WCNF usage
                    elif (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "WCNF"
                    ):
                        wcnf_found = True
                        self.logger.debug("Found WCNF formula in code")

                if not maxsat_solver_found:
                    self.logger.warning("No MaxSAT solver found in the code")
                    # Create an error response with consistent success=False when there's an error
                    response = {
                        "message": "No MaxSAT solver found in the code. For optimization problems, you need to use RC2 or another MaxSAT solver.",
                        "success": False,
                        "error": "Missing MaxSAT solver",
                        "status": "error",
                        "code_analysis": "Missing RC2 solver instantiation",
                    }
                    return response

                if not wcnf_found:
                    self.logger.warning("No WCNF formula found in the code")
                    # Create an error response with consistent success=False when there's an error
                    response = {
                        "message": "No WCNF formula found in the code. MaxSAT optimization requires a WCNF formula with soft clauses.",
                        "success": False,
                        "error": "Missing WCNF formula",
                        "status": "error",
                        "code_analysis": "Missing WCNF formula instantiation",
                    }
                    return response

                # Check for solver.compute() calls (for RC2)
                compute_calls = 0
                for node in ast.walk(ast_tree):
                    if isinstance(node, ast.Call):
                        if (
                            isinstance(node.func, ast.Attribute)
                            and node.func.attr == "compute"
                            and isinstance(node.func.value, ast.Name)
                        ):
                            compute_calls += 1

                if compute_calls == 0:
                    self.logger.warning("No solver.compute() call found in the code")
                    # Create an error response with consistent success=False when there's an error
                    response = {
                        "message": "No solver.compute() call found in the code. For MaxSAT optimization with RC2, use solver.compute() instead of solver.solve().",
                        "success": False,
                        "error": "Missing compute call",
                        "status": "error",
                        "code_analysis": "Missing solver.compute() call",
                    }
                    return response

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
                    # Generate a warning but don't return an error anymore
                    # since it might be using a different solution export approach
                    if "warnings" not in self.last_solution:
                        self.last_solution["warnings"] = []
                    self.last_solution["warnings"].append(
                        "Recommendation: Use export_solution() to return MaxSAT results. This function is automatically available in the environment."
                    )

                # Check for common logical errors in the code
                line_issues = []

                # Check for incorrect item value summation (a common error)
                for node in ast.walk(ast_tree):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "sum"
                    ):
                        # Check if we're summing a dictionary directly instead of values
                        if len(node.args) == 1 and isinstance(node.args[0], ast.Name):
                            dict_name = node.args[0].id
                            # If it looks like a dictionary with values
                            if dict_name.endswith("_values") or dict_name.endswith(
                                "_weights"
                            ):
                                line_num = getattr(node, "lineno", "?")
                                line_issues.append(
                                    {
                                        "line": line_num,
                                        "issue": f"Incorrect summation: sum({dict_name}) will sum the dictionary keys, not values. Use sum({dict_name}.values()) instead.",
                                    }
                                )

                # Check for incomplete assignments (like missing value after equals sign)
                for i, line in enumerate(code_string.split("\n")):
                    # Look for lines that end with an equals sign or have incomplete assignments
                    if re.search(r"=\s*$", line) or re.search(r"=\s*#", line):
                        line_issues.append(
                            {
                                "line": i + 1,
                                "issue": "Incomplete assignment: Line ends with '=' but has no value assigned.",
                            }
                        )

                    # Check for incomplete wcnf.append() calls - a common error
                    if re.search(r"wcnf\.append\(\s*,\s*weight", line):
                        line_issues.append(
                            {
                                "line": i + 1,
                                "issue": "Incomplete append: wcnf.append call is missing the list of literals. Use wcnf.append([x], weight=1) instead of wcnf.append(, weight=1).",
                            }
                        )

                    # Check for wcnf.append without literals list
                    if re.search(r"wcnf\.append\([^[]", line) and "weight=" in line:
                        # If it has something other than a list but has weight=
                        line_issues.append(
                            {
                                "line": i + 1,
                                "issue": "Incorrect append format: Literals must be in a list. Use wcnf.append([x], weight=1) with square brackets.",
                            }
                        )

            except SyntaxError as e:
                line_num = e.lineno if hasattr(e, "lineno") else "?"
                col_num = e.offset if hasattr(e, "offset") else "?"
                error_text = (
                    e.text.strip() if hasattr(e, "text") and e.text else "unknown"
                )

                self.logger.error(
                    f"Syntax error in code at line {line_num}, column {col_num}: {e!s}"
                )

                # Find the actual line of code in the user's original code
                # by looking at the line numbers in the original code items
                original_line = "unknown"
                original_item = None
                if isinstance(line_num, int):
                    line_count = 0
                    for item_index, item_content in enumerate(self.code_items):
                        item_lines = item_content.count("\n") + 1
                        if line_count + item_lines >= line_num:
                            # This is the item containing the error
                            original_item = item_index + 1  # 1-based indexing
                            relative_line = line_num - line_count
                            item_lines_list = item_content.split("\n")
                            if 0 <= relative_line - 1 < len(item_lines_list):
                                original_line = item_lines_list[relative_line - 1]
                            break
                        line_count += item_lines

                # Create a detailed error response with original line information
                error_response = {
                    "success": False,
                    "message": f"Syntax error at line {line_num}, column {col_num}: {e!s}",
                    "error": "Syntax error",
                    "status": "error",
                    "error_details": {
                        "line": line_num,
                        "column": col_num,
                        "code": error_text,
                        "message": str(e),
                        "original_item": original_item,
                        "original_line": original_line,
                    },
                }
                return error_response
            except Exception as e:
                self.logger.error(f"Error analyzing code: {e!s}")
                # Continue despite analysis error

            # Modify the code to enhance debugging - no need to inject imports
            # since export_maxsat_solution will be made available directly in the environment
            modified_code = self._enhance_code_for_debugging(code_string)

            # Set timeout
            timeout_seconds = timeout.total_seconds()

            # If we found logical issues in static analysis, warn the user before executing
            if line_issues:
                # Format the issues into a warning message
                warnings = [
                    f"Line {issue['line']}: {issue['issue']}" for issue in line_issues
                ]
                warning_msg = (
                    "Potential logical issues detected in your code:\n"
                    + "\n".join(warnings)
                )

                self.logger.warning(f"Logical issues detected: {warnings}")

                # Include warnings but still proceed with execution
                response = get_standardized_response(
                    success=True,  # Not a fatal error
                    message="Your code contains potential logical issues that may affect the result.",
                    warnings=warnings,
                    code_analysis="Potential logical issues detected",
                )

                # Log the warning but proceed with execution

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

                # Include any logical issues we found earlier in the error response
                if line_issues:
                    error_response = {
                        "success": False,
                        "message": f"Error executing code: {error_msg}",
                        "error": "Execution error",
                        "status": "error",
                        "error_details": {
                            "execution_error": error_msg,
                            "logical_issues": [
                                f"Line {issue['line']}: {issue['issue']}"
                                for issue in line_issues
                            ],
                        },
                    }
                else:
                    error_response = {
                        "success": False,
                        "message": f"Error executing code: {error_msg}",
                        "error": "Execution error",
                        "status": "error",
                    }

                # With automatic injection, we shouldn't see NameError for export_maxsat_solution anymore
                # But keep error handling just in case something else goes wrong
                if "NameError" in error_msg and "export_maxsat_solution" in error_msg:
                    error_response["message"] = (
                        "Error accessing export_maxsat_solution function. This is likely an internal error with the MaxSAT environment."
                    )
                    error_response["error"] = (
                        "Error with export_maxsat_solution function"
                    )

                return error_response

            # Extract solver output to check for solution data
            output = self.last_result.get("output", "")
            satisfiable = False
            maxsat_data_present = False

            # Look for MaxSAT marker in the output
            maxsat_marker = re.search(r"_is_maxsat_solution", output)
            if maxsat_marker:
                maxsat_data_present = True
                self.logger.debug("Found MaxSAT solution marker in output")

            # Parse output for explicit satisfiability result
            sat_match = re.search(
                r"PYSAT_DEBUG_OUTPUT: model_is_satisfiable=(\w+)", output
            )
            if sat_match:
                satisfiable = sat_match.group(1).lower() == "true"
                self.logger.debug(
                    f"Found explicit satisfiability result: {satisfiable}"
                )

            # Extract solution if available
            if self.last_result.get("solution"):
                self.last_solution = self.last_result["solution"]
            else:
                # Create a minimal solution with just the satisfiability flag
                self.last_solution = {
                    "satisfiable": satisfiable,
                    "status": "optimal" if satisfiable else "unsatisfiable",
                    "values": {},
                }

            # Ensure there's a 'values' dictionary for standardized access
            if "values" not in self.last_solution:
                self.last_solution["values"] = {}

            # Try to find direct MaxSAT result in the output first
            maxsat_result_pattern = re.compile(r"MaxSAT Result: ({.*})")
            maxsat_result_match = maxsat_result_pattern.search(output)
            if maxsat_result_match:
                try:
                    maxsat_result_str = maxsat_result_match.group(1)
                    self.logger.debug(
                        f"Found direct MaxSAT result output: {maxsat_result_str}"
                    )
                    # Try to evaluate it as a Python dictionary
                    import ast

                    maxsat_data = ast.literal_eval(maxsat_result_str)
                    if isinstance(maxsat_data, dict):
                        self.logger.debug("Successfully parsed direct MaxSAT result")
                        # Create a proper MaxSAT solution structure
                        self.last_solution = {
                            "satisfiable": maxsat_data.get("satisfiable", True),
                            "status": maxsat_data.get("status", "optimal"),
                            "values": maxsat_data.get("values", {}),
                            "maxsat_result": maxsat_data,
                        }
                        # Add direct fields for convenience
                        if "cost" in maxsat_data:
                            self.last_solution["cost"] = maxsat_data["cost"]
                        if "total_value" in maxsat_data:
                            self.last_solution["objective"] = maxsat_data["total_value"]
                        if "selected_items" in maxsat_data:
                            self.last_solution["selected_items"] = maxsat_data[
                                "selected_items"
                            ]
                        # Mark as MaxSAT data
                        maxsat_data_present = True
                except Exception as e:
                    self.logger.error(f"Error parsing direct MaxSAT result: {e!s}")

            # Also try the standard debug output
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

                            # Check for MaxSAT data in the solution
                            if "_is_maxsat_solution" in last_solution_data:
                                maxsat_data_present = True
                                self.logger.debug(
                                    "Found MaxSAT solution marker in data"
                                )

                                # Extract satisfiability from the MaxSAT solution
                                satisfiable = last_solution_data.get(
                                    "satisfiable", False
                                )
                                self.last_solution["satisfiable"] = satisfiable

                                # For MaxSAT, use optimal/unsatisfiable as status
                                self.last_solution["status"] = (
                                    "optimal" if satisfiable else "unsatisfiable"
                                )

                            # Check for maxsat_data or maxsat_result directly
                            if "maxsat_data" in last_solution_data:
                                maxsat_data_present = True
                                self.logger.debug("Found maxsat_data in solution")

                                # Ensure the maxsat_data is properly copied to the solution
                                if isinstance(last_solution_data["maxsat_data"], dict):
                                    maxsat_data = last_solution_data["maxsat_data"]
                                    self.last_solution["maxsat_result"] = maxsat_data

                                    # Extract key metrics
                                    if "cost" in maxsat_data:
                                        self.last_solution["cost"] = maxsat_data["cost"]
                                    if "objective" in maxsat_data:
                                        self.last_solution["objective"] = maxsat_data[
                                            "objective"
                                        ]
                                    if "status" in maxsat_data:
                                        self.last_solution["status"] = maxsat_data[
                                            "status"
                                        ]

                            elif "maxsat_result" in last_solution_data:
                                maxsat_data_present = True
                                self.logger.debug("Found maxsat_result in solution")

                                # Extract satisfiability from MaxSAT result if available
                                if isinstance(
                                    last_solution_data["maxsat_result"], dict
                                ):
                                    maxsat_result = last_solution_data["maxsat_result"]
                                    if "satisfiable" in maxsat_result:
                                        satisfiable = maxsat_result["satisfiable"]
                                        self.last_solution["satisfiable"] = satisfiable

                                    # For MaxSAT, use optimal/unsatisfiable as status
                                    if "status" in maxsat_result:
                                        self.last_solution["status"] = maxsat_result[
                                            "status"
                                        ]
                                    else:
                                        self.last_solution["status"] = (
                                            "optimal"
                                            if satisfiable
                                            else "unsatisfiable"
                                        )

                    except json.JSONDecodeError as e:
                        self.logger.error(f"Error parsing solution JSON: {e!s}")
                        self.logger.debug(
                            f"Problematic JSON string: {last_solution_str[:100]}..."
                        )

                        # Attempt alternative parsing using ast.literal_eval which is more forgiving
                        if self._try_alternative_parsing(last_solution_str):
                            # Check for MaxSAT data after alternative parsing
                            if (
                                "_is_maxsat_solution" in self.last_solution
                                or "maxsat_data" in self.last_solution
                            ):
                                maxsat_data_present = True
                                self.logger.debug(
                                    "Found MaxSAT data after alternative parsing"
                                )

                        # Even with parsing error, we'll keep the solution data we have
                        self.last_solution["warning"] = (
                            f"Solution parsing warning: {e!s}"
                        )
                except Exception as e:
                    self.logger.error(f"Error extracting solution: {e!s}")
                    self.last_solution["warning"] = f"Solution extraction error: {e!s}"

            # Add solve time to solution
            self.last_solution["solve_time"] = f"{self.last_solve_time:.6f} seconds"

            # Verify this is actually a MaxSAT optimization problem
            if not maxsat_data_present:
                self.logger.warning("No MaxSAT data found in the solution")
                warning_msg = (
                    "Solution does not contain MaxSAT optimization data. "
                    "Make sure you're using a WCNF formula with soft clauses, an RC2 solver, "
                    "and calling export_maxsat_solution() to return results."
                )

                if "warnings" not in self.last_solution:
                    self.last_solution["warnings"] = []
                self.last_solution["warnings"].append(warning_msg)

            # Prepare the optimization-specific response
            is_optimal = self.last_solution.get("status") == "optimal"

            # Construct message based on solution data
            if satisfiable:
                message = "MaxSAT optimization completed successfully"
                if is_optimal:
                    message += " (optimal solution found)"
                else:
                    message += " (feasible solution found)"

                # Add cost/objective details if available
                if "cost" in self.last_solution:
                    message += f" with cost: {self.last_solution['cost']}"
                elif "objective" in self.last_solution:
                    message += f" with objective: {self.last_solution['objective']}"
            else:
                message = "MaxSAT optimization completed (no feasible solution found)"
                if (
                    "status" in self.last_solution
                    and self.last_solution["status"] != "unsatisfiable"
                ):
                    message += f" - status: {self.last_solution['status']}"

            # Build the response
            response = {
                "message": message,
                "success": True,
                "solve_time": f"{self.last_solve_time:.6f} seconds",
                "output": output,
                "satisfiable": satisfiable,
                "status": self.last_solution.get(
                    "status", "optimal" if satisfiable else "unsatisfiable"
                ),
                "is_optimization": True,
            }

            # Include optimization-specific fields
            if "cost" in self.last_solution:
                response["cost"] = self.last_solution["cost"]
            if "objective" in self.last_solution:
                response["objective"] = self.last_solution["objective"]

            # Include the maxsat_result if available
            if "maxsat_result" in self.last_solution:
                response["maxsat_result"] = self.last_solution["maxsat_result"]

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

        # Handle MaxSAT specific patterns
        # Fix any stray double quotes that might be inside strings
        # Avoid using look-behind/look-ahead which can cause regex errors
        clean_str = clean_str.replace('""', '\\"')

        # Replace NaN and Infinity with null (which is JSON compatible)
        clean_str = re.sub(r"NaN", "null", clean_str)
        clean_str = re.sub(r"Infinity", "null", clean_str)

        # Try to normalize any maxsat_data structure that might be causing issues
        if "maxsat_data" in clean_str:
            self.logger.debug("Found maxsat_data in solution, cleaning up structure")

            # Try to handle double quotes in maxsat_data fields
            # This is tricky but helps with common extraction issues
            clean_str = re.sub(
                r'(maxsat_data":\s*{"[^}]*})',
                lambda m: m.group(1).replace('\\"', "'"),
                clean_str,
            )

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
            # Special handling for list fields
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
            self.last_solution["status"] = "optimal" if is_sat else "unsatisfiable"
            extracted_something = True
            self.logger.debug(f"Extracted satisfiability from regex: {is_sat}")

        # Try to extract cost/objective
        cost_match = re.search(r"'cost':\s*(\d+)", solution_str)
        if cost_match:
            try:
                cost = int(cost_match.group(1))
                self.last_solution["cost"] = cost
                extracted_something = True
                self.logger.debug(f"Extracted cost from regex: {cost}")
            except ValueError:
                pass

        objective_match = re.search(r"'objective':\s*(-?\d+)", solution_str)
        if objective_match:
            try:
                objective = int(objective_match.group(1))
                self.last_solution["objective"] = objective
                extracted_something = True
                self.logger.debug(f"Extracted objective from regex: {objective}")
            except ValueError:
                pass

        # Try to extract MaxSAT status
        status_match = re.search(r"'status':\s*\"(\w+)\"", solution_str)
        if status_match:
            status = status_match.group(1)
            self.last_solution["status"] = status
            extracted_something = True
            self.logger.debug(f"Extracted status from regex: {status}")

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

        # Add line number comments to help with error reporting
        lines = modified_code.split("\n")

        # Add line number comments at the start of each line
        lines_with_numbers = []
        for i, line in enumerate(lines):
            # Add comment with line number, but only to non-empty lines
            if line.strip() and not line.strip().startswith("#"):
                # Preserve indentation and add comment
                lines_with_numbers.append(f"{line}  # __LINE_NUMBER_{i + 1}__")
            else:
                lines_with_numbers.append(line)

        modified_code = "\n".join(lines_with_numbers)

        # Modify the code to add debug info around solver.compute() calls
        modified_lines = []
        lines = modified_code.split("\n")

        for _, line in enumerate(lines):
            modified_lines.append(line)

            # Add debugging for if solver.compute():
            if re.search(r"if\s+\w+\.compute\(\)", line):
                # Capture the solver variable name
                solver_var = re.search(r"if\s+(\w+)\.compute\(\)", line)
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

            # Add debugging for model = solver.compute():
            compute_match = re.search(r"(\w+)\s*=\s*(\w+)\.compute\(\)", line)
            if compute_match:
                # Capture both the result variable and solver variable names
                result_var = compute_match.group(1)
                solver_name = compute_match.group(2)
                # Indent level of the original line
                indent = re.match(r"^(\s*)", line).group(1)

                # Add debug prints after the compute call
                # Note: RC2's compute() returns the model directly, not as an attribute
                modified_lines.append(
                    f'{indent}print(f"PYSAT_DEBUG_OUTPUT: compute_returned_value={{True if {result_var} else False}}")'
                )
                modified_lines.append(
                    f'{indent}print(f"PYSAT_DEBUG_OUTPUT: solver={solver_name!r}")'
                )

            # Add exception handling around export_maxsat_solution calls
            if "export_maxsat_solution(" in line:
                # Find indentation
                indent = re.match(r"^(\s*)", line).group(1)

                # Find the start of the function call
                call_start = line.find("export_maxsat_solution(")

                # If the call is not at the beginning of the line, we need to preserve what comes before it
                prefix = line[:call_start]

                # Extract the call and arguments
                call_match = re.search(r"export_maxsat_solution\((.*)\)", line)
                if call_match:
                    args = call_match.group(1)

                    # Replace the line with a try-except block
                    modified_lines[-1] = f"{indent}try:"
                    modified_lines.append(
                        f"{indent}    {prefix}export_maxsat_solution({args})"
                    )
                    modified_lines.append(f"{indent}except Exception as e:")
                    modified_lines.append(
                        f'{indent}    print(f"PYSAT_DEBUG_OUTPUT: export_maxsat_solution_error={{str(e)}}")'
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
        # This handles custom dictionaries like 'selected_features'
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

    def get_optimization_result(self) -> dict[str, Any]:
        """
        Get the optimization result including cost and objective values.

        Returns:
            A dictionary with optimization result information
        """
        if not self.last_solution:
            return {"message": "No solution available", "success": False}

        # Check if this is truly an optimization result
        is_optimization = False
        if (
            "cost" in self.last_solution
            or "objective" in self.last_solution
            or "maxsat_result" in self.last_solution
        ):
            is_optimization = True

        if not is_optimization:
            return {
                "message": "No optimization result available",
                "success": False,
                "error": "Not an optimization problem",
            }

        # Extract optimization data
        result = {
            "message": "Optimization result retrieved",
            "success": True,
            "satisfiable": self.last_solution.get("satisfiable", False),
            "status": self.last_solution.get("status", "unknown"),
        }

        # Add cost/objective if available
        if "cost" in self.last_solution:
            result["cost"] = self.last_solution["cost"]
        if "objective" in self.last_solution:
            result["objective"] = self.last_solution["objective"]

        # Add MaxSAT specific data if available
        if "maxsat_result" in self.last_solution:
            result["maxsat_result"] = self.last_solution["maxsat_result"]

        return result
