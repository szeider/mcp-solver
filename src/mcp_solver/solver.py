import asyncio
import logging
import re
from datetime import timedelta
from typing import Dict, Optional, Tuple, Any

from minizinc import Model, Instance, Solver, Result
from minizinc.error import MiniZincError

class SolverError(Exception):
    """Custom exception for solver-specific errors"""
    pass

class ModelError(Exception):
    """Custom exception for model-related errors"""
    pass


class SolverSession:
    """Manages a MiniZinc constraint programming session."""

    def __init__(self, solver_name: str = "chuffed"):
        self.current_model: Optional[Model] = None
        self.current_instance: Optional[Instance] = None
        self.solver: Solver = Solver.lookup(solver_name)
        self.defined_parameters: Dict[str, Optional[int]] = {}
        self.param_store: Dict[str, Optional[int]] = {}
        self.last_solve_time: Optional[float] = None
        self.current_solution: Optional[Dict[str, Any]] = None

    async def validate_and_define_model(self, model_str: str) -> Tuple[bool, str]:
        """
        Validates a MiniZinc model, converting fixed integers into parameters
        while leaving 'var int: x' alone.
        """
        try:
            # Step 1: Convert fixed integers to parameters
            updated_model_str, extracted_parameters = self.convert_parameters(model_str)

            # Step 2: Add the include globals statement, just in case
            updated_model_str = 'include "globals.mzn";\n' + updated_model_str

            # Step 3: Validate the modified model
            is_valid, error_msg = await self.validate_minizinc_model(updated_model_str)
            if not is_valid:
                return False, f"Validation failed: {error_msg}"

            # Step 4: Store updated model and parameters
            self.current_model = Model()
            self.current_model.add_string(updated_model_str)
            self.current_instance = Instance(self.solver, self.current_model)
            self.defined_parameters = extracted_parameters

            # Step 5: Initialize param_store with defaults or None
            for name, value in extracted_parameters.items():
                self.param_store[name] = value  # could be an int or None

            # Also set them in the current instance if they have a definite value
            for name, value in self.param_store.items():
                if value is not None:
                    self.current_instance[name] = value

            return True, "Model validated, and parameters initialized dynamically."

        except Exception as e:
            logging.error("Error during model processing", exc_info=True)
            return False, f"Error during model processing: {str(e)}"

    def convert_parameters(self, model_str: str) -> Tuple[str, Dict[str, int]]:
        """
        Converts integer declarations with values into parameters and tracks placeholders.
        Ensures arrays are not mistakenly converted.
        """
        # Match scalar integer declarations: int: name = value;
        scalar_pattern = r"^\s*int:\s*(\w+)\s*(?:=\s*(\d+))?;"  # Only process lines starting with 'int:'

        parameters = {}
        placeholder_flags = {}

        # Split model string into lines for processing
        lines = model_str.splitlines()
        updated_lines = []

        for line in lines:
            match = re.match(scalar_pattern, line)
            if match:
                name, value = match.groups()
                if value:  # Parameter has a value
                    parameters[name] = int(value)
                    placeholder_flags[name] = False  # Not a placeholder
                else:  # Uninitialized parameter
                    parameters[name] = 0  # Placeholder value
                    placeholder_flags[name] = True  # Mark as placeholder
                # Replace the line with a parameter declaration
                updated_lines.append(f"int: {name};")
            else:
                # Keep non-matching lines unchanged
                updated_lines.append(line)

        updated_model = "\n".join(updated_lines)
        self.placeholder_flags = placeholder_flags  # Store placeholder tracking
        return updated_model, parameters

   
   

    async def validate_minizinc_model(self, model_str: str) -> Tuple[bool, str]:
        """
        Validates a MiniZinc model string.
        """
        try:
            model = Model()
            model.add_string(model_str)
            instance = Instance(self.solver, model)
            # Flatten to check syntax or semantics
            await asyncio.to_thread(instance.flat)
            return True, ""
        except MiniZincError as e:
            return False, f"MiniZinc error: {str(e)}"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"

    def set_parameter(self, param_name: str, param_value: int) -> None:
        """
        Sets a parameter value in the MiniZinc instance.
        """
        if not self.current_model:
            raise ModelError("No model has been defined yet")

        if param_name not in self.defined_parameters:
            raise ModelError(f"Parameter '{param_name}' is not defined as a parameter in the model.")

        try:
            param_value = int(param_value)  # ensure int
            self.param_store[param_name] = param_value

            # Rebuild instance with the updated param_store
            new_instance = Instance(self.solver, self.current_model)

            for existing_param, value in self.param_store.items():
                if value is not None:
                    new_instance[existing_param] = value

            self.current_instance = new_instance
            logging.info(f"Parameter '{param_name}' set to {param_value}.")

        except Exception as e:
            raise ModelError(f"Failed to set parameter '{param_name}': {str(e)}")

    async def solve_model(self, timeout: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Solves the current model instance.
        """
        timeout = timeout or timedelta(seconds=30)
        if not self.current_instance:
            raise ModelError("No model has been submitted yet")

        # Ensure all parameters are assigned before solving
        for pname, pval in self.param_store.items():
            if pval is None:
                raise ModelError(f"Parameter '{pname}' is not set. Please assign a value before solving.")

        try:
            result: Result = await asyncio.to_thread(
                self.current_instance.solve,
                timeout=timeout
            )
            return self._process_result(result)

        except MiniZincError as e:
            logging.error("MiniZinc error", exc_info=True)
            return {
                "status": "ERROR",
                "message": f"MiniZinc error: {str(e)}",
            }

    def _process_result(self, result: Result) -> Dict[str, Any]:
        """
        Processes the MiniZinc result and extracts the solution.
        """
        time_stat = result.statistics.get("time", 0)
        if isinstance(time_stat, timedelta):
            self.last_solve_time = time_stat.total_seconds()
        else:
            self.last_solve_time = float(time_stat)

        if result.status.has_solution():
            self.current_solution = result.solution
            return {
                "status": "SUCCESS",
                "solution": self.current_solution,
                "solve_time": self.last_solve_time,
            }

        return {
            "status": "UNSATISFIABLE",
            "message": "No solution found."
        }

    def get_solve_time(self) -> Optional[float]:
        """Return the most recent solve time if available."""
        return self.last_solve_time

    def get_current_solution(self) -> Optional[Dict[str, Any]]:
        """Return the most recent solution if available."""
        return self.current_solution

    def get_variable_value(self, variable_name: str) -> Optional[Any]:
        """Return the value of a specific variable from the current solution."""
        if not self.current_solution:
            return None
        return self.current_solution.get(variable_name)

    def get_solver_state(self) -> str:
        """Return the current state of the solver."""
        if not self.current_model:
            return "No model loaded"
        if not self.current_solution:
            return "Model loaded but not solved"
        return f"Model solved (took {self.last_solve_time:.3f} seconds)"
