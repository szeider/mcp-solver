import asyncio
import logging
from datetime import timedelta
from typing import Dict, Optional, Tuple, Any
from minizinc import Model, Instance, Solver, Result
from minizinc.error import MiniZincError
import re

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
        self.defined_parameters: Dict[str, int] = {}
        self.param_store: Dict[str, int] = {}
        self.last_solve_time: Optional[float] = None
        self.current_solution: Optional[Dict[str, Any]] = None

    async def validate_and_define_model(self, model_str: str) -> Tuple[bool, str]:
        """
        Validates a MiniZinc model, converting fixed integers into parameters.

        Args:
            model_str (str): The MiniZinc model string.

        Returns:
            Tuple[bool, str]: Validation success and an informative message.
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

            # Step 5: Set parameters dynamically
            for name, value in extracted_parameters.items():
                self.current_instance[name] = value
                self.param_store[name] = value

            return True, "Model validated, and parameters initialized dynamically."

        except Exception as e:
            logging.error("Error during model processing", exc_info=True)
            return False, f"Error during model processing: {str(e)}"

    def convert_parameters(self, model_str: str) -> Tuple[str, Dict[str, int]]:
        """
        Converts integer declarations with values into parameters.

        Args:
            model_str (str): The MiniZinc model string.

        Returns:
            Tuple[str, Dict[str, int]]: Updated model string and extracted parameters.
        """
        pattern = r"int:\s*(\w+)\s*=\s*(\d+);"
        matches = re.findall(pattern, model_str)
        parameters = {name: int(value) for name, value in matches}

        # Replace all fixed declarations with parameter declarations
        updated_model = re.sub(pattern, r"int: \1;", model_str)
        return updated_model, parameters

    async def validate_minizinc_model(self, model_str: str) -> Tuple[bool, str]:
        """
        Validates a MiniZinc model string.

        Args:
            model_str (str): The MiniZinc model.

        Returns:
            Tuple[bool, str]: Whether validation passed and error message if any.
        """
        try:
            model = Model()
            model.add_string(model_str)
            instance = Instance(self.solver, model)
            await asyncio.to_thread(instance.flat)
            return True, ""
        except MiniZincError as e:
            return False, f"MiniZinc error: {str(e)}"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"

    def set_parameter(self, param_name: str, param_value: int) -> None:
        """
        Sets a parameter value in the MiniZinc instance.

        Args:
            param_name (str): The name of the parameter to set.
            param_value (int): The value to assign to the parameter.

        Raises:
            ModelError: If the parameter is not defined or the instance is not ready.
        """
        if not self.current_model:
            raise ModelError("No model has been defined yet")

        if param_name not in self.defined_parameters:
            raise ModelError(f"Parameter '{param_name}' is not defined in the model.")

        try:
            # Force param_value to be an integer to avoid MiniZinc string-to-int errors
            param_value = int(param_value)

            # Update param_store with the new parameter value
            self.param_store[param_name] = param_value

            # Create a fresh instance
            new_instance = Instance(self.solver, self.current_model)

            # Assign *all* parameters in a single pass
            for existing_param, value in self.param_store.items():
                new_instance[existing_param] = value

            # Replace the current instance
            self.current_instance = new_instance

            logging.info(f"Parameter '{param_name}' set to {param_value}.")

        except Exception as e:
            raise ModelError(f"Failed to set parameter '{param_name}': {str(e)}")

    async def solve_model(self, timeout: Optional[timedelta] = None) -> Dict[str, any]:
        """
        Solves the current model instance.

        Args:
            timeout (Optional[timedelta]): Timeout for the solve process.

        Returns:
            Dict[str, any]: Results including status, solution, and solve time.
        """
        timeout = timeout or timedelta(seconds=30)
        if not self.current_instance:
            raise ModelError("No model has been submitted yet")

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


    def _process_result(self, result: Result) -> Dict[str, any]:
        """
        Processes the MiniZinc result and extracts the solution.
        """
        # Convert timedelta to float seconds
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
