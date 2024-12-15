import asyncio
import logging
import time
from enum import Enum
from typing import Any, Dict, Optional, Tuple
from minizinc import Model, Instance, Solver, Result

logger = logging.getLogger(__name__)

class SolverState(Enum):
    IDLE = "idle"
    BUSY = "busy"
    DONE = "done"
    ERROR = "error"

class SolverSession:
    """Manages a MiniZinc constraint programming session."""

    def __init__(self, solver_name: str = "chuffed"):
        """Initialize a new solver session."""
        self.current_model: Optional[Model] = None
        self.current_instance: Optional[Instance] = None
        self.solver = Solver.lookup(solver_name)
        self.current_solution: Optional[Dict[str, Any]] = None
        self.param_store: Dict[str, Any] = {}
        self.required_parameters: Optional[list] = []
        self.solve_time: Optional[float] = None
        self.state: SolverState = SolverState.IDLE
        self.solve_task: Optional[asyncio.Task] = None
        self.start_time: Optional[float] = None

    def validate_and_define_model(self, model_str: str) -> Tuple[bool, str]:
        """Validate and define a new MiniZinc model.
        
        Instructions:
        - Declare parameters without default values (e.g., `int: param;`).
        - All parameters are assumed to be integers.
        """
        from .validation import validate_minizinc_model, add_required_includes
        
        logger.debug("Validating model:\n%s", model_str)
        
        # Add required includes before validation
        model_str = add_required_includes(model_str)
        logger.debug("Model after adding required includes:\n%s", model_str)
        
        # Extract undefined parameters (e.g., `int: n;`)
        import re
        parameter_pattern = r"(int): (\w+);"
        self.required_parameters = [match[1] for match in re.findall(parameter_pattern, model_str)]
        logger.debug("Undefined parameters detected: %s", self.required_parameters)
        
        # Validate the model
        is_valid, error_msg = validate_minizinc_model(model_str)
        if not is_valid:
            return False, f"Model validation failed:\n{error_msg}"
        
        try:
            self.current_model = Model()
            self.current_model.add_string(model_str)
            self.current_instance = Instance(self.solver, self.current_model)
            return True, "Model validated and prepared for solving"
        except Exception as e:
            return False, f"Model definition failed: {str(e)}"

    async def solve_model(self, parameters: Optional[Dict[str, Any]] = None, timeout: float = 300) -> Dict[str, Any]:
        """Solve the current model asynchronously."""
        # Check if solver is already running
        if self.state == SolverState.BUSY:
            elapsed = time.perf_counter() - (self.start_time or 0)
            return {
                "status": "BUSY",
                "message": f"Solver is still running (elapsed time: {elapsed:.1f}s). Please try again later.",
                "elapsed_time": elapsed
            }

        if not self.current_instance:
            raise ValueError("No model has been submitted yet. Use validate_and_define_model first.")
        
        # Ensure all required parameters are set
        parameters = parameters or {}
        missing_params = [p for p in self.required_parameters if p not in parameters and p not in self.param_store]
        if missing_params:
            raise ValueError(f"Missing parameter values for: {', '.join(missing_params)}")
        
        # Apply provided parameters
        for name, value in parameters.items():
            self.set_parameter(name, value)
        
        try:
            self.state = SolverState.BUSY
            self.start_time = time.perf_counter()

            # Create a task for the solve operation
            solve_coro = self.current_instance.solve_async()
            self.solve_task = asyncio.create_task(solve_coro)

            # Wait for the result with timeout
            try:
                result: Result = await asyncio.wait_for(self.solve_task, timeout=timeout)
            except asyncio.TimeoutError:
                self.state = SolverState.ERROR
                if self.solve_task:
                    self.solve_task.cancel()
                return {
                    "status": "TIMEOUT",
                    "message": f"Solver exceeded time limit of {timeout} seconds",
                    "elapsed_time": time.perf_counter() - self.start_time
                }

            end_time = time.perf_counter()
            self.solve_time = end_time - self.start_time
            
            status = str(result.status.name)
            solution_dict = {}
            
            if not result.status.has_solution():
                self.current_solution = None
                self.state = SolverState.DONE
                return {
                    "status": status,
                    "message": f"No solution found (status: {status})",
                    "solve_time": self.solve_time
                }
            
            if hasattr(result, 'solution') and result.solution is not None:
                solution_dict = getattr(result.solution, '__dict__', {})
            
            self.current_solution = {
                "status": status,
                "message": "Solution found",
                "solution": solution_dict,
                "solve_time": self.solve_time
            }
            
            self.state = SolverState.DONE
            return {
                "status": status,
                "message": f"Solution found in {self.solve_time:.3f} seconds",
                "solve_time": self.solve_time,
                "solution": solution_dict
            }
        
        except Exception as e:
            self.state = SolverState.ERROR
            error_msg = str(e)
            if "UNSATISFIABLE" in error_msg:
                return {
                    "status": "UNSATISFIABLE",
                    "message": "The model constraints cannot be satisfied - no solution exists",
                    "solve_time": self.solve_time if self.solve_time else None
                }
            elif "UNKNOWN" in error_msg:
                return {
                    "status": "UNKNOWN",
                    "message": "The solver could not determine if a solution exists (timeout or resource limit reached)",
                    "solve_time": self.solve_time if self.solve_time else None
                }
            else:
                raise ValueError(f"Solve operation failed: {error_msg}")

    def set_parameter(self, param_name: str, param_value: Any) -> str:
        """Set or update a parameter value."""
        if not self.current_model:
            raise ValueError("Cannot set parameters - no model defined yet. Submit a model first.")
        
        # Check if this is a known required parameter
        if param_name not in self.required_parameters:
            raise ValueError(f"Unknown parameter: {param_name}. Available parameters: {', '.join(self.required_parameters)}")
        
        # Enforce integer type for parameters
        try:
            param_value = int(param_value)
        except (ValueError, TypeError):
            raise ValueError(f"Parameter '{param_name}' must be an integer.")
        
        # Store the parameter value
        self.param_store[param_name] = param_value
        
        # Recreate the instance with updated parameters
        try:
            self.current_instance = Instance(self.solver, self.current_model)
            self._apply_parameters()  # Apply all stored parameters to the new instance
        except Exception as e:
            logger.error(f"Failed to recreate instance with parameter {param_name}={param_value}: {e}")
            raise ValueError(f"Failed to apply parameter {param_name}={param_value}")
        
        return f"Successfully set parameter '{param_name}' = {param_value}"

    def get_current_solution(self) -> Optional[Dict[str, Any]]:
        """Get the current stored solution."""
        return self.current_solution

    def get_variable_value(self, variable_name: str) -> Optional[Any]:
        """Get a specific variable's value from the solution."""
        if not self.current_solution or "solution" not in self.current_solution:
            return None
        return self.current_solution["solution"].get(variable_name)

    def get_solve_time(self) -> Optional[float]:
        """Get the running time of the last solve operation."""
        return self.solve_time

    def get_solver_state(self) -> Dict[str, Any]:
        """Get current solver state and elapsed time if running."""
        response = {
            "state": self.state.value
        }
        
        if self.state == SolverState.BUSY and self.start_time:
            elapsed = time.perf_counter() - self.start_time
            response["elapsed_time"] = elapsed
            response["message"] = f"Solver is running (elapsed time: {elapsed:.1f}s)"
        
        return response

    def _apply_parameters(self) -> None:
        """Internal helper to apply all stored parameters."""
        if not self.current_instance:
            return
        
        # First analyze the instance to ensure all parameters are recognized
        self.current_instance.analyse()
        
        # Then apply all stored parameters
        for name, value in self.param_store.items():
            try:
                self.current_instance[name] = value
            except Exception as e:
                logger.error(f"Failed to apply parameter {name}={value}: {e}")
                raise ValueError(f"Failed to apply parameter {name}={value}")
