import asyncio
from typing import Dict, Optional, Any, List, Tuple
from datetime import timedelta
from minizinc import Model, Instance, Solver, Result, Status
from minizinc.error import MiniZincError, SyntaxError, TypeError
import logging

from ..core.base_manager import SolverManager
from ..core.constants import (
    MIN_SOLVE_TIMEOUT,
    MAX_SOLVE_TIMEOUT,
    VALIDATION_TIMEOUT,
    CLEANUP_TIMEOUT,
)

logger = logging.getLogger(__name__)


def error_response(
    code: str, message: str, details: Optional[dict] = None
) -> Dict[str, Any]:
    """Helper function to create a standardized error response."""
    return {"error": {"code": code, "message": message, "details": details or {}}}


class ModelError(Exception):
    """Custom exception for model-related errors"""

    pass


class MiniZincModelManager(SolverManager):
    def __init__(self, solver_name: str = "chuffed"):
        super().__init__()
        self.items: List[Tuple[int, str]] = []
        self.current_solution: Optional[Any] = None
        self.solver = Solver.lookup(solver_name)
        self.last_solve_time: Optional[float] = None
        self.current_process = None
        self.cleanup_lock = asyncio.Lock()
        self.solve_progress = 0.0
        self.solve_status = ""
        self.initialized = True

    @property
    def model_string(self) -> str:
        return "\n".join(content for _, content in self.items)

    def get_model(self) -> List[Tuple[int, str]]:
        return self.items

    async def clear_model(self) -> Dict[str, Any]:
        self.items = []
        self.current_solution = None
        self.last_solve_time = None
        self.solve_progress = 0.0
        self.solve_status = ""
        return {"message": "Model cleared"}

    def _update_progress(self, progress: float, status: str):
        self.solve_progress = progress
        self.solve_status = status

    def get_solve_progress(self) -> Tuple[float, str]:
        return self.solve_progress, self.solve_status

    async def _cleanup(self):
        async with self.cleanup_lock:
            if self.current_process:
                try:
                    async with asyncio.timeout(CLEANUP_TIMEOUT.total_seconds()):
                        self.current_process.terminate()
                        await asyncio.sleep(0.1)
                        if self.current_process.is_alive():
                            self.current_process.kill()
                except asyncio.TimeoutError:
                    logger.warning("Cleanup timeout reached, forcing process kill")
                    if self.current_process:
                        self.current_process.kill()
                except Exception as e:
                    logger.error(f"Error cleaning up process: {e}")
                finally:
                    self.current_process = None

    async def _validate_hypothetical_model(
        self, proposed_items: List[Tuple[int, str]], timeout: Optional[timedelta] = None
    ) -> None:
        """Validates a hypothetical model state by creating a temporary instance"""
        timeout = timeout or VALIDATION_TIMEOUT

        try:
            async with asyncio.timeout(timeout.total_seconds()):
                model = Model()
                model_text = "\n".join(content for _, content in proposed_items)
                if model_text.strip():
                    if 'include "globals.mzn"' not in model_text:
                        model_text = 'include "globals.mzn";\n' + model_text
                    model.add_string(model_text)
                    instance = Instance(self.solver, model)
                    instance.analyse()
        except asyncio.TimeoutError:
            raise ModelError(
                f"Model validation timed out after {timeout.total_seconds()} seconds"
            )
        except MiniZincError as e:
            if isinstance(e, SyntaxError):
                raise ModelError(f"Syntax error: {str(e)}")
            elif isinstance(e, TypeError):
                raise ModelError(f"Type error: {str(e)}")
            else:
                raise ModelError(f"Model error: {str(e)}")

    async def add_item(
        self, index: int, content: str, validation_timeout: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Adds a new item at the specified index.
        Returns a standardized error response if the index is invalid or validation fails.
        """
        if not content.strip():
            return error_response("EMPTY_CONTENT", "Content is empty")

        if not 0 <= index <= len(self.items):
            return error_response(
                "INVALID_INDEX",
                f"Index {index} out of bounds (0-{len(self.items)})",
                {"valid_range": f"0-{len(self.items)}"},
            )

        proposed_items = (
            self.items[:index]
            + [(index, content)]
            + [(i + 1, c) for i, c in self.items[index:]]
        )

        try:
            await self._validate_hypothetical_model(
                proposed_items, timeout=validation_timeout
            )
        except ModelError as e:
            return error_response("MODEL_VALIDATION_FAILED", str(e))
        self.items = proposed_items
        return {"message": f"Item added\nCurrent model:\n{self.model_string}"}

    async def delete_item(
        self, index: int, validation_timeout: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        if not self.items:
            return error_response(
                "MODEL_EMPTY",
                "Operation 'delete_item' cannot be performed on an empty model",
            )
        if not 0 <= index < len(self.items):
            return error_response(
                "INVALID_INDEX",
                f"Index {index} out of bounds (0-{len(self.items)-1})",
                {"valid_range": f"0-{len(self.items)-1}"},
            )

        proposed_items = self.items[:index] + [
            (i - 1, c) for i, c in self.items[index + 1 :]
        ]

        try:
            await self._validate_hypothetical_model(
                proposed_items, timeout=validation_timeout
            )
        except ModelError as e:
            return error_response("MODEL_VALIDATION_FAILED", str(e))
        self.items = proposed_items
        return {"message": f"Item deleted\nCurrent model:\n{self.model_string}"}

    async def replace_item(
        self, index: int, content: str, validation_timeout: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        if not self.items:
            return error_response(
                "MODEL_EMPTY",
                "Operation 'replace_item' cannot be performed on an empty model",
            )
        if not content.strip():
            return error_response("EMPTY_CONTENT", "Content is empty")

        if not 0 <= index < len(self.items):
            return error_response(
                "INVALID_INDEX",
                f"Index {index} out of bounds (0-{len(self.items)-1})",
                {"valid_range": f"0-{len(self.items)-1}"},
            )

        proposed_items = (
            self.items[:index] + [(index, content)] + self.items[index + 1 :]
        )

        try:
            await self._validate_hypothetical_model(
                proposed_items, timeout=validation_timeout
            )
        except ModelError as e:
            return error_response("MODEL_VALIDATION_FAILED", str(e))
        self.items = proposed_items
        return {"message": f"Item replaced\nCurrent model:\n{self.model_string}"}

    # Reintroduce alias: if a tool call uses "insert_item", it is mapped to add_item.
    insert_item = add_item

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._cleanup()

    async def solve_model(self, timeout: timedelta) -> Dict[str, Any]:
        if not self.model_string.strip():
            return error_response(
                "MODEL_EMPTY", "Model is empty. Cannot solve an empty model."
            )
        async with self:
            return await self._solve_model_impl(timeout)

    async def _solve_model_impl(self, timeout: timedelta) -> Dict[str, Any]:
        # We should now always have a valid timeout from the server
        if timeout > MAX_SOLVE_TIMEOUT:
            return error_response(
                "TIMEOUT_EXCEEDED",
                f"Timeout {timeout} exceeds maximum allowed timeout {MAX_SOLVE_TIMEOUT}",
            )

        try:
            model = Model()
            model_text = (
                'include "globals.mzn";\n' + self.model_string
                if 'include "globals.mzn"' not in self.model_string
                else self.model_string
            )
            model.add_string(model_text)
            instance = Instance(self.solver, model)

            self._update_progress(0.0, "Starting solve")
            result = await asyncio.wait_for(
                asyncio.to_thread(instance.solve, timeout=timeout),
                timeout=timeout.total_seconds() + 1.0,
            )
            self._update_progress(1.0, "Solve completed")

            return self._process_result(result, timeout.total_seconds())

        except asyncio.TimeoutError:
            self._update_progress(1.0, "Timeout reached")
            if hasattr(instance, "cancel"):
                await instance.cancel()
            return error_response(
                "SOLVER_TIMEOUT",
                "Solver reached timeout without conclusion",
                {"solve_time": timeout.total_seconds()},
            )
        except MiniZincError as e:
            self._update_progress(1.0, "Error occurred")
            logger.error("MiniZinc solve error", exc_info=True)
            return error_response("MINIZINC_ERROR", f"MiniZinc error: {str(e)}")

    def _process_result(self, result: Result, timeout_seconds: float) -> Dict[str, Any]:
        self.current_solution = result
        self.last_solve_time = (
            result.statistics["solveTime"].total_seconds()
            if "solveTime" in result.statistics
            else None
        )

        # Build a standardized solution format
        solution = {"status": str(result.status)}

        if (
            result.status == Status.SATISFIED
            or result.status == Status.ALL_SOLUTIONS
            or result.status == Status.OPTIMAL_SOLUTION
        ):
            solution["satisfiable"] = True

            # Extract solution values
            solution_values = {}
            for name, value in result.solution.__dict__.items():
                if not name.startswith("_"):  # Skip private attributes
                    solution_values[name] = value
            solution["solution"] = solution_values

            # Add objective value if it exists
            if hasattr(result, "objective") and result.objective is not None:
                solution["objective"] = result.objective

            # Add optimization status
            if result.status == Status.OPTIMAL_SOLUTION:
                solution["optimal"] = True
            else:
                solution["optimal"] = False

        elif result.status == Status.UNSATISFIABLE:
            solution["satisfiable"] = False
        else:
            solution["satisfiable"] = False
            solution["message"] = (
                f"Solver status: {result.status} after {timeout_seconds} seconds"
            )

        return solution

    def get_solution(self) -> Dict[str, Any]:
        if not self.current_solution:
            return error_response(
                "NO_SOLUTION", "No solution is available. Please solve the model first."
            )
        return self._process_result(self.current_solution, 0)

    def get_variable_value(self, variable_name: str) -> Dict[str, Any]:
        if not self.current_solution:
            return error_response(
                "NO_SOLUTION", "No solution is available. Please solve the model first."
            )

        if variable_name not in self.current_solution.solution.__dict__:
            return error_response(
                "VARIABLE_NOT_FOUND",
                f"Variable '{variable_name}' not found in solution",
            )

        return {
            "name": variable_name,
            "value": self.current_solution.solution.__dict__[variable_name],
        }

    def get_solve_time(self) -> Dict[str, Any]:
        if self.last_solve_time is None:
            return error_response(
                "NO_SOLUTION", "No solve time available. Please solve the model first."
            )
        return {"solve_time": self.last_solve_time}
