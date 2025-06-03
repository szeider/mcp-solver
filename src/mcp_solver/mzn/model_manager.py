import asyncio
import logging
from datetime import timedelta
from typing import Any

from minizinc import Instance, Model, Result, Solver, Status
from minizinc.error import MiniZincError, SyntaxError, TypeError

from ..core.base_model_manager import BaseModelManager
from ..core.constants import (
    CLEANUP_TIMEOUT,
    MAX_SOLVE_TIMEOUT,
    VALIDATION_TIMEOUT,
)


logger = logging.getLogger(__name__)


def error_response(
    code: str, message: str, details: dict | None = None
) -> dict[str, Any]:
    """Helper function to create a standardized error response."""
    return {"error": {"code": code, "message": message, "details": details or {}}}


class ModelError(Exception):
    """Custom exception for model-related errors"""

    pass


class MiniZincModelManager(BaseModelManager):
    def __init__(self, solver_name: str = "chuffed"):
        super().__init__()
        self.current_solution: Any | None = None
        self.solver = Solver.lookup(solver_name)
        self.current_process = None
        self.cleanup_lock = asyncio.Lock()
        self.solve_progress = 0.0
        self.solve_status = ""
        self.initialized = True

    @property
    def model_string(self) -> str:
        return self._get_full_code()

    def _update_progress(self, progress: float, status: str):
        self.solve_progress = progress
        self.solve_status = status

    def get_solve_progress(self) -> tuple[float, str]:
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
                except TimeoutError:
                    logger.warning("Cleanup timeout reached, forcing process kill")
                    if self.current_process:
                        self.current_process.kill()
                except Exception as e:
                    logger.error(f"Error cleaning up process: {e}")
                finally:
                    self.current_process = None

    async def _validate_hypothetical_model(
        self, proposed_items: list[str], timeout: timedelta | None = None
    ) -> None:
        """Validates a hypothetical model state by creating a temporary instance"""
        timeout = timeout or VALIDATION_TIMEOUT

        try:
            async with asyncio.timeout(timeout.total_seconds()):
                model = Model()
                model_text = "\n".join(proposed_items)
                if model_text.strip():
                    if 'include "globals.mzn"' not in model_text:
                        model_text = 'include "globals.mzn";\n' + model_text
                    model.add_string(model_text)
                    instance = Instance(self.solver, model)
                    instance.analyse()
        except TimeoutError:
            raise ModelError(
                f"Model validation timed out after {timeout.total_seconds()} seconds"
            )
        except MiniZincError as e:
            if isinstance(e, SyntaxError):
                raise ModelError(f"Syntax error: {e!s}")
            elif isinstance(e, TypeError):
                raise ModelError(f"Type error: {e!s}")
            else:
                raise ModelError(f"Model error: {e!s}")

    async def add_item(
        self, index: int, content: str, validation_timeout: timedelta | None = None
    ) -> dict[str, Any]:
        """
        Adds a new item at the specified index (1-based).
        Returns a standardized error response if the index is invalid or validation fails.
        """
        if not content.strip():
            return error_response("EMPTY_CONTENT", "Content is empty")

        # First, let the parent handle the list operation
        result = await super().add_item(index, content)

        if not result.get("success"):
            # Convert parent's error format to MiniZinc's error format
            return error_response(
                "INVALID_INDEX",
                result.get("error", "Invalid index"),
                {"valid_range": f"0-{len(self.code_items)}"},
            )

        # Validate the model after the change
        try:
            await self._validate_hypothetical_model(
                self.code_items, timeout=validation_timeout
            )
        except ModelError as e:
            # Rollback the change
            del self.code_items[index]
            return error_response("MODEL_VALIDATION_FAILED", str(e))

        return {"message": f"Item added\nCurrent model:\n{self.model_string}"}

    async def delete_item(
        self, index: int, validation_timeout: timedelta | None = None
    ) -> dict[str, Any]:
        if not self.code_items:
            return error_response(
                "MODEL_EMPTY",
                "Operation 'delete_item' cannot be performed on an empty model",
            )

        # Store the item to restore if validation fails
        if 0 <= index < len(self.code_items):
            removed_item = self.code_items[index]

        # Let parent handle the delete
        result = await super().delete_item(index)

        if not result.get("success"):
            return error_response(
                "INVALID_INDEX",
                result.get("error", "Invalid index"),
                {"valid_range": f"0-{len(self.code_items) - 1}"},
            )

        # Validate the model after deletion
        try:
            await self._validate_hypothetical_model(
                self.code_items, timeout=validation_timeout
            )
        except ModelError as e:
            # Rollback the change
            self.code_items.insert(index, removed_item)
            return error_response("MODEL_VALIDATION_FAILED", str(e))

        return {"message": f"Item deleted\nCurrent model:\n{self.model_string}"}

    async def replace_item(
        self, index: int, content: str, validation_timeout: timedelta | None = None
    ) -> dict[str, Any]:
        if not self.code_items:
            return error_response(
                "MODEL_EMPTY",
                "Operation 'replace_item' cannot be performed on an empty model",
            )
        if not content.strip():
            return error_response("EMPTY_CONTENT", "Content is empty")

        # Store old content for rollback
        if 0 <= index < len(self.code_items):
            old_content = self.code_items[index]

        # Let parent handle the replace
        result = await super().replace_item(index, content)

        if not result.get("success"):
            return error_response(
                "INVALID_INDEX",
                result.get("error", "Invalid index"),
                {"valid_range": f"0-{len(self.code_items) - 1}"},
            )

        # Validate the model after replacement
        try:
            await self._validate_hypothetical_model(
                self.code_items, timeout=validation_timeout
            )
        except ModelError as e:
            # Rollback the change
            self.code_items[index] = old_content
            return error_response("MODEL_VALIDATION_FAILED", str(e))

        return {"message": f"Item replaced\nCurrent model:\n{self.model_string}"}

    async def clear_model(self) -> dict[str, Any]:
        """Override to maintain MiniZinc-specific state"""
        result = await super().clear_model()
        self.current_solution = None
        self.solve_progress = 0.0
        self.solve_status = ""
        return {"message": "Model cleared"}

    # Reintroduce alias: if a tool call uses "insert_item", it is mapped to add_item.
    insert_item = add_item

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._cleanup()

    async def solve_model(self, timeout: timedelta) -> dict[str, Any]:
        if not self.model_string.strip():
            return error_response(
                "MODEL_EMPTY", "Model is empty. Cannot solve an empty model."
            )
        async with self:
            return await self._solve_model_impl(timeout)

    async def _solve_model_impl(self, timeout: timedelta) -> dict[str, Any]:
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

        except TimeoutError:
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
            return error_response("MINIZINC_ERROR", f"MiniZinc error: {e!s}")

    def _process_result(self, result: Result, timeout_seconds: float) -> dict[str, Any]:
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

        # Store in format compatible with BaseModelManager
        self.last_solution = {
            "satisfiable": solution.get("satisfiable", False),
            "status": str(result.status),
            "values": solution.get("solution", {}),
        }

        if "objective" in solution:
            self.last_solution["objective"] = solution["objective"]
        if "optimal" in solution:
            self.last_solution["optimal"] = solution["optimal"]

        return solution

    def get_solution(self) -> dict[str, Any]:
        if not self.current_solution:
            return error_response(
                "NO_SOLUTION", "No solution is available. Please solve the model first."
            )
        return self._process_result(self.current_solution, 0)

    def get_variable_value(self, variable_name: str) -> dict[str, Any]:
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

    def get_solve_time(self) -> dict[str, Any]:
        if self.last_solve_time is None:
            return error_response(
                "NO_SOLUTION", "No solve time available. Please solve the model first."
            )
        return {"solve_time": self.last_solve_time}
