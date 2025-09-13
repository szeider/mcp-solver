# model_manager.py
# Manages ASP models, runs clingo, and parses results

from datetime import timedelta

import clingo

from ..core.base_model_manager import BaseModelManager

# --- Integration: Import new error handling and solution modules ---
from . import error_handling, solution


class ModelError(Exception):
    """Custom exception for model-related errors"""

    pass


class ASPModelManager(BaseModelManager):
    """
    ASP model manager implementation using clingo.
    Manages ASP models, runs clingo, and parses answer sets.
    """

    def __init__(self):
        super().__init__()
        self.initialized = True
        self.last_solution = None
        self._last_raw_result = None

    async def _validate_model(
        self, items: list[str], validate_grounding: bool = True
    ) -> tuple[str, clingo.Control | None]:
        """
        Validates the full model through syntax and optionally grounding checks.

        Args:
            items: List of ASP code fragments to validate
            validate_grounding: If True, also checks for grounding errors

        Returns:
            Tuple of (list of error messages, Control object if grounding successful)
            The Control object is only returned if validate_grounding is True and grounding succeeds
        """
        asp_code = "\n".join(items)
        if not asp_code.strip():
            return "", None  # Empty model is valid

        # REMOVED: The call to error_handling.validate_asp_code is no longer needed.
        # We now rely solely on Clingo for validation, which properly supports
        # optimization statements (#maximize, #minimize) and weak constraints (:~)

        if not validate_grounding:
            # If only syntax check was desired, we still use clingo for basic validation
            try:
                ctl = clingo.Control()
                ctl.add("base", [], asp_code)
            except Exception as e:
                return f"Syntax error: {e}", None
            return "", None

        # Now check grounding using clingo
        messages = []

        def logger(_, message):
            messages.append(message)

        try:
            ctl = clingo.Control(logger=logger)
            ctl.add("base", [], asp_code)
            ctl.ground([("base", [])])
        except Exception as e:
            # Capture runtime errors from clingo during grounding
            error_message = str(e)
            if messages:
                error_message = "Grounding error:\n" + "\n".join(messages)
            return error_message, None

        if messages:
            # Capture warnings or errors reported via the logger
            return "Grounding error:\n" + "\n".join(messages), None

        return "", ctl

    async def add_item(self, index: int, content: str) -> dict:
        """
        Attempt to add an item (ASP code fragment) at the specified index.

        This method implements the MCP editing protocol's incremental validation:
        1. It creates a hypothetical model with the new item inserted.
        2. It validates the entire model using clingo to ensure it is syntactically correct and groundable.
        3. If validation succeeds, the item is added to the model.
        4. If validation fails, the change is rejected and an error is returned.

        Args:
            index: The position at which to insert the new item (0-based).
            content: The ASP code fragment to add.

        Returns:
            A dictionary indicating success or failure, and an error message if rejected.
        """
        if not (0 <= index <= len(self.code_items)):
            return {
                "success": False,
                "error": f"Invalid index {index}. Valid range: 0 to {len(self.code_items)}",
            }

        hypothetical_items = self.code_items[:]
        hypothetical_items.insert(index, content)

        validation_errors, _ = await self._validate_model(hypothetical_items)
        if validation_errors:
            return {
                "success": False,
                "error": f"Validation failed: {validation_errors}",
            }

        return await super().add_item(index, content)

    async def delete_item(self, index: int) -> dict:
        """
        Attempt to delete an item (ASP code fragment) at the specified index.

        This method implements incremental validation:
        1. It creates a hypothetical model with the item at the given index removed.
        2. It validates the entire model using clingo to ensure it remains groundable.
        3. If validation succeeds, the item is deleted from the model.
        4. If validation fails, the change is rejected and an error is returned.

        Args:
            index: The position of the item to delete (0-based).

        Returns:
            A dictionary indicating success or failure, and an error message if rejected.
        """
        if not (0 <= index < len(self.code_items)):
            return {
                "success": False,
                "error": f"Invalid index {index}. Valid range: 0 to {len(self.code_items) - 1}",
            }

        hypothetical_items = self.code_items[:]
        del hypothetical_items[index]

        validation_errors, _ = await self._validate_model(hypothetical_items)
        if validation_errors:
            return {
                "success": False,
                "error": f"Validation failed after deletion: {validation_errors}",
            }

        return await super().delete_item(index)

    async def replace_item(self, index: int, content: str) -> dict:
        """
        Attempt to replace an item (ASP code fragment) at the specified index.

        This method implements the MCP editing protocol's incremental validation:
        1. It creates a hypothetical model with the item at the given index replaced.
        2. It validates the entire model using clingo to ensure it is syntactically correct and groundable.
        3. If validation succeeds, the replacement is committed to the model.
        4. If validation fails, the change is rejected and an error is returned.

        Args:
            index: The position of the item to replace (0-based).
            content: The new ASP code fragment to use as replacement.

        Returns:
            A dictionary indicating success or failure, and an error message if rejected.
        """
        if not (0 <= index < len(self.code_items)):
            return {
                "success": False,
                "error": f"Invalid index {index}. Valid range: 0 to {len(self.code_items) - 1}",
            }

        hypothetical_items = self.code_items[:]
        hypothetical_items[index] = content

        validation_errors, _ = await self._validate_model(hypothetical_items)
        if validation_errors:
            return {
                "success": False,
                "error": f"Validation failed: {validation_errors}",
            }

        return await super().replace_item(index, content)

    async def solve_model(self, timeout: timedelta) -> dict:
        """
        Solve the current ASP model using clingo, with enhanced error handling and solution formatting.
        Args:
            timeout: Maximum time to spend on solving
        Returns:
            A dictionary with the solving result and answer sets, or a structured error solution
        """
        if not self.code_items:
            error_sol = solution.export_solution(
                error_handling.ASPError(
                    "No model items to solve", context="Empty model"
                )
            )
            self.last_solution = error_sol
            return error_sol

        # Validate and get pre-grounded control object if successful
        validation_errors, ctl = await self._validate_model(self.code_items)
        if validation_errors:
            err = error_handling.ASPError(
                "Model validation failed", context=validation_errors
            )
            error_sol = solution.export_solution(err)
            self.last_solution = error_sol
            return error_sol

        answer_sets = []
        try:

            def on_model(model):
                atoms = [str(atom) for atom in model.symbols(shown=True)]
                answer_sets.append(atoms)

            with ctl.solve(on_model=on_model, async_=True) as handle:
                handle.wait(timeout.total_seconds())
                handle.cancel()
                handle.get()
            stats = ctl.statistics
            self.last_solve_time = (
                stats.get("summary", {}).get("times", {}).get("solve", 0.0)
            )
            # --- Integration: Use solution.export_solution for standardized output ---
            sol = solution.export_solution(answer_sets)
            self.last_solution = sol
            self._last_raw_result = ctl
            return sol
        except Exception as e:
            # --- Integration: Use solution.export_solution for error reporting ---
            error_sol = solution.export_solution(e)
            self.last_solution = error_sol
            return error_sol

    def get_solution(self) -> dict:
        """
        Get the last computed answer sets.
        Returns:
            A dictionary with the answer sets
        """
        if not self.last_solution:
            return {"success": False, "message": "No solution available"}
        return {"success": True, "solution": self.last_solution}

    def get_variable_value(self, variable_name: str) -> dict:
        """
        Get the value of a variable (atom) from the last answer set.
        This searches for atoms starting with the given variable_name in all answer sets.

        Args:
            variable_name: The atom name to search for
        Returns:
            A dictionary with the value(s) of the atom
        """
        if not self.last_solution:
            return {"success": False, "message": "No solution available"}
        # Search for the atom in all answer sets
        results = []
        for answer_set in self.last_solution:
            values = [atom for atom in answer_set if atom.startswith(variable_name)]
            results.append(values)

        if not any(results):
            return {
                "success": False,
                "message": f"Atom starting with '{variable_name}' not found in any answer set.",
            }

        return {"success": True, "variable": variable_name, "values": results}

    def get_solve_time(self) -> dict:
        """
        Get the time taken for the last solve operation.

        Returns:
            A dictionary with the solve time information
        """
        if self.last_solve_time is None:
            return {
                "success": False,
                "message": "No solve operation has been performed",
            }
        return {"success": True, "solve_time": self.last_solve_time, "unit": "seconds"}

    async def clear_model(self) -> dict:
        """Clears the model and resets the solver's internal state."""
        result = await super().clear_model()
        self.last_solution = None
        self._last_raw_result = None
        result["message"] = "ASP model cleared"
        return result
