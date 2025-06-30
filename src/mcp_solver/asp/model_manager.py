# model_manager.py
# Manages ASP models, runs clingo, and parses results 

from ..core.base_model_manager import BaseModelManager
from datetime import timedelta
import clingo

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

    async def _validate_model(self, items: list[str]):
        """Validates the full model by trying to ground it."""
        asp_code = "\n".join(items)
        if not asp_code.strip():
            return  # Empty model is valid
        
        def logger(_,msg):
            pass
        try:
            ctl = clingo.Control(logger=logger)
            ctl.add("base", [], asp_code)
            ctl.ground([("base", [])])
        except RuntimeError as e:
            raise ModelError(f"ASP Model Error: {e}")

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

        try:
            await self._validate_model(hypothetical_items)
        except ModelError as e:
            return {"success": False, "error": f"Validation failed: {e}"}

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

        try:
            await self._validate_model(hypothetical_items)
        except ModelError as e:
            return {"success": False, "error": f"Validation failed after deletion: {e}"}

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

        try:
            await self._validate_model(hypothetical_items)
        except ModelError as e:
            return {"success": False, "error": f"Validation failed: {e}"}
            
        return await super().replace_item(index, content)

    async def solve_model(self, timeout: timedelta) -> dict:
        """
        Solve the current ASP model using clingo.
        Args:
            timeout: Maximum time to spend on solving
        Returns:
            A dictionary with the solving result and answer sets
        """
        if not self.code_items:
            return {"success": False, "message": "No model items to solve", "error": "Empty model"}
        
        messages = []
        def logger(_, message):
            messages.append(message)

        ctl = clingo.Control(logger=logger)
        asp_code = "\n".join(self.code_items)
        try:
            ctl.add("base", [], asp_code)
            ctl.ground([("base", [])])
        except RuntimeError as e:
            error_message = "\n".join(messages)
            return {"success": False, "message": "Error grounding ASP program", "error": error_message or str(e)}

        answer_sets = []

        def on_model(model):
            # Get all atoms in the model
            atoms = [str(atom) for atom in model.symbols(shown=True)]
            answer_sets.append(atoms)

        try:
            with ctl.solve(on_model=on_model, async_=True) as handle:
                handle.wait(timeout.total_seconds())
                handle.cancel()
                handle.get()

            stats = ctl.statistics
            self.last_solve_time = stats.get("summary", {}).get("times", {}).get("solve", 0.0)
            self.last_solution = answer_sets
            self._last_raw_result = ctl

            return {
                "success": True,
                "message": f"Solved ASP model. Found {len(answer_sets)} answer set(s).",
                "solution": self.last_solution,
                "solve_time": self.last_solve_time,
            }
        except RuntimeError as e:
            return {"success": False, "message": "Error solving ASP program", "error": str(e)}

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
            return {"success": False, "message": f"Atom starting with '{variable_name}' not found in any answer set."}

        return {"success": True, "variable": variable_name, "values": results}

    def get_solve_time(self) -> dict:
        """
        Get the time taken for the last solve operation.

        Returns:
            A dictionary with the solve time information
        """
        if self.last_solve_time is None:
            return {"success": False, "message": "No solve operation has been performed"}
        return {"success": True, "solve_time": self.last_solve_time, "unit": "seconds"}

    async def clear_model(self) -> dict:
        """Clears the model and resets the solver's internal state."""
        result = await super().clear_model()
        self.last_solution = None
        self._last_raw_result = None
        result["message"] = "ASP model cleared"
        return result