from abc import abstractmethod
from datetime import timedelta
from typing import Any

from .base_manager import SolverManager


class BaseModelManager(SolverManager):
    """Base implementation with common model management functionality."""

    def __init__(self):
        super().__init__()
        self.code_items: list[str] = []  # Changed to simple list
        self.last_result = None
        self.last_solution = None
        self.last_solve_time = None

    async def clear_model(self) -> dict[str, Any]:
        """Clear the current model."""
        self.code_items = []
        self.last_result = None
        self.last_solution = None
        return {"success": True, "message": "Model cleared"}

    async def add_item(self, index: int, content: str) -> dict[str, Any]:
        """Add item with standard list semantics (0-based indexing)."""
        # Validate index (0 to len)
        if index < 0 or index > len(self.code_items):
            return {
                "success": False,
                "error": f"Invalid index {index}. Valid range: 0 to {len(self.code_items)}",
            }

        # Insert at position
        self.code_items.insert(index, content)
        return {"success": True, "message": f"Added item at index {index}"}

    async def delete_item(self, index: int) -> dict[str, Any]:
        """Delete item with standard list semantics (0-based indexing)."""
        if index < 0 or index >= len(self.code_items):
            return {
                "success": False,
                "error": f"Invalid index {index}. Valid range: 0 to {len(self.code_items) - 1}",
            }

        del self.code_items[index]
        return {"success": True, "message": f"Deleted item at index {index}"}

    async def replace_item(self, index: int, content: str) -> dict[str, Any]:
        """Replace item (no shifting) with 0-based indexing."""
        if index < 0 or index >= len(self.code_items):
            return {
                "success": False,
                "error": f"Invalid index {index}. Valid range: 0 to {len(self.code_items) - 1}",
            }

        self.code_items[index] = content
        return {"success": True, "message": f"Replaced item at index {index}"}

    def get_model(self) -> list[tuple[int, str]]:
        """Get current model with 0-based indices."""
        items = []
        for i, content in enumerate(self.code_items):
            items.append((i, content))
        return items

    def _get_full_code(self) -> str:
        """Get the full code as a single string."""
        return "\n\n".join(self.code_items)

    def get_solution(self) -> dict[str, Any]:
        """Get the current solution."""
        if self.last_solution is None:
            return {"error": "No solution available. Please run solve_model first."}
        return self.last_solution

    def get_variable_value(self, variable_name: str) -> dict[str, Any]:
        """Get the value of a variable from the current solution."""
        if self.last_solution is None:
            return {"error": "No solution available. Please run solve_model first."}

        # Check if we have variables in the solution
        if "variables" in self.last_solution:
            variables = self.last_solution["variables"]
            if variable_name in variables:
                return {"variable": variable_name, "value": variables[variable_name]}
            else:
                return {"error": f"Variable '{variable_name}' not found in solution"}
        else:
            return {"error": "Solution does not contain variable information"}

    def get_solve_time(self) -> dict[str, Any]:
        """Get the time taken for the last solve operation."""
        if self.last_solve_time is None:
            return {"error": "No solve operation has been performed yet."}
        return {"solve_time": self.last_solve_time, "unit": "seconds"}

    @abstractmethod
    async def solve_model(self, timeout: timedelta) -> dict[str, Any]:
        """Each mode must implement its own solve logic."""
        pass
