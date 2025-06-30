# model_manager.py
# Manages ASP models, runs clingo, and parses results 

from ..core.base_model_manager import BaseModelManager
from datetime import timedelta
import clingo

class ASPModelManager(BaseModelManager):
    """
    ASP model manager implementation using clingo.
    Manages ASP models, runs clingo, and parses answer sets.
    """
    def __init__(self):
        super().__init__()
        self.initialized = True
        self.models = []  # List of answer sets
        self.solve_time = timedelta(0)

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

        self.models = []

        # Define a callback for each model
        def on_model(model):
            # Get all atoms in the model
            atoms = []
            for atom in model.symbols(shown=True):
                atoms.append(str(atom))
            self.models.append(atoms)

        try:
            with ctl.solve(on_model=on_model, async_=True) as handle:
                handle.wait(timeout.total_seconds())
                handle.cancel()
                handle.get()

            stats = ctl.statistics
            solve_time_sec = stats.get("summary", {}).get("times", {}).get("solve", 0.0)
            self.solve_time = timedelta(seconds=solve_time_sec)

            return {
                "success": True,
                "message": f"Solved ASP model. Found {len(self.models)} answer set(s).",
                "answer_sets": self.models,
            }
        except Exception as e:
            return {"success": False, "message": "Error solving ASP program", "error": str(e)}

    def get_solution(self) -> dict:
        """
        Get the last computed answer sets.
        Returns:
            A dictionary with the answer sets
        """
        if not self.models:
            return {"success": False, "message": "No solution available"}
        return {"success": True, "answer_sets": self.models}

    def get_variable_value(self, variable_name: str) -> dict:
        """
        Get the value of a variable (atom) from the last answer set.
        This searches for atoms starting with the given variable_name in all answer sets.

        Args:
            variable_name: The atom name to search for
        Returns:
            A dictionary with the value(s) of the atom
        """
        if not self.models:
            return {"success": False, "message": "No solution available"}
        # Search for the atom in all answer sets
        results = []
        for answer_set in self.models:
            values = [atom for atom in answer_set if atom.startswith(variable_name)]
            results.append(values)

        if not any(results):
            return {"success": False, "message": f"Atom starting with '{variable_name}' not found in any answer set."}

        return {"success": True, "variable": variable_name, "values": results}

    def get_solve_time(self) -> timedelta:
        """
        Get the time taken to solve the model from clingo statistics.

        Returns:
            Time taken to solve
        """
        return self.solve_time