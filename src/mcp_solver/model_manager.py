import asyncio
from typing import Dict, Optional, Tuple, Any
from datetime import timedelta
from minizinc import Model, Instance, Solver, Result, Status
from minizinc.error import MiniZincError
import logging

from .constants import DEFAULT_SOLVE_TIMEOUT, FLATTEN_TIMEOUT

logger = logging.getLogger(__name__)

class ModelError(Exception):
    """Custom exception for model-related errors"""
    pass

class ModelManager:
    """Manages MiniZinc model text and solving operations."""
    
    def __init__(self, solver_name: str = "chuffed"):
        self.model_string = ""
        self.current_solution: Optional[Any] = None
        self.solver = Solver.lookup(solver_name)
        self.last_solve_time: Optional[float] = None
        
    def get_lines(self) -> list[str]:
        """Get model content as list of lines."""
        return self.model_string.splitlines(keepends=True) if self.model_string else []
        
    def edit_range(self, line_start: int, line_end: int | None, new_content: str) -> None:
        """Edit model content within given line range (1-based indexing)."""
        lines = self.get_lines()
        line_start = max(1, line_start) - 1
        
        if not lines:  # Empty or new content
            self.model_string = new_content
            return

        line_end = len(lines) if line_end is None else min(line_end, len(lines))
        if line_start > len(lines):
            line_start = len(lines)
            
        if not new_content.endswith('\n'):
            new_content += '\n'
            
        new_lines = new_content.splitlines(keepends=True)
        lines[line_start:line_end] = new_lines
        self.model_string = ''.join(lines)

    async def validate_model(self) -> Tuple[bool, str]:
        """Validate current model string."""
        if not self.model_string.strip():
            return False, "Model is empty"
            
        try:
            # Add globals include if not present
            model_text = self.model_string
            if 'include "globals.mzn"' not in model_text:
                model_text = 'include "globals.mzn";\n' + model_text
                
            model = Model()
            model.add_string(model_text)
            instance = Instance(self.solver, model)
            # Flatten to check syntax/semantics
            await asyncio.to_thread(instance.flat)
            return True, "Model is valid"
            
        except MiniZincError as e:
            return False, f"MiniZinc error: {str(e)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    async def solve_model(self, timeout: Optional[timedelta] = None) -> Dict[str, Any]:
        """Solve current model and store solution."""
        timeout = timeout or timedelta(seconds=30)
        
        # Validate first
        valid, error = await self.validate_model()
        if not valid:
            return {
                "status": "ERROR",
                "message": f"Model validation failed: {error}"
            }
            
        try:
            model = Model()
            # Add globals include if not present
            model_text = self.model_string
            if 'include "globals.mzn"' not in model_text:
                model_text = 'include "globals.mzn";\n' + model_text
            model.add_string(model_text)
            instance = Instance(self.solver, model)
            
            result: Result = await asyncio.to_thread(
                instance.solve,
                timeout=timeout
            )
            return self._process_result(result, timeout)
            
        except MiniZincError as e:
            logging.error("MiniZinc error", exc_info=True)
            return {
                "status": "ERROR", 
                "message": f"MiniZinc error: {str(e)}"
            }

    def _process_result(self, result: Result, timeout: timedelta) -> Dict[str, Any]:
        """Process solver result and store solution."""
        time_stat = result.statistics.get("time", 0)
        self.last_solve_time = float(time_stat.total_seconds()) if isinstance(time_stat, timedelta) else float(time_stat)

        has_timed_out = self.last_solve_time >= timeout.total_seconds()

        if result.status.has_solution():
            self.current_solution = result.solution
            return {
                "status": "SAT",
                "solution": self.current_solution,
                "solve_time": self.last_solve_time
            }
        elif result.status == Status.UNSATISFIABLE:
            return {
                "status": "UNSAT",
                "message": "Problem is unsatisfiable"
            }
        elif has_timed_out:
            return {
                "status": "TIMEOUT",
                "message": "Solver reached timeout without conclusion"
            }
        else:
            return {
                "status": "UNKNOWN",
                "message": "Solver still running"
            }

    def get_solution(self) -> Optional[Any]:
        """Get most recent solution."""
        return self.current_solution
        
    def get_variable_value(self, variable_name: str) -> Optional[Any]:
        """Get value of specific variable from solution."""
        if not self.current_solution:
            return None
        return getattr(self.current_solution, variable_name, None)
        
    def get_solve_time(self) -> Optional[float]:
        """Get last solve time."""
        return self.last_solve_time