import asyncio
from typing import Dict, Optional, Any, List, Tuple
from datetime import timedelta
from minizinc import Model, Instance, Solver, Result, Status
from minizinc.error import MiniZincError, SyntaxError, TypeError
import logging

from .constants import (
    DEFAULT_SOLVE_TIMEOUT,
    MAX_SOLVE_TIMEOUT,
    VALIDATION_TIMEOUT,
    CLEANUP_TIMEOUT
)

logger = logging.getLogger(__name__)

class ModelError(Exception):
    """Custom exception for model-related errors"""
    pass

class ModelManager:
    def __init__(self, solver_name: str = "chuffed"):
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
        return '\n'.join(content for _, content in self.items)

    def get_model(self) -> List[Tuple[int, str]]:
        return self.items

    def clear_model(self) -> None:
        self.items = []
        self.current_solution = None
        self.last_solve_time = None
        self.solve_progress = 0.0
        self.solve_status = ""

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
        self, 
        proposed_items: List[Tuple[int, str]], 
        timeout: Optional[timedelta] = None
    ) -> None:
        """Validates a hypothetical model state by creating a temporary instance"""
        timeout = timeout or VALIDATION_TIMEOUT
        
        try:
            async with asyncio.timeout(timeout.total_seconds()):
                model = Model()
                model_text = '\n'.join(content for _, content in proposed_items)
                if model_text.strip():
                    if 'include "globals.mzn"' not in model_text:
                        model_text = 'include "globals.mzn";\n' + model_text
                    model.add_string(model_text)
                    instance = Instance(self.solver, model)
                    instance.analyse()
        except asyncio.TimeoutError:
            raise ModelError(f"Model validation timed out after {timeout.total_seconds()} seconds")
        except MiniZincError as e:
            if isinstance(e, SyntaxError):
                raise ModelError(f"Syntax error: {str(e)}")
            elif isinstance(e, TypeError):
                raise ModelError(f"Type error: {str(e)}")
            else:
                raise ModelError(f"Model error: {str(e)}")

    async def insert_item(
        self, 
        index: int, 
        content: str,
        validation_timeout: Optional[timedelta] = None
    ) -> None:
        if not content.strip():
            raise ModelError("Empty content")
            
        if not 0 <= index <= len(self.items):
            raise ModelError(f"Index {index} out of bounds (0-{len(self.items)})")
            
        proposed_items = (
            self.items[:index] + 
            [(index, content)] + 
            [(i+1, c) for i,c in self.items[index:]]
        )
        
        await self._validate_hypothetical_model(
            proposed_items,
            timeout=validation_timeout
        )
        self.items = proposed_items

    async def delete_item(
        self, 
        index: int,
        validation_timeout: Optional[timedelta] = None
    ) -> None:
        if not 0 <= index < len(self.items):
            raise ModelError(f"Index {index} out of bounds (0-{len(self.items)-1})")
            
        proposed_items = (
            self.items[:index] + 
            [(i-1, c) for i,c in self.items[index+1:]]
        )
        
        await self._validate_hypothetical_model(
            proposed_items,
            timeout=validation_timeout
        )
        self.items = proposed_items

    async def replace_item(
        self, 
        index: int, 
        content: str,
        validation_timeout: Optional[timedelta] = None
    ) -> None:
        if not content.strip():
            raise ModelError("Empty content")
            
        if not 0 <= index < len(self.items):
            raise ModelError(f"Index {index} out of bounds (0-{len(self.items)-1})")
            
        proposed_items = (
            self.items[:index] + 
            [(index, content)] + 
            self.items[index+1:]
        )
        
        await self._validate_hypothetical_model(
            proposed_items,
            timeout=validation_timeout
        )
        self.items = proposed_items

    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._cleanup()

    async def solve_model(self, timeout: Optional[timedelta] = None) -> Dict[str, Any]:
        async with self:
            return await self._solve_model_impl(timeout)

    async def _solve_model_impl(self, timeout: Optional[timedelta] = None) -> Dict[str, Any]:
        if not self.model_string.strip():
            raise ModelError("Model is empty")
            
        timeout = timeout or DEFAULT_SOLVE_TIMEOUT
        if timeout > MAX_SOLVE_TIMEOUT:
            raise ModelError(f"Timeout {timeout} exceeds maximum allowed timeout {MAX_SOLVE_TIMEOUT}")
        
        try:
            model = Model()
            model_text = ('include "globals.mzn";\n' + self.model_string 
                         if 'include "globals.mzn"' not in self.model_string 
                         else self.model_string)
            model.add_string(model_text)
            instance = Instance(self.solver, model)
            
            # Use asyncio.wait_for to enforce timeout
            self._update_progress(0.0, "Starting solve")
            result = await asyncio.wait_for(
                asyncio.to_thread(instance.solve, timeout=timeout),
                timeout=timeout.total_seconds() + 1.0  # Add 1s buffer for MiniZinc overhead
            )
            self._update_progress(1.0, "Solve completed")
            
            return self._process_result(result, timeout.total_seconds())
            
        except asyncio.TimeoutError:
            self._update_progress(1.0, "Timeout reached")
            if hasattr(instance, 'cancel'):
                await instance.cancel()
            return {
                "status": "TIMEOUT",
                "message": "Solver reached timeout without conclusion",
                "solve_time": timeout.total_seconds()
            }
        except MiniZincError as e:
            self._update_progress(1.0, "Error occurred")
            logger.error("MiniZinc solve error", exc_info=True)
            raise ModelError(f"MiniZinc error: {str(e)}")

    def _process_result(self, result: Result, timeout_seconds: float) -> Dict[str, Any]:
        time_stat = result.statistics.get("time", 0)
        self.last_solve_time = float(time_stat.total_seconds()) if isinstance(time_stat, timedelta) else float(time_stat)
        has_timed_out = self.last_solve_time >= timeout_seconds

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
                "message": "Problem is unsatisfiable",
                "solve_time": self.last_solve_time
            }
        elif has_timed_out:
            return {
                "status": "TIMEOUT",
                "message": "Solver reached timeout without conclusion",
                "solve_time": self.last_solve_time
            }
        return {
            "status": "UNKNOWN",
            "message": "Solver still running",
            "solve_time": self.last_solve_time
        }

    def get_solution(self) -> Optional[Any]:
        return self.current_solution
        
    def get_variable_value(self, variable_name: str) -> Optional[Any]:
        if not self.current_solution:
            return None
        return getattr(self.current_solution, variable_name, None)
        
    def get_solve_time(self) -> Optional[float]:
        return self.last_solve_time