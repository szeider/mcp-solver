from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel

class ToolNames(str, Enum):
    SUBMIT_MODEL = "submit-model"
    SOLVE_MODEL = "solve-model"
    GET_SOLUTION = "get-solution"
    SET_PARAMETER = "set-parameter"
    GET_VARIABLE = "get-variable"
    GET_SOLVE_TIME = "get-solve-time"
    GET_SOLVER_STATE = "get-solver-state"

class ModelInput(BaseModel):
    model: str

class ParameterInput(BaseModel):
    param_name: str
    param_value: Any

class VariableInput(BaseModel):
    variable_name: str

class SolverResult(BaseModel):
    status: str
    message: str
    solution: Optional[Dict[str, Any]] = None
    solve_time: Optional[float] = None
