import asyncio

from minizinc import Model, Solver, Instance, MiniZincError
from typing import Tuple

def add_required_includes(model_str: str) -> str:
    """Add globals.mzn include if not already present."""
    if 'include "globals.mzn";' not in model_str:
        return 'include "globals.mzn";\n' + model_str
    return model_str

async def validate_minizinc_model(model_str: str) -> Tuple[bool, str | None]:
    """Validates a MiniZinc model string."""
    try:
        model = Model()
        model.add_string(model_str)
        solver = Solver.lookup("chuffed")
        instance = Instance(solver, model)
        await asyncio.to_thread(instance.flat)
        return True, None
    except MiniZincError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error during validation: {str(e)}"
