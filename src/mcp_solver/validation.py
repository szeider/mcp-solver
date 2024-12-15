from minizinc import Model, Solver, Instance, MiniZincError
from typing import Tuple

# Dictionary mapping constraint keywords to required include files
CONSTRAINT_INCLUDES = {
    "alldifferent": "globals.mzn",
    "circuit": "globals.mzn", 
    "all_different": "globals.mzn",
    "lex_less": "lex.mzn",
    "lex_lesseq": "lex.mzn",
    "value_precede": "value_precede.mzn",
    "inverse": "globals.mzn",
    "regular": "globals.mzn",
    "table": "globals.mzn",
    "member": "globals.mzn"
}

def add_required_includes(model_str: str) -> str:
    """Add required include statements based on constraints used in the model."""
    needed_includes = set()
    for keyword, include in CONSTRAINT_INCLUDES.items():
        if keyword in model_str and f'include "{include}";' not in model_str:
            needed_includes.add(f'include "{include}";')
    return '\n'.join(needed_includes) + '\n' + model_str

def validate_minizinc_model(model_str: str) -> Tuple[bool, str | None]:
    """Validates a MiniZinc model string.
    
    Args:
        model_str: The MiniZinc model code to validate
        
    Returns:
        Tuple of (is_valid, error_message_or_None)
    """
    try:
        model = Model()
        model.add_string(model_str)
        solver = Solver.lookup("chuffed")
        Instance(solver, model)
        return True, None
    except MiniZincError as e:
        return False, str(e)
