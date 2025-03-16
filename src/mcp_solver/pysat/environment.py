"""
PySAT secure execution environment.

This module provides a secure environment for executing PySAT code,
with timeout handling and output capturing.
"""

import sys
import os
import io
import signal
import traceback
import time
import contextlib
from typing import Dict, Any, Optional, List, Callable
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import re
import random

# Import path management to ensure we get the correct PySAT
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if current_dir in sys.path:
    sys.path.remove(current_dir)
if parent_dir in sys.path:
    sys.path.remove(parent_dir)

# Add site-packages to the front of the path
import site
site_packages = site.getsitepackages()
for p in reversed(site_packages):
    if p not in sys.path:
        sys.path.insert(0, p)

# Import PySAT but protect against failure
try:
    from pysat.formula import CNF, WCNF
    from pysat.solvers import Solver, Glucose3, Glucose4, Lingeling, Cadical153
    from pysat.card import CardEnc, EncType
    from pysat.examples.rc2 import RC2
    import pysat
except ImportError:
    print("PySAT solver not found. Install with: pip install python-sat")
    sys.exit(1)

# Local imports - must be after path adjustment
from .solution import export_solution, _LAST_SOLUTION
from .templates.cardinality_templates import (
    at_most_k, at_least_k, exactly_k
)
from .constraints import (
    at_most_one, exactly_one, 
    implies, mutually_exclusive, if_then_else
)

# Exception for timeouts
class TimeoutException(Exception):
    """Exception raised when code execution times out."""
    pass

@contextmanager
def time_limit(seconds: float):
    """
    Context manager to limit execution time of a code block.
    
    Args:
        seconds: Maximum execution time in seconds
        
    Raises:
        TimeoutException: If execution time exceeds the limit
    """
    def signal_handler(signum, frame):
        raise TimeoutException(f"Execution timed out after {seconds} seconds")
    
    # Set signal handler for SIGALRM
    signal.signal(signal.SIGALRM, signal_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    
    try:
        yield
    finally:
        # Reset signal handler
        signal.setitimer(signal.ITIMER_REAL, 0)

def execute_pysat_code(code_string: str, timeout: float = 5.0) -> Dict[str, Any]:
    """
    Execute PySAT Python code in a secure environment with timeout handling.
    
    Args:
        code_string: The PySAT Python code to execute
        timeout: Maximum execution time in seconds
        
    Returns:
        Dictionary with execution results
    """
    result = {
        "success": False,
        "output": "",
        "error": None,
        "solution": None,
        "execution_time": 0,
    }
    
    # Capture stdout and stderr
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    
    # Start timer
    start_time = time.time()
    
    # Set up restricted globals for execution
    restricted_globals = {
        "__builtins__": {
            # Allow a subset of builtins
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "int": int,
            "isinstance": isinstance,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "print": print,
            "range": range,
            "reversed": reversed,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
        },
        # Add PySAT core classes
        "CNF": CNF,
        "WCNF": WCNF,
        "Solver": Solver,
        # Popular SAT solvers
        "Glucose3": Glucose3,
        "Glucose4": Glucose4,
        "Lingeling": Lingeling,
        "Cadical153": Cadical153,  # Recommended solver
        # Cardinality constraints
        "CardEnc": CardEnc,
        "EncType": EncType,
        # MaxSAT solver
        "RC2": RC2,
        # Solution export
        "export_solution": export_solution,
        # Python standard libraries
        "time": time,
        # Our helper functions for cardinality constraints
        "at_most_k": at_most_k,
        "at_least_k": at_least_k,
        "exactly_k": exactly_k,
        "at_most_one": at_most_one,
        "exactly_one": exactly_one,
        "implies": implies,
        "mutually_exclusive": mutually_exclusive,
        "if_then_else": if_then_else,
        # Random module
        "random": random,
    }
    
    # Process imports and add to globals
    import_pattern = r'^from\s+([\w.]+)\s+import\s+(.+)$|^import\s+([\w.]+)(?:\s+as\s+([\w]+))?$'
    code_lines = code_string.split('\n')
    cleaned_code_lines = []
    
    # Whitelist of allowed modules (restrictive)
    allowed_modules = {
        'pysat', 'pysat.formula', 'pysat.solvers', 'pysat.card', 'pysat.examples', 
        'pysat.examples.rc2', 'time', 'math', 'itertools', 'random', 'collections'
    }
    
    for line in code_lines:
        match = re.match(import_pattern, line.strip())
        if match:
            # Handle "from X import Y" style
            if match.group(1) and match.group(2):
                module_name = match.group(1)
                if module_name in allowed_modules:
                    try:
                        module = __import__(module_name, fromlist=['.'])
                        for item in match.group(2).split(','):
                            item = item.strip()
                            if item:
                                try:
                                    # Add imported item to globals
                                    obj = getattr(module, item)
                                    restricted_globals[item] = obj
                                except AttributeError:
                                    result["output"] += f"Warning: Could not import {item} from {module_name}\n"
                    except ImportError:
                        result["output"] += f"Warning: Could not import module {module_name}\n"
                else:
                    result["output"] += f"Warning: Import of {module_name} not allowed for security reasons\n"
                
                # Skip this line in the executed code
                continue
            
            # Handle "import X" or "import X as Y" style
            elif match.group(3):
                module_name = match.group(3)
                alias = match.group(4) if match.group(4) else module_name
                
                if module_name in allowed_modules:
                    try:
                        module = __import__(module_name)
                        restricted_globals[alias] = module
                    except ImportError:
                        result["output"] += f"Warning: Could not import module {module_name}\n"
                else:
                    result["output"] += f"Warning: Import of {module_name} not allowed for security reasons\n"
                
                # Skip this line in the executed code
                continue
        
        # Keep all non-import lines
        cleaned_code_lines.append(line)
    
    # Rejoin the code without the import statements
    cleaned_code = '\n'.join(cleaned_code_lines)
    
    # Execute the code with timeout
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer), time_limit(timeout):
            # Execute the code in the restricted environment
            exec(cleaned_code, restricted_globals, restricted_globals)
            
            # Check if a solution was exported
            if 'solution' in restricted_globals:
                result["solution"] = restricted_globals['solution']
            elif _LAST_SOLUTION is not None:
                result["solution"] = _LAST_SOLUTION
            
            result["success"] = True
    except TimeoutException as e:
        result["success"] = False
        result["error"] = f"Execution timed out after {timeout} seconds"
    except Exception as e:
        result["success"] = False
        tb = traceback.format_exc()
        result["error"] = f"Error: {str(e)}\n{tb}"
    
    # Calculate execution time
    result["execution_time"] = time.time() - start_time
    
    # Get output from buffers
    result["output"] = stdout_buffer.getvalue()
    if stderr_buffer.getvalue():
        result["output"] += "\nSTDERR:\n" + stderr_buffer.getvalue()
    
    return result 