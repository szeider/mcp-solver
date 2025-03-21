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
import collections
import itertools
import math
import json
from typing import Dict, Any, Optional, List, Callable
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import re
import random
import logging

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
    # Import our local solution module
    from . import solution
    from .solution import export_solution
except ImportError as e:
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
from .templates.mapping import VariableMap

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

def safe_import(name, *args, **kwargs):
    """A restricted version of __import__ that only allows importing safe modules."""
    ALLOWED_MODULES = {
        # Standard modules
        'math', 'random', 'collections', 'itertools', 're', 'json',
        
        # PySAT modules
        'pysat', 'pysat.formula', 'pysat.solvers', 'pysat.card', 
        'pysat.examples', 'pysat.examples.rc2', 'pysat.pb', 'pysat.engines'
    }
    if name not in ALLOWED_MODULES:
        raise ImportError(f"Module '{name}' is not allowed in the restricted environment")
    return __import__(name, *args, **kwargs)

def execute_pysat_code(code: str, timeout: float = 10.0) -> Dict[str, Any]:
    """
    Executes PySAT Python code in a secure environment.
    
    Args:
        code: The PySAT Python code to execute
        timeout: Maximum execution time in seconds
        
    Returns:
        A dictionary containing the execution results:
        {
            'success': bool,
            'output': str,
            'error': Optional[str],
            'solution': Optional[Dict]
        }
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"Executing PySAT code with timeout {timeout} seconds")
    
    # Initialize result dictionary
    result = {
        'success': False,
        'output': '',
        'error': None,
        'solution': None
    }
    
    # Capture standard output and error
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        # Process imports
        # We need to modify the code to directly import the modules we need
        processed_code = ""
        in_import = False
        import_lines = []
        regular_lines = []
        
        for line in code.split('\n'):
            if line.strip().startswith(('import ', 'from ')) or in_import:
                # Track multiline imports
                if line.strip().endswith('\\'):
                    in_import = True
                else:
                    in_import = False
                import_lines.append(line)
            else:
                regular_lines.append(line)
        
        # Add fixed imports at the top of the code
        processed_code = """
# Standard library imports provided by the environment
import collections
import itertools
import math
import json
import re
import random

# PySAT imports
from pysat.formula import CNF, WCNF
from pysat.solvers import Glucose3, Cadical153
from pysat.card import CardEnc, EncType

# Constraint helper functions
def at_most_one(variables):
    \"\"\"Returns clauses ensuring at most one variable is true\"\"\"
    clauses = []
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            clauses.append([-variables[i], -variables[j]])
    return clauses

def exactly_one(variables):
    \"\"\"Returns clauses ensuring exactly one variable is true\"\"\"
    clauses = []
    # At least one is true
    clauses.append(list(variables))
    # At most one is true
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            clauses.append([-variables[i], -variables[j]])
    return clauses

def implies(a, b):
    \"\"\"Returns a clause representing a -> b (if a then b)\"\"\"
    return [[-a, b]]

def mutually_exclusive(variables):
    \"\"\"Returns clauses ensuring variables are mutually exclusive\"\"\"
    return at_most_one(variables)

def if_then_else(condition, then_var, else_var):
    \"\"\"Returns clauses for if-then-else construct\"\"\"
    return [[-condition, then_var], [condition, else_var]]

# Cardinality template functions
def at_most_k(variables, k):
    \"\"\"Returns clauses ensuring at most k variables are true\"\"\"
    clauses = []
    if k < 0:
        # Trivially unsatisfiable
        clauses.append([])  # Empty clause means contradiction
    elif k == 0:
        # All variables must be false
        for var in variables:
            clauses.append([-var])
    elif k >= len(variables):
        # Trivially satisfiable
        pass
    else:
        # For each combination of k+1 variables, at least one must be false
        from itertools import combinations
        for combo in combinations(variables, k + 1):
            clauses.append([-var for var in combo])
    return clauses

def at_least_k(variables, k):
    \"\"\"Returns clauses ensuring at least k variables are true\"\"\"
    clauses = []
    if k <= 0:
        # Trivially satisfiable
        pass
    elif k > len(variables):
        # Trivially unsatisfiable
        clauses.append([])  # Empty clause means contradiction
    else:
        # For each combination of n-k+1 variables, at least one must be true
        from itertools import combinations
        n = len(variables)
        negated_vars = [-var for var in variables]
        for combo in combinations(negated_vars, n - k + 1):
            clauses.append([-var for var in combo])
    return clauses

def exactly_k(variables, k):
    \"\"\"Returns clauses ensuring exactly k variables are true\"\"\"
    at_most = at_most_k(variables, k)
    at_least = at_least_k(variables, k)
    return at_most + at_least

""" + '\n'.join(regular_lines)
        
        # Setup restricted globals for execution
        restricted_globals = {
            '__builtins__': {
                # Allowed builtins
                'abs': abs,
                'all': all,
                'any': any,
                'bool': bool,
                'dict': dict,
                'divmod': divmod,
                'enumerate': enumerate,
                'filter': filter,
                'float': float,
                'id': id,
                'int': int,
                'isinstance': isinstance,
                'issubclass': issubclass,
                'iter': iter,
                'len': len,
                'list': list,
                'map': map,
                'max': max,
                'min': min,
                'next': next,
                'print': print,
                'range': range,
                'reversed': reversed,
                'round': round,
                'set': set,
                'sorted': sorted,
                'str': str,
                'sum': sum,
                'tuple': tuple,
                'zip': zip,
                'True': True,
                'False': False,
                'None': None,
                'Exception': Exception,
                'ValueError': ValueError,
                'TypeError': TypeError,
                '__import__': safe_import,
            },
            # Provide the PySAT environment - only include stable solvers
            'CNF': CNF,
            'WCNF': WCNF,
            'Glucose3': Glucose3,  # Standard solver 1
            'Cadical153': Cadical153,  # Standard solver 2
            'CardEnc': CardEnc,
            'EncType': EncType,
            'export_solution': export_solution,
            'collections': collections,
            'itertools': itertools,
            'math': math,
            'random': random,
            're': re,
            'json': json,
            # Add cardinality template functions
            'at_most_k': at_most_k,
            'at_least_k': at_least_k,
            'exactly_k': exactly_k,
            # Add constraint helper functions
            'at_most_one': at_most_one,
            'exactly_one': exactly_one,
            'implies': implies,
            'mutually_exclusive': mutually_exclusive,
            'if_then_else': if_then_else,
        }
        
        # Add common variable types needed for PySAT code
        restricted_globals['dict_keys'] = type({}.keys())
        restricted_globals['dict_values'] = type({}.values())
        restricted_globals['dict_items'] = type({}.items())
        restricted_globals['map_iterator'] = type(map(lambda x: x, []))
        restricted_globals['filter_iterator'] = type(filter(lambda x: x, []))
        restricted_globals['enumerate_iterator'] = type(enumerate([]))
        restricted_globals['zip_iterator'] = type(zip())
        restricted_globals['range_iterator'] = type(range(0))
        
        # Add logging capability so we can trace execution
        restricted_globals['logging'] = logging
        
        # Wrap code execution in a try-except block to catch syntax errors
        try:
            # First, compile the code to catch syntax errors before execution
            compiled_code = compile(processed_code, '<pysat_code>', 'exec')
            
            # Execute code with timeout and capture output
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                with time_limit(timeout):
                    # Execute the compiled code
                    exec(compiled_code, restricted_globals)
            
            # Extract standard output and error
            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()
            
            # Check if _LAST_SOLUTION was set by export_solution
            solution = None
            if '_LAST_SOLUTION' in restricted_globals:
                solution = restricted_globals['_LAST_SOLUTION']
                logger.debug(f"_LAST_SOLUTION set to: {solution}")
                
            # Return success result with output and solution
            result['success'] = True
            result['output'] = stdout + '\n' + stderr
            result['solution'] = solution
            
        except SyntaxError as e:
            # Handle syntax errors in the code
            line_num = e.lineno if hasattr(e, 'lineno') else '?'
            col_num = e.offset if hasattr(e, 'offset') else '?'
            error_text = e.text.strip() if hasattr(e, 'text') and e.text else 'unknown'
            
            error_msg = f"Syntax error at line {line_num}, column {col_num}: {str(e)}"
            logger.error(error_msg)
            
            # Add detailed syntax error information
            result['error'] = error_msg
            result['error_details'] = {
                'type': 'SyntaxError',
                'line': line_num,
                'column': col_num,
                'text': error_text,
                'message': str(e)
            }
            result['output'] = f"Syntax error: {error_msg}\n{stderr_capture.getvalue()}"
            
        except NameError as e:
            # Handle undefined variable errors
            var_match = re.search(r"name '(\w+)' is not defined", str(e))
            var_name = var_match.group(1) if var_match else "unknown"
            
            error_msg = f"Undefined variable: '{var_name}'. {str(e)}"
            logger.error(error_msg)
            
            # Provide helpful suggestions for common undefined variables
            suggestions = {}
            if var_name in ['variables', 'variable', 'vars', 'var']:
                suggestions['hint'] = "Did you forget to initialize a variables dictionary? Use 'variables = {}' before adding variables."
            elif var_name in ['cnf', 'formula', 'problem']:
                suggestions['hint'] = "Did you forget to create a CNF object? Use 'cnf = CNF()' before adding clauses."
            elif var_name in ['solver', 'sat_solver']:
                suggestions['hint'] = "Did you forget to create a solver? Use 'solver = Glucose3(cnf)' or another supported solver."
            
            result['error'] = error_msg
            result['error_details'] = {
                'type': 'NameError',
                'variable': var_name,
                'message': str(e),
                'suggestions': suggestions if suggestions else None
            }
            result['output'] = f"NameError: {error_msg}\n{stdout_capture.getvalue()}\n{stderr_capture.getvalue()}"
            
        except TypeError as e:
            # Handle type errors
            error_msg = f"Type error: {str(e)}"
            logger.error(error_msg)
            
            # Extract function name if it's a function call error
            func_match = re.search(r"(\w+)\(\) takes", str(e))
            func_name = func_match.group(1) if func_match else None
            
            # Add call context for method/function type errors
            result['error'] = error_msg
            result['error_details'] = {
                'type': 'TypeError',
                'function': func_name,
                'message': str(e)
            }
            
            # Add helpful suggestions for common type errors
            if "takes 1 positional argument but 2 were given" in str(e):
                result['error_details']['hint'] = "You may be calling a method without using the dot notation (object.method instead of method(object))."
            elif "object is not callable" in str(e):
                result['error_details']['hint'] = "You may be trying to call something that is not a function."
                
            result['output'] = f"TypeError: {error_msg}\n{stdout_capture.getvalue()}\n{stderr_capture.getvalue()}"
            
        except AttributeError as e:
            # Handle attribute errors
            error_msg = f"Attribute error: {str(e)}"
            logger.error(error_msg)
            
            # Extract object and attribute names
            attr_match = re.search(r"'(\w+)' object has no attribute '(\w+)'", str(e))
            if attr_match:
                obj_type, attr_name = attr_match.groups()
                
                # Add detailed attribute error information
                result['error_details'] = {
                    'type': 'AttributeError',
                    'object_type': obj_type,
                    'attribute': attr_name,
                    'message': str(e)
                }
                
                # Common attribute error suggestions
                if obj_type == 'NoneType':
                    result['error_details']['hint'] = "You're trying to access an attribute on a None value. Check if your variable was properly initialized."
                elif attr_name == 'solve' and obj_type in ['dict', 'list', 'int', 'str']:
                    result['error_details']['hint'] = f"The '{obj_type}' object is not a solver. Did you forget to create a solver with 'solver = Glucose3(cnf)' or similar?"
                elif attr_name == 'add_clause' and obj_type in ['dict', 'list', 'int', 'str']:
                    result['error_details']['hint'] = f"The '{obj_type}' object is not a CNF formula. Did you forget to create a CNF with 'cnf = CNF()' or similar?"
            
            result['error'] = error_msg
            result['output'] = f"AttributeError: {error_msg}\n{stdout_capture.getvalue()}\n{stderr_capture.getvalue()}"
            
        except KeyError as e:
            # Handle key errors
            key = str(e).strip("'")
            error_msg = f"Key error: Key '{key}' not found"
            logger.error(error_msg)
            
            result['error'] = error_msg
            result['error_details'] = {
                'type': 'KeyError',
                'key': key,
                'message': str(e)
            }
            result['output'] = f"KeyError: {error_msg}\n{stdout_capture.getvalue()}\n{stderr_capture.getvalue()}"
            
        except IndexError as e:
            # Handle index errors
            error_msg = f"Index error: {str(e)}"
            logger.error(error_msg)
            
            result['error'] = error_msg
            result['error_details'] = {
                'type': 'IndexError',
                'message': str(e)
            }
            result['output'] = f"IndexError: {error_msg}\n{stdout_capture.getvalue()}\n{stderr_capture.getvalue()}"
            
        except TimeoutException:
            # Handle timeout errors
            error_msg = f"Execution timed out after {timeout} seconds"
            logger.error(error_msg)
            
            result['error'] = error_msg
            result['error_details'] = {
                'type': 'TimeoutError',
                'timeout': timeout,
                'message': error_msg
            }
            result['output'] = f"Timeout Error: {error_msg}\n{stdout_capture.getvalue()}\n{stderr_capture.getvalue()}"
            
        except Exception as e:
            # Handle all other exceptions
            error_msg = f"Execution error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Get traceback information
            tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            
            result['error'] = error_msg
            result['error_details'] = {
                'type': type(e).__name__,
                'message': str(e),
                'traceback': tb_str
            }
            result['output'] = f"Error: {error_msg}\n{tb_str}\n{stdout_capture.getvalue()}\n{stderr_capture.getvalue()}"
            
    except Exception as e:
        # Handle unexpected errors in the environment itself
        error_msg = f"Environment error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        result['error'] = error_msg
        result['output'] = f"Environment Error: {error_msg}\n{stdout_capture.getvalue()}\n{stderr_capture.getvalue()}"
        
    finally:
        # Ensure output is captured even in case of error
        if not result['output']:
            result['output'] = stdout_capture.getvalue() + '\n' + stderr_capture.getvalue()
    
    return result 