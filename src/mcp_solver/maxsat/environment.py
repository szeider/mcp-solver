# Import and re-export the environment module from PySAT
# We need to be careful to only import what we need and not confuse users
import sys
import os
import tempfile
import logging
import math
import random
import collections
import itertools
import re
import json
import traceback
from typing import Any

# Import specific PySAT environment utilities but not solution functions
from mcp_solver.pysat.environment import execute_pysat_code

# Define the default timeout
DEFAULT_TIMEOUT_SECONDS = 10.0

# Import our solution functions instead
from mcp_solver.maxsat.solution import export_maxsat_solution

# Add a reference that will be available in the environment
def get_export_maxsat_solution():
    """
    Helper function to get a reference to the export_maxsat_solution function.
    This is needed because the function might not be directly importable in the environment.
    """
    return export_maxsat_solution

# Import MaxSAT templates
from mcp_solver.maxsat.templates import (
    # From basic_templates
    create_maxsat_solver,
    solve_maxsat_problem,
    add_hard_constraint,
    add_soft_constraint,
    encode_binary_variable,
    
    # From optimization_templates
    feature_selection_problem,
    weighted_max_cut,
    set_cover_problem,
    knapsack_problem,
    
    # From variable_mapping
    VariableMap,
    create_variable_map
)

# Hide pysat's export_solution to prevent confusion
__all__ = [
    # Environment functions
    "execute_pysat_code",
    "DEFAULT_TIMEOUT_SECONDS",
    
    # Solution functions
    "export_maxsat_solution",
    
    # Basic templates
    "create_maxsat_solver",
    "solve_maxsat_problem",
    "add_hard_constraint",
    "add_soft_constraint",
    "encode_binary_variable",
    
    # Optimization templates
    "feature_selection_problem",
    "weighted_max_cut",
    "set_cover_problem",
    "knapsack_problem",
    
    # Variable mapping
    "VariableMap",
    "create_variable_map"
]
