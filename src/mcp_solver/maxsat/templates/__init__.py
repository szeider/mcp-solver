"""
MaxSAT templates package.

This package provides template functions for common MaxSAT patterns to make it
easier to build and solve MaxSAT problems.
"""

from .basic_templates import *
from .optimization_templates import *
from .variable_mapping import *
from .cardinality_constraints import *
from .objective_helpers import *

__all__ = [
    # From basic_templates
    "create_maxsat_solver",
    "solve_maxsat_problem",
    "add_hard_constraint",
    "add_soft_constraint",
    "encode_binary_variable",
    
    # From optimization_templates
    "feature_selection_problem",
    "weighted_max_cut",
    "set_cover_problem",
    "knapsack_problem",
    
    # From variable_mapping
    "VariableMap",
    "create_variable_map",
    
    # From cardinality_constraints
    "at_most_k",
    "at_least_k",
    "prefer_at_least_k",
    "prefer_at_most_k",
    
    # From objective_helpers
    "maximize_sum",
    "minimize_sum",
    "optimize_net_value",
    "calculate_objective_value",
    "encode_weighted_selection"
]