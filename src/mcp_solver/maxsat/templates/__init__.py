"""
MaxSAT templates package.

This package provides template functions for common MaxSAT patterns to make it
easier to build and solve MaxSAT problems.
"""

from .basic_templates import *
from .optimization_templates import *
from .variable_mapping import *

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
    "create_variable_map"
]