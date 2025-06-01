"""
Objective helper templates for MaxSAT.

This module provides helper functions for encoding common optimization
objectives in MaxSAT formulas.
"""

from typing import List, Tuple, Dict, Any

from pysat.formula import WCNF


def maximize_sum(wcnf: WCNF, var_value_pairs: List[Tuple[int, int]]) -> None:
    """
    Add soft constraints to maximize the sum of values for selected variables.
    
    Since MaxSAT minimizes the sum of unsatisfied soft clause weights,
    we add a soft clause [var] with weight=value for each variable.
    This way, NOT selecting the variable (var=FALSE) costs its value.
    
    Args:
        wcnf: WCNF formula to modify
        var_value_pairs: List of (variable_id, value) tuples
    """
    for var, value in var_value_pairs:
        if value > 0:  # Only add positive values
            wcnf.append([var], weight=value)


def minimize_sum(wcnf: WCNF, var_cost_pairs: List[Tuple[int, int]]) -> None:
    """
    Add soft constraints to minimize the sum of costs for selected variables.
    
    We add a soft clause [-var] with weight=cost for each variable.
    This way, selecting the variable (var=TRUE) costs its value.
    
    Args:
        wcnf: WCNF formula to modify
        var_cost_pairs: List of (variable_id, cost) tuples
    """
    for var, cost in var_cost_pairs:
        if cost > 0:  # Only add positive costs
            wcnf.append([-var], weight=cost)


def optimize_net_value(wcnf: WCNF, var_info: List[Tuple[int, int, int]]) -> None:
    """
    Add soft constraints for variables with both benefits and costs.
    
    For each variable, we consider net_value = benefit - cost:
    - If net_value > 0: We want to select it (add soft clause [var] with weight=net_value)
    - If net_value < 0: We want to avoid it (add soft clause [-var] with weight=|net_value|)
    
    Args:
        wcnf: WCNF formula to modify
        var_info: List of (variable_id, benefit, cost) tuples
    """
    for var, benefit, cost in var_info:
        net_value = benefit - cost
        if net_value > 0:
            # Positive net value: prefer to select
            wcnf.append([var], weight=net_value)
        elif net_value < 0:
            # Negative net value: prefer not to select
            wcnf.append([-var], weight=-net_value)
        # If net_value == 0, no preference needed


def calculate_objective_value(
    model: List[int], 
    var_value_pairs: List[Tuple[int, int]], 
    maximize: bool = True
) -> int:
    """
    Calculate the objective value achieved by a model.
    
    This is useful for reporting the actual objective value achieved,
    as opposed to the solver's cost which represents unsatisfied weights.
    
    Args:
        model: List of true variable IDs from the solver
        var_value_pairs: List of (variable_id, value) tuples
        maximize: Whether this is a maximization (True) or minimization (False) problem
    
    Returns:
        The objective value achieved
    """
    total = 0
    for var, value in var_value_pairs:
        if var in model:
            total += value
    
    return total if maximize else -total


def encode_weighted_selection(
    wcnf: WCNF,
    items: Dict[str, Dict[str, Any]],
    var_mapping: Dict[str, int],
    value_key: str = "value",
    cost_key: str = "cost"
) -> None:
    """
    Encode a weighted selection problem where items have values and/or costs.
    
    This helper handles common patterns in optimization problems where
    you need to select items to maximize value or minimize cost.
    
    Args:
        wcnf: WCNF formula to modify
        items: Dictionary mapping item names to their properties
        var_mapping: Dictionary mapping item names to variable IDs
        value_key: Key in item properties for the value (benefit)
        cost_key: Key in item properties for the cost
    """
    for item_name, properties in items.items():
        if item_name not in var_mapping:
            continue
            
        var = var_mapping[item_name]
        value = properties.get(value_key, 0)
        cost = properties.get(cost_key, 0)
        
        if value > 0 and cost > 0:
            # Both value and cost: use net value
            net = value - cost
            if net > 0:
                wcnf.append([var], weight=net)
            elif net < 0:
                wcnf.append([-var], weight=-net)
        elif value > 0:
            # Only value: maximize
            wcnf.append([var], weight=value)
        elif cost > 0:
            # Only cost: minimize
            wcnf.append([-var], weight=cost)