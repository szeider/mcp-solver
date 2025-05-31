"""
Optimization templates for MaxSAT.

This module provides template functions for common MaxSAT optimization problems,
making it easier to solve complex optimization tasks.
"""

import sys
from typing import Any, Dict, List, Tuple, Union, Optional

# Import PySAT but protect against failure
try:
    from pysat.formula import CNF, WCNF
    from pysat.examples.rc2 import RC2
except ImportError:
    print("PySAT solver not found. Install with: pip install python-sat")
    sys.exit(1)

# Import from our solution module
from mcp_solver.maxsat.solution import export_maxsat_solution
from .variable_mapping import VariableMap


def feature_selection_problem(
    features: Dict[str, int],
    dependencies: List[Tuple[str, str]] = None,
    exclusions: List[Tuple[str, str]] = None,
    max_features: Optional[int] = None,
    required_features: List[str] = None
) -> Dict[str, Any]:
    """
    Create and solve a feature selection optimization problem.
    
    This function creates a MaxSAT problem that selects features to maximize
    the total value while respecting dependencies and constraints.
    
    Args:
        features: Dictionary mapping feature names to their values (weights)
        dependencies: List of (feature, dependency) pairs where feature requires dependency
        exclusions: List of (feature1, feature2) pairs that can't both be selected
        max_features: Maximum number of features to select (optional)
        required_features: List of features that must be selected (optional)
        
    Returns:
        Dictionary with the MaxSAT solution
    """
    # Create WCNF formula and variable mapping
    wcnf = WCNF()
    var_map = VariableMap()
    
    # Create variables for each feature
    feature_vars = {f: var_map.create_var(f) for f in features}
    
    # Add soft constraints for feature values
    for feature, value in features.items():
        if value > 0:  # Only add positive values as soft constraints
            wcnf.append([feature_vars[feature]], weight=value)
    
    # Add hard constraints for dependencies
    if dependencies:
        for feature, dependency in dependencies:
            if feature in feature_vars and dependency in feature_vars:
                # feature → dependency (¬feature ∨ dependency)
                wcnf.append([-feature_vars[feature], feature_vars[dependency]])
    
    # Add hard constraints for mutual exclusions
    if exclusions:
        for f1, f2 in exclusions:
            if f1 in feature_vars and f2 in feature_vars:
                # ¬(f1 ∧ f2) ≡ (¬f1 ∨ ¬f2)
                wcnf.append([-feature_vars[f1], -feature_vars[f2]])
    
    # Add hard constraint for maximum number of features
    if max_features is not None and max_features < len(features):
        # We need to add clauses that ensure at most max_features are true
        # For small problems, we can use the direct encoding
        import itertools
        all_feature_vars = list(feature_vars.values())
        
        # For each combination of max_features+1 variables, add a clause
        # stating that they can't all be true
        for combo in itertools.combinations(all_feature_vars, max_features + 1):
            wcnf.append([-var for var in combo])
    
    # Add hard constraints for required features
    if required_features:
        for feature in required_features:
            if feature in feature_vars:
                wcnf.append([feature_vars[feature]])
    
    # Solve with RC2
    with RC2(wcnf) as solver:
        model = solver.compute()
        
        if model is not None:
            # Extract selected features
            selected = {f: (feature_vars[f] in model) for f in features}
            
            # Calculate total value
            total_value = sum(value for f, value in features.items() 
                              if feature_vars[f] in model)
            
            # Export the solution with additional information
            return export_maxsat_solution({
                "satisfiable": True,
                "status": "optimal",
                "selected_features": selected,
                "total_value": total_value,
                "cost": solver.cost,
                "num_selected": sum(1 for f in selected.values() if f)
            }, var_map.get_mapping())
        else:
            return export_maxsat_solution({
                "satisfiable": False,
                "status": "unsatisfiable",
                "message": "No solution exists that satisfies all constraints"
            })


def weighted_max_cut(
    edges: List[Tuple[int, int, int]]
) -> Dict[str, Any]:
    """
    Create and solve a weighted maximum cut problem.
    
    This function creates a MaxSAT problem that finds a cut (partition of nodes)
    that maximizes the total weight of edges crossing the cut.
    
    Args:
        edges: List of (node1, node2, weight) tuples representing the graph
        
    Returns:
        Dictionary with the MaxSAT solution
    """
    # Create WCNF formula
    wcnf = WCNF()
    
    # Determine the set of nodes
    nodes = set()
    for u, v, _ in edges:
        nodes.add(u)
        nodes.add(v)
    
    # Create variable mapping: node ID to variable ID
    var_map = VariableMap()
    node_vars = {node: var_map.create_var(f"node{node}") for node in nodes}
    
    # For each edge, we want to maximize the weight of edges where
    # the two nodes are on opposite sides of the cut
    for u, v, weight in edges:
        # The clause (u XOR v) represents nodes on opposite sides
        # For MaxSAT, we use two soft clauses with the same weight:
        
        # Clause for nodes on opposite sides: NOT u OR NOT v
        wcnf.append([-node_vars[u], -node_vars[v]], weight=weight)
        
        # Clause for nodes on opposite sides: u OR v
        wcnf.append([node_vars[u], node_vars[v]], weight=weight)
    
    # Solve with RC2
    with RC2(wcnf) as solver:
        model = solver.compute()
        
        if model is not None:
            # Extract the cut (which nodes are on which side)
            side_a = [node for node in nodes if node_vars[node] in model]
            side_b = [node for node in nodes if node_vars[node] not in model]
            
            # Calculate the cut weight
            cut_weight = sum(
                weight for u, v, weight in edges
                if (node_vars[u] in model and node_vars[v] not in model) or
                   (node_vars[u] not in model and node_vars[v] in model)
            )
            
            # Extract cut edges
            cut_edges = [
                (u, v) for u, v, _ in edges
                if (node_vars[u] in model and node_vars[v] not in model) or
                   (node_vars[u] not in model and node_vars[v] in model)
            ]
            
            # Export the solution
            return export_maxsat_solution({
                "satisfiable": True,
                "status": "optimal",
                "side_a": side_a,
                "side_b": side_b,
                "cut_weight": cut_weight,
                "cut_edges": cut_edges,
                "cost": solver.cost
            }, var_map.get_mapping())
        else:
            return export_maxsat_solution({
                "satisfiable": False,
                "message": "No solution exists"
            })


def set_cover_problem(
    universe: List[Any],
    sets: Dict[str, List[Any]],
    costs: Dict[str, int] = None
) -> Dict[str, Any]:
    """
    Create and solve a weighted set cover problem.
    
    This function creates a MaxSAT problem that finds a minimum-cost collection
    of sets that covers all elements in the universe.
    
    Args:
        universe: List of elements to be covered
        sets: Dictionary mapping set names to the elements they contain
        costs: Dictionary mapping set names to their costs (defaults to 1)
        
    Returns:
        Dictionary with the MaxSAT solution
    """
    # Create WCNF formula
    wcnf = WCNF()
    var_map = VariableMap()
    
    # Create variables for each set
    set_vars = {s: var_map.create_var(s) for s in sets}
    
    # Default costs
    if costs is None:
        costs = {s: 1 for s in sets}
    
    # Add soft constraints for set costs (we want to minimize the total cost)
    for s, cost in costs.items():
        if s in set_vars:
            # Negative selection: we prefer sets NOT to be selected
            wcnf.append([-set_vars[s]], weight=cost)
    
    # Add hard constraints to ensure every element is covered
    for element in universe:
        # Find all sets that contain this element
        covering_sets = [s for s, elements in sets.items() 
                        if element in elements and s in set_vars]
        
        if covering_sets:
            # At least one of the covering sets must be selected
            wcnf.append([set_vars[s] for s in covering_sets])
    
    # Solve with RC2
    with RC2(wcnf) as solver:
        model = solver.compute()
        
        if model is not None:
            # Extract selected sets
            selected_sets = [s for s in sets if set_vars[s] in model]
            
            # Calculate total cost
            total_cost = sum(costs[s] for s in selected_sets)
            
            # Verify coverage
            covered = set()
            for s in selected_sets:
                covered.update(sets[s])
            
            all_covered = set(universe).issubset(covered)
            
            # Export the solution
            return export_maxsat_solution({
                "satisfiable": True,
                "status": "optimal",
                "selected_sets": selected_sets,
                "total_cost": total_cost,
                "cost": solver.cost,
                "all_covered": all_covered,
                "coverage_percentage": len(covered) / len(universe) * 100
            }, var_map.get_mapping())
        else:
            return export_maxsat_solution({
                "satisfiable": False,
                "message": "No solution exists that covers all elements"
            })


def knapsack_problem(
    items: Dict[str, Dict[str, Union[int, float]]],
    capacity: int,
    dependencies: List[Tuple[str, str]] = None
) -> Dict[str, Any]:
    """
    Create and solve a knapsack problem with dependencies.
    
    This function creates a MaxSAT problem that selects items to maximize
    the total value while respecting a capacity constraint and dependencies.
    
    Args:
        items: Dictionary mapping item names to their properties (weight, value)
        capacity: Maximum total weight capacity
        dependencies: List of (item, dependency) pairs where item requires dependency
        
    Returns:
        Dictionary with the MaxSAT solution
    """
    # Create WCNF formula
    wcnf = WCNF()
    var_map = VariableMap()
    
    # Create variables for each item
    item_vars = {item: var_map.create_var(item) for item in items}
    
    # Add soft constraints for item values
    for item, props in items.items():
        value = props.get("value", 0)
        if value > 0:
            wcnf.append([item_vars[item]], weight=value)
    
    # Add hard constraints for dependencies
    if dependencies:
        for item, dependency in dependencies:
            if item in item_vars and dependency in item_vars:
                # item → dependency (¬item ∨ dependency)
                wcnf.append([-item_vars[item], item_vars[dependency]])
    
    # Add hard constraint for capacity
    # This is complex for MaxSAT since we need to encode a linear constraint
    # We'll use a simplified approach that works for small problems
    
    # For each subset of items that exceeds capacity, add a constraint
    # that they can't all be selected
    import itertools
    for r in range(1, len(items) + 1):
        for combo in itertools.combinations(items.keys(), r):
            total_weight = sum(items[item].get("weight", 0) for item in combo)
            if total_weight > capacity:
                # These items can't all be selected
                wcnf.append([-item_vars[item] for item in combo])
    
    # Solve with RC2
    with RC2(wcnf) as solver:
        model = solver.compute()
        
        if model is not None:
            # Extract selected items
            selected = {item: (item_vars[item] in model) for item in items}
            selected_items = [item for item in items if selected[item]]
            
            # Calculate total value and weight
            total_value = sum(items[item].get("value", 0) 
                             for item in selected_items)
            total_weight = sum(items[item].get("weight", 0) 
                              for item in selected_items)
            
            # Export the solution
            return export_maxsat_solution({
                "satisfiable": True,
                "status": "optimal",
                "selected_items": selected,
                "total_value": total_value,
                "total_weight": total_weight,
                "remaining_capacity": capacity - total_weight,
                "cost": solver.cost
            }, var_map.get_mapping())
        else:
            return export_maxsat_solution({
                "satisfiable": False,
                "message": "No solution exists that satisfies all constraints"
            })