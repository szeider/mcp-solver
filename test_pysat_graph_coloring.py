#!/usr/bin/env python3
"""
Test file to verify the PySAT solution extraction with a graph coloring problem.

This tests that our fix works with a completely different problem structure than
the casting problem we used in the original test.
"""

import os
import sys
import asyncio
import json
from datetime import timedelta
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from mcp_solver.pysat.model_manager import PySATModelManager

async def test_graph_coloring():
    """Test with a graph coloring problem to verify our solution extraction."""
    print("Testing PySAT solution extraction with graph coloring problem...\n")
    
    # Create a PySAT model manager
    manager = PySATModelManager(lite_mode=True)
    
    # Clear any existing model
    await manager.clear_model()
    
    # Add a graph coloring problem
    test_code = """
from pysat.formula import CNF
from pysat.solvers import Glucose3

# Create a CNF formula for graph coloring
formula = CNF()

# Graph description:
# We have 4 nodes (A, B, C, D) with the following edges:
# A -- B, A -- C, B -- C, B -- D, C -- D
# We want to color the graph with 3 colors so that no adjacent nodes have the same color

# Variable encoding:
# For each node N and color C, we have a variable N_C which is true if node N has color C
# A_1: Node A has color 1, variable 1
# A_2: Node A has color 2, variable 2
# A_3: Node A has color 3, variable 3
# B_1: Node B has color 1, variable 4
# B_2: Node B has color 2, variable 5
# B_3: Node B has color 3, variable 6
# C_1: Node C has color 1, variable 7
# C_2: Node C has color 2, variable 8
# C_3: Node C has color 3, variable 9
# D_1: Node D has color 1, variable 10
# D_2: Node D has color 2, variable 11
# D_3: Node D has color 3, variable 12

# For readability, create variable mapping
variables = {
    "A_1": 1, "A_2": 2, "A_3": 3,
    "B_1": 4, "B_2": 5, "B_3": 6,
    "C_1": 7, "C_2": 8, "C_3": 9,
    "D_1": 10, "D_2": 11, "D_3": 12
}

# Each node must have exactly one color
# Node A has exactly one color
formula.append([variables["A_1"], variables["A_2"], variables["A_3"]])  # At least one color
formula.append([-variables["A_1"], -variables["A_2"]])  # Not both colors 1 and 2
formula.append([-variables["A_1"], -variables["A_3"]])  # Not both colors 1 and 3
formula.append([-variables["A_2"], -variables["A_3"]])  # Not both colors 2 and 3

# Node B has exactly one color
formula.append([variables["B_1"], variables["B_2"], variables["B_3"]])
formula.append([-variables["B_1"], -variables["B_2"]])
formula.append([-variables["B_1"], -variables["B_3"]])
formula.append([-variables["B_2"], -variables["B_3"]])

# Node C has exactly one color
formula.append([variables["C_1"], variables["C_2"], variables["C_3"]])
formula.append([-variables["C_1"], -variables["C_2"]])
formula.append([-variables["C_1"], -variables["C_3"]])
formula.append([-variables["C_2"], -variables["C_3"]])

# Node D has exactly one color
formula.append([variables["D_1"], variables["D_2"], variables["D_3"]])
formula.append([-variables["D_1"], -variables["D_2"]])
formula.append([-variables["D_1"], -variables["D_3"]])
formula.append([-variables["D_2"], -variables["D_3"]])

# Edge constraints: adjacent nodes cannot have the same color
# Edge A -- B
formula.append([-variables["A_1"], -variables["B_1"]])  # A and B cannot both have color 1
formula.append([-variables["A_2"], -variables["B_2"]])  # A and B cannot both have color 2
formula.append([-variables["A_3"], -variables["B_3"]])  # A and B cannot both have color 3

# Edge A -- C
formula.append([-variables["A_1"], -variables["C_1"]])
formula.append([-variables["A_2"], -variables["C_2"]])
formula.append([-variables["A_3"], -variables["C_3"]])

# Edge B -- C
formula.append([-variables["B_1"], -variables["C_1"]])
formula.append([-variables["B_2"], -variables["C_2"]])
formula.append([-variables["B_3"], -variables["C_3"]])

# Edge B -- D
formula.append([-variables["B_1"], -variables["D_1"]])
formula.append([-variables["B_2"], -variables["D_2"]])
formula.append([-variables["B_3"], -variables["D_3"]])

# Edge C -- D
formula.append([-variables["C_1"], -variables["D_1"]])
formula.append([-variables["C_2"], -variables["D_2"]])
formula.append([-variables["C_3"], -variables["D_3"]])

# Create solver and solve
solver = Glucose3()
solver.append_formula(formula)

# Solve and export results
if solver.solve():
    model = solver.get_model()
    
    # Extract the coloring from the model
    coloring = {}
    for var_name, var_id in variables.items():
        node = var_name.split('_')[0]  # Extract node name (A, B, C, D)
        color = var_name.split('_')[1]  # Extract color (1, 2, 3)
        
        if var_id in model:  # If variable is true
            if node not in coloring:
                coloring[node] = color
    
    # Create custom result structure
    result = {
        "satisfiable": True,
        "coloring": coloring,           # Custom dictionary mapping nodes to colors
        "node_colors": coloring.copy(), # Duplicate for testing multiple custom dictionaries
        "model": model,                  # Original model from the solver
        "values": {}                     # Initialize the values dictionary
    }
    
    # Also store node colors in the values dictionary for individual access
    for node, color in coloring.items():
        result["values"][node] = color
    
    print(f"Found valid coloring: {coloring}")
    
    # Export solution
    export_solution(result)
else:
    export_solution({"satisfiable": False, "message": "No valid coloring exists"})

# Free solver memory
solver.delete()
"""
    
    # Add the code to the model manager
    await manager.add_item(1, test_code)
    
    # Solve the model
    print("Solving model...")
    solve_result = await manager.solve_model(timeout=timedelta(seconds=5))
    print(f"Solve result: {json.dumps(solve_result, indent=2)}")
    
    # Get the solution
    print("\nGetting solution...")
    solution_result = manager.get_solution()
    print(f"Solution structure: {json.dumps(solution_result, indent=2)}")
    
    # Try to get individual node colors
    print("\nAttempting to get node colors as individual variables...")
    for node in ["A", "B", "C", "D"]:
        result = manager.get_variable_value(node)
        print(f"Value for {node}: {result}")
    
    # Try to get custom dictionaries
    print("\nAttempting to get custom dictionaries...")
    for dict_name in ["coloring", "node_colors"]:
        result = manager.get_variable_value(dict_name)
        print(f"Value for '{dict_name}': {result}")
    
    print("\nTest completed.")

if __name__ == "__main__":
    # Run the async function
    asyncio.run(test_graph_coloring()) 