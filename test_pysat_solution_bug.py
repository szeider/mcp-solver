#!/usr/bin/env python3
"""
Test file to reproduce PySAT solution bug.

This test demonstrates the bug where custom solution dictionaries
aren't properly accessible via get_variable_value.
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

async def test_pysat_solution_bug():
    """Test that demonstrates the bug with solution extraction."""
    print("Testing PySAT solution extraction bug...\n")
    
    # Create a PySAT model manager
    manager = PySATModelManager(lite_mode=True)
    
    # Clear any existing model
    await manager.clear_model()
    
    # Add a simple SAT problem with custom solution structure
    test_code = """
from pysat.formula import CNF
from pysat.solvers import Glucose3

# Create a CNF formula
formula = CNF()

# Define variables:
# 1: Actress Alvarez (A)
# 2: Actor Branislavsky (B)
# 3: Actor Cohen (C)
# 4: Actor Davenport (D)

# Some constraints
formula.append([1, 3])   # Alvarez OR Cohen
formula.append([-1, -3]) # NOT(Alvarez AND Cohen)
formula.append([2])      # Branislavsky must be cast

# Create solver and solve
solver = Glucose3()
solver.append_formula(formula)

# Solve and export results with custom structure
if solver.solve():
    model = solver.get_model()
    print(f"DEBUG - Model: {model}")
    
    # Map variables to actor names
    variables = {
        "Alvarez": 1,
        "Branislavsky": 2,
        "Cohen": 3,
        "Davenport": 4
    }
    print(f"DEBUG - Variables: {variables}")
    
    # Create custom result structure
    result = {
        "satisfiable": True,
        "casting": {}  # Custom key, not "values" or "assignment"
    }
    
    # Populate with assignment
    for actor_name, var_id in variables.items():
        is_cast = var_id in model
        result["casting"][actor_name] = is_cast
        print(f"DEBUG - Actor {actor_name} (ID: {var_id}) is cast: {is_cast}")
    
    print(f"DEBUG - Final result structure: {result}")
    
    # Export solution
    export_solution(result)
else:
    export_solution({"satisfiable": False})

# Free solver memory
solver.delete()
"""
    
    # Add the code to the model manager
    await manager.add_item(1, test_code)
    
    # Solve the model
    print("Solving model...")
    solve_result = await manager.solve_model(timeout=timedelta(seconds=5))
    print(f"Solve result: {json.dumps(solve_result, indent=2)}")
    
    # Get the solution - using manager methods without await
    print("\nGetting solution...")
    solution_result = manager.get_solution()
    print(f"Solution structure: {json.dumps(solution_result, indent=2)}")
    
    # Print raw solution data for debugging
    if "solution" in solution_result:
        print("\nRaw solution data:")
        print(json.dumps(solution_result["solution"], indent=2))
    
    # Try to get variable values
    print("\nAttempting to get variable values...")
    for actor in ["Alvarez", "Branislavsky", "Cohen", "Davenport"]:
        result = manager.get_variable_value(actor)
        print(f"Value for {actor}: {result}")
    
    # Try to get the custom 'casting' dictionary
    print("\nAttempting to get the custom 'casting' dictionary...")
    result = manager.get_variable_value("casting")
    print(f"Value for 'casting': {result}")
    
    print("\nTest completed.")

if __name__ == "__main__":
    # Run the async function
    asyncio.run(test_pysat_solution_bug()) 