#!/usr/bin/env python3
"""
Simple test for the global MaxSAT functionality.

This example validates that the global MaxSAT functionality works as expected.
"""

import sys
import os
import json

# Add the src directory to the path if running from the mcp-solver directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mcp_solver.pysat.environment import execute_pysat_code

def main():
    """Test the global MaxSAT functionality."""
    # Test case using the global MaxSAT API
    code = """
# Initialize a new MaxSAT problem
initialize_maxsat()

# Define variables:
# 1: Course 1 on Monday
# 2: Course 1 on Tuesday
# 3: Course 1 on Wednesday
# 4: Course 2 on Monday
# 5: Course 2 on Tuesday
# 6: Course 2 on Wednesday

# Hard constraints
# Each course must be scheduled exactly once
add_hard_clause([1, 2, 3])  # Course 1 on at least one day
add_hard_clause([4, 5, 6])  # Course 2 on at least one day

add_hard_clause([-1, -2])   # Course 1 not on both Monday and Tuesday
add_hard_clause([-1, -3])   # Course 1 not on both Monday and Wednesday
add_hard_clause([-2, -3])   # Course 1 not on both Tuesday and Wednesday

add_hard_clause([-4, -5])   # Course 2 not on both Monday and Tuesday
add_hard_clause([-4, -6])   # Course 2 not on both Monday and Wednesday
add_hard_clause([-5, -6])   # Course 2 not on both Tuesday and Wednesday

# Soft constraints
add_soft_clause([-1], weight=2)  # Prefer Course 1 not on Monday (weight 2)
add_soft_clause([6], weight=3)   # Prefer Course 2 on Wednesday (weight 3)

# Solve the problem
model, cost = solve_maxsat(timeout=5.0)

# Define variable meanings for clarity
variables = {
    "course1_monday": 1,
    "course1_tuesday": 2,
    "course1_wednesday": 3,
    "course2_monday": 4,
    "course2_tuesday": 5,
    "course2_wednesday": 6
}

# Create solution dictionary
if model is not None:
    solution = {var_name: var_id in model for var_name, var_id in variables.items()}
    print(f"Solution found: {solution}")
    print(f"Objective (penalty) value: {cost}")
    
    # Export the final result
    result = {
        "satisfiable": True, 
        "solution": solution,
        "objective": cost,
        "model": model
    }
    export_solution(result)
else:
    print("No solution found")
    export_solution({"satisfiable": False})
"""
    result = execute_pysat_code(code)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main() 