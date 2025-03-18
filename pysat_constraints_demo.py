#!/usr/bin/env python3
"""
Demo script for PySAT constraint functions.
Tests all available constraint helper functions in the MCP Solver.
"""

from src.mcp_solver.pysat.environment import execute_pysat_code

# Define test code that uses all constraint functions
TEST_CODE = """
from pysat.formula import CNF
from pysat.solvers import Glucose3

# Create a CNF formula
formula = CNF()

# Create 6 variables for our demo
variables = {f"v{i}": i for i in range(1, 7)}

# Demonstrate at_most_one - at most one of v1, v2, v3 can be true
print("\\nTesting at_most_one constraint:")
for clause in at_most_one([variables["v1"], variables["v2"], variables["v3"]]):
    print(f"  Adding clause: {clause}")
    formula.append(clause)

# Demonstrate exactly_one - exactly one of v1, v2, v3 must be true
print("\\nTesting exactly_one constraint:")
for clause in exactly_one([variables["v4"], variables["v5"], variables["v6"]]):
    print(f"  Adding clause: {clause}")
    formula.append(clause)

# Demonstrate implies - if v1 is true, then v4 must be true
print("\\nTesting implies constraint:")
for clause in implies(variables["v1"], variables["v4"]):
    print(f"  Adding clause: {clause}")
    formula.append(clause)

# Demonstrate mutually_exclusive - same as at_most_one but semantic difference
print("\\nTesting mutually_exclusive constraint:")
for clause in mutually_exclusive([variables["v2"], variables["v5"]]):
    print(f"  Adding clause: {clause}")
    formula.append(clause)

# Demonstrate if_then_else - if v3 is true use v1, otherwise use v2
print("\\nTesting if_then_else constraint:")
for clause in if_then_else(variables["v3"], variables["v1"], variables["v2"]):
    print(f"  Adding clause: {clause}")
    formula.append(clause)

# Demonstrate at_most_k - at most 2 of v1, v2, v3, v4 can be true
print("\\nTesting at_most_k constraint:")
for clause in at_most_k([variables["v1"], variables["v2"], variables["v3"], variables["v4"]], 2):
    print(f"  Adding clause: {clause}")
    formula.append(clause)

# Demonstrate at_least_k - at least 1 of v1, v2, v3 must be true
print("\\nTesting at_least_k constraint:")
for clause in at_least_k([variables["v1"], variables["v2"], variables["v3"]], 1):
    print(f"  Adding clause: {clause}")
    formula.append(clause)

# Demonstrate exactly_k - exactly 2 of v1, v2, v3, v4, v5 must be true
print("\\nTesting exactly_k constraint:")
for clause in exactly_k([variables["v1"], variables["v2"], variables["v3"], variables["v4"], variables["v5"]], 2):
    print(f"  Adding clause: {clause}")
    formula.append(clause)

# Create solver and add the formula
solver = Glucose3()
solver.append_formula(formula)

# Solve the formula
print("\\nSolving formula...")
if solver.solve():
    model = solver.get_model()
    print(f"SAT with model: {model}")
    
    # Extract variable assignments
    assignments = {}
    for var_name, var_id in variables.items():
        assignments[var_name] = var_id in model
    
    # Export solution
    export_solution({
        "satisfiable": True,
        "model": model,
        "assignment": assignments
    })
    
    # Print assignments for readability
    print("\\nVariable assignments:")
    for var_name, is_true in assignments.items():
        print(f"  {var_name}: {is_true}")
else:
    print("UNSAT - no solution exists")
    export_solution({
        "satisfiable": False
    })

# Free solver memory
solver.delete()
"""

def main():
    # Execute the test code
    print("Running PySAT constraint functions demo...")
    result = execute_pysat_code(TEST_CODE, timeout=5)
    
    print("\nResult:")
    print(f"Success: {result['success']}")
    
    if result.get('error'):
        print(f"\nError: {result['error']}")
    
    print("\nOutput:")
    print(result['output'])
    
    if result.get('solution'):
        print("\nSolution:")
        print(result['solution'])

if __name__ == "__main__":
    main() 