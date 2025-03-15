"""
Test script for Z3 function templates functionality.
This script tests the integration of the function templates with the Z3 environment.
"""

import sys
from src.mcp_solver.z3.environment import execute_z3_code
from src.mcp_solver.z3.templates import (
    constraint_satisfaction_template,
    optimization_template,
    array_template,
    quantifier_template
)

def print_separator():
    print("-" * 50)

print("Z3 Function Templates Test Script")
print_separator()

# Test 1: Direct import of function templates
print("Test 1: Direct function template imports")
try:
    # Test constraint satisfaction template
    print("  Testing constraint_satisfaction_template... ", end="")
    s, vars = constraint_satisfaction_template()
    print("OK")
    
    # Test optimization template
    print("  Testing optimization_template... ", end="")
    opt, vars = optimization_template()
    print("OK")
    
    # Test array template
    print("  Testing array_template... ", end="")
    s, vars = array_template()
    print("OK")
    
    # Test quantifier template
    print("  Testing quantifier_template... ", end="")
    s, vars = quantifier_template()
    print("OK")
except Exception as e:
    print(f"FAILED: {e}")

print_separator()

# Test 2: Constraint satisfaction template in Z3 environment
print("Test 2: Constraint satisfaction template in Z3 environment")
test_code = """
from z3 import *
from mcp_solver.z3.templates import constraint_satisfaction_template

def build_model():
    # Define variables
    x = Int('x')
    y = Int('y')
    
    # Create solver
    s = Solver()
    
    # Add constraints
    s.add(x > 0)
    s.add(y > 0)
    s.add(x + y <= 10)
    
    return s, {'x': x, 'y': y}

solver, variables = build_model()
export_solution(solver=solver, variables=variables)
"""

try:
    result = execute_z3_code(test_code)
    print(f"  Result: {result['status']}")
    if result['status'] == 'sat':
        print(f"  Variables: {result['variables']}")
    print("  Constraint satisfaction template test: OK")
except Exception as e:
    print(f"  FAILED: {e}")

print_separator()

# Test 3: Optimization template in Z3 environment
print("Test 3: Optimization template in Z3 environment")
test_code = """
from z3 import *
from mcp_solver.z3.templates import optimization_template

def build_model():
    # Define variables
    x = Int('x')
    y = Int('y')
    
    # Create optimizer
    opt = Optimize()
    
    # Add constraints
    opt.add(x >= 0)
    opt.add(y >= 0)
    opt.add(x + y <= 10)
    
    # Define objective
    objective = x + 2*y
    opt.maximize(objective)
    
    return opt, {'x': x, 'y': y, 'objective': objective}

solver, variables = build_model()
export_solution(solver=solver, variables=variables)
"""

try:
    result = execute_z3_code(test_code)
    print(f"  Result: {result['status']}")
    if result['status'] == 'sat':
        print(f"  Variables: {result['variables']}")
    print("  Optimization template test: OK")
except Exception as e:
    print(f"  FAILED: {e}")

print_separator()

# Test 4: Array template in Z3 environment
print("Test 4: Array template in Z3 environment")
test_code = """
from z3 import *
from mcp_solver.z3.templates import array_template

def build_model():
    # Define size
    n = 3
    
    # Define array
    arr = Array('arr', IntSort(), IntSort())
    
    # Create solver
    s = Solver()
    
    # Add constraints
    for i in range(n):
        s.add(arr[i] >= 0, arr[i] < 10)
    
    s.add(arr[0] < arr[1])
    s.add(arr[1] < arr[2])
    
    return s, {f'arr[{i}]': arr[i] for i in range(n)}

solver, variables = build_model()
export_solution(solver=solver, variables=variables)
"""

try:
    result = execute_z3_code(test_code)
    print(f"  Result: {result['status']}")
    if result['status'] == 'sat':
        print(f"  Variables: {result['variables']}")
    print("  Array template test: OK")
except Exception as e:
    print(f"  FAILED: {e}")

print_separator()

print("All tests completed!")

if __name__ == "__main__":
    print("\nRun this script with: uv run python test_function_templates.py") 