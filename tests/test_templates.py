"""
Test script for Z3 quantifier templates functionality.
This script tests the integration of the templates module with the Z3 environment.
"""

import sys
import os

# Add parent directory to path to ensure we can import from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mcp_solver.z3.environment import execute_z3_code
from mcp_solver.z3.templates import (
    array_is_sorted, all_distinct, array_contains,
    exactly_k, at_most_k, at_least_k,
    function_is_injective, function_is_surjective
)

# Test 1: Direct import of template functions
print("Test 1: Direct template function imports")
try:
    # Test array_is_sorted
    print("  Testing array_is_sorted... ", end="")
    from z3 import Array, IntSort, Solver, Int
    arr = Array('arr', IntSort(), IntSort())
    constraint = array_is_sorted(arr, 5)
    print("OK")
    
    # Test all_distinct
    print("  Testing all_distinct... ", end="")
    constraint = all_distinct(arr, 5)
    print("OK")
    
    # Test cardinality constraints
    print("  Testing cardinality constraints... ", end="")
    from z3 import Bool
    bool_vars = [Bool(f'b_{i}') for i in range(5)]
    constraint = exactly_k(bool_vars, 2)
    print("OK")
except Exception as e:
    print(f"FAILED: {e}")

# Test 2: Template function usage in execute_z3_code
print("\nTest 2: Template usage in Z3 environment")
test_code = """
from z3 import *
from z3_templates import array_is_sorted, all_distinct, exactly_k

# Define variables
n = 5
arr = Array('arr', IntSort(), IntSort())
s = Solver()

# Add template constraints
s.add(array_is_sorted(arr, n))
s.add(all_distinct(arr, n))

# Add specific values
s.add(arr[0] == 10)
s.add(arr[1] == 20)
s.add(arr[2] == 30)
s.add(arr[3] == 40)
s.add(arr[4] == 50)

# Export solution
export_solution(solver=s, variables={
    **{f'arr[{i}]': arr[i] for i in range(n)}
})
"""

result = execute_z3_code(test_code, timeout=5.0)
print(f"  Execution status: {result['status']}")
print(f"  Execution time: {result['execution_time']:.3f} seconds")

if result['status'] == 'success':
    print("  Solution values:")
    for key, value in result.get('solution', {}).get('values', {}).items():
        print(f"    {key}: {value}")
    print("PASSED")
else:
    print(f"  Error: {result.get('error', 'Unknown error')}")
    print(f"  Output: {result.get('output', [])}")
    print("FAILED")

# Test 3: Cardinality constraints
print("\nTest 3: Testing cardinality constraints")
test_code = """
from z3 import *
from z3_templates import exactly_k, at_most_k, at_least_k

# Define boolean variables
bool_vars = [Bool(f'b_{i}') for i in range(5)]
s = Solver()

# Add cardinality constraints
s.add(exactly_k(bool_vars, 2))  # Exactly 2 must be true

# Export solution
export_solution(solver=s, variables={
    **{f'b_{i}': bool_vars[i] for i in range(5)}
})
"""

result = execute_z3_code(test_code, timeout=5.0)
print(f"  Execution status: {result['status']}")

if result['status'] == 'success':
    print("  Solution values:")
    values = result.get('solution', {}).get('values', {})
    for key, value in values.items():
        print(f"    {key}: {value}")
    
    # Verify exactly 2 are true
    true_count = sum(1 for v in values.values() if v is True)
    print(f"  Number of True values: {true_count}")
    if true_count == 2:
        print("PASSED")
    else:
        print(f"FAILED: Expected exactly 2 true values, got {true_count}")
else:
    print(f"  Error: {result.get('error', 'Unknown error')}")
    print("FAILED")

# Test 4: Function properties
print("\nTest 4: Testing function properties")
test_code = """
from z3 import *
from z3_templates import function_is_injective

# Define a function as an array
n = 4
func = Array('func', IntSort(), IntSort())
s = Solver()

# Function must be injective (one-to-one)
s.add(function_is_injective(func, n))

# Domain is 0 to n-1, range is also 0 to n-1
for i in range(n):
    s.add(0 <= func[i], func[i] < n)

# Export solution
export_solution(solver=s, variables={
    **{f'func[{i}]': func[i] for i in range(n)}
})
"""

result = execute_z3_code(test_code, timeout=5.0)
print(f"  Execution status: {result['status']}")

if result['status'] == 'success':
    print("  Solution values:")
    values = result.get('solution', {}).get('values', {})
    for key, value in values.items():
        print(f"    {key}: {value}")
    
    # Verify all function values are distinct (injective property)
    func_values = [values[f'func[{i}]'] for i in range(4)]
    distinct_values = len(set(func_values)) == len(func_values)
    print(f"  Function values: {func_values}")
    if distinct_values:
        print("PASSED (Function is injective)")
    else:
        print("FAILED: Function is not injective")
else:
    print(f"  Error: {result.get('error', 'Unknown error')}")
    print("FAILED") 