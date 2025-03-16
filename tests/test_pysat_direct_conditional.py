"""
Test for direct conditional pattern in PySAT solver.

This script tests both the direct conditional pattern and variable assignment pattern
to ensure our changes to model_manager.py and solution.py work correctly.
"""

import sys
import os
import time
import json
import asyncio

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath('src'))

from mcp_solver.pysat.environment import execute_pysat_code
from mcp_solver.pysat.model_manager import PySATModelManager
from datetime import timedelta

def test_direct_conditional():
    """Test using direct conditional pattern."""
    code = """
from pysat.formula import CNF
from pysat.solvers import Glucose3

# Create a CNF formula
formula = CNF()

# Add clauses: (a OR b) AND (NOT a OR c) AND (NOT b OR NOT c)
formula.append([1, 2])       # Clause 1: a OR b
formula.append([-1, 3])      # Clause 2: NOT a OR c
formula.append([-2, -3])     # Clause 3: NOT b OR NOT c

# Create solver and add the formula
solver = Glucose3()
solver.append_formula(formula)

# Using direct conditional check
if solver.solve():
    model = solver.get_model()
    result = {"satisfiable": True, "model": model, "test_type": "direct_conditional"}
    export_solution(result)
else:
    export_solution({"satisfiable": False, "test_type": "direct_conditional"})

# Free solver memory
solver.delete()
"""
    result = execute_pysat_code(code)
    print("Direct conditional pattern result:")
    print(json.dumps(result, indent=2))
    
    # Verify the result
    success = result.get("success", False)
    solution = result.get("solution", {})
    satisfiable = solution.get("satisfiable", False)
    
    print(f"Success: {success}")
    print(f"Satisfiable: {satisfiable}")
    print(f"Test type: {solution.get('test_type', 'unknown')}")
    
    assert success, "Execution should succeed"
    assert satisfiable, "Problem should be satisfiable"
    assert solution.get("test_type") == "direct_conditional", "Should use direct conditional pattern"
    
    print("PASSED: Direct conditional pattern test")
    print()
    return success and satisfiable


def test_variable_assignment():
    """Test using variable assignment pattern."""
    code = """
from pysat.formula import CNF
from pysat.solvers import Glucose3

# Create a CNF formula
formula = CNF()

# Add clauses: (a OR b) AND (NOT a OR c) AND (NOT b OR NOT c)
formula.append([1, 2])       # Clause 1: a OR b
formula.append([-1, 3])      # Clause 2: NOT a OR c
formula.append([-2, -3])     # Clause 3: NOT b OR NOT c

# Create solver and add the formula
solver = Glucose3()
solver.append_formula(formula)

# Using variable assignment
satisfiable = solver.solve()
if satisfiable:
    model = solver.get_model()
    result = {"satisfiable": True, "model": model, "test_type": "variable_assignment"}
    export_solution(result)
else:
    export_solution({"satisfiable": False, "test_type": "variable_assignment"})

# Free solver memory
solver.delete()
"""
    result = execute_pysat_code(code)
    print("Variable assignment pattern result:")
    print(json.dumps(result, indent=2))
    
    # Verify the result
    success = result.get("success", False)
    solution = result.get("solution", {})
    satisfiable = solution.get("satisfiable", False)
    
    print(f"Success: {success}")
    print(f"Satisfiable: {satisfiable}")
    print(f"Test type: {solution.get('test_type', 'unknown')}")
    
    assert success, "Execution should succeed"
    assert satisfiable, "Problem should be satisfiable"
    assert solution.get("test_type") == "variable_assignment", "Should use variable assignment pattern"
    
    print("PASSED: Variable assignment pattern test")
    print()
    return success and satisfiable


async def test_model_manager():
    """Test using model manager with both patterns."""
    
    print("\n=== Testing with Model Manager ===")
    
    # Create a model manager
    manager = PySATModelManager(lite_mode=True)
    
    # Test with direct conditional
    print("\nTesting direct conditional pattern with model manager:")
    await manager.clear_model()
    
    direct_code = """
from pysat.formula import CNF
from pysat.solvers import Glucose3

# Create a CNF formula
formula = CNF()
formula.append([1, 2])
formula.append([-1, 3])
formula.append([-2, -3])

# Create solver and add the formula
solver = Glucose3()
solver.append_formula(formula)

# Using direct conditional check
if solver.solve():
    model = solver.get_model()
    print("Direct conditional: satisfiable")
else:
    print("Direct conditional: unsatisfiable")
"""
    
    await manager.add_item(1, direct_code)
    result_direct = await manager.solve_model(timeout=timedelta(seconds=5))
    
    print(f"Success: {result_direct.get('success', False)}")
    print(f"Satisfiable: {result_direct.get('satisfiable', False)}")
    print(f"Output: {result_direct.get('output', '')}")
    
    assert result_direct.get('success', False), "Execution should succeed"
    assert result_direct.get('satisfiable', False), "Problem should be satisfiable"
    
    # Test with variable assignment
    print("\nTesting variable assignment pattern with model manager:")
    await manager.clear_model()
    
    var_code = """
from pysat.formula import CNF
from pysat.solvers import Glucose3

# Create a CNF formula
formula = CNF()
formula.append([1, 2])
formula.append([-1, 3])
formula.append([-2, -3])

# Create solver and add the formula
solver = Glucose3()
solver.append_formula(formula)

# Using variable assignment
satisfiable = solver.solve()
if satisfiable:
    model = solver.get_model()
    print("Variable assignment: satisfiable")
else:
    print("Variable assignment: unsatisfiable")
"""
    
    await manager.add_item(1, var_code)
    result_var = await manager.solve_model(timeout=timedelta(seconds=5))
    
    print(f"Success: {result_var.get('success', False)}")
    print(f"Satisfiable: {result_var.get('satisfiable', False)}")
    print(f"Output: {result_var.get('output', '')}")
    
    assert result_var.get('success', False), "Execution should succeed"
    assert result_var.get('satisfiable', False), "Problem should be satisfiable"
    
    print("PASSED: Model manager test")
    return True


def run_all_tests():
    """Run all tests."""
    print("=== Testing Both PySAT Code Patterns ===\n")
    
    start_time = time.time()
    
    # Test direct conditional
    print("=== Direct Conditional Pattern Test ===")
    direct_success = test_direct_conditional()
    
    # Test variable assignment
    print("=== Variable Assignment Pattern Test ===")
    var_success = test_variable_assignment()
    
    # Test model manager
    asyncio.run(test_model_manager())
    
    elapsed_time = time.time() - start_time
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Direct Conditional: {'✅ Passed' if direct_success else '❌ Failed'}")
    print(f"Variable Assignment: {'✅ Passed' if var_success else '❌ Failed'}")
    print(f"Elapsed time: {elapsed_time:.3f} seconds")
    
    if direct_success and var_success:
        print("\n✅ ALL TESTS PASSED")
        return True
    else:
        print("\n❌ SOME TESTS FAILED")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 