"""
Test for direct conditional pattern in PySAT solver.

This script tests the direct conditional pattern to ensure our changes to model_manager.py 
and solution.py work correctly.
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

async def test_model_manager():
    """Test the direct conditional pattern in the model manager."""
    print("\n=== Testing with Model Manager ===")
    
    # Create model manager
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
    
    print("PASSED: Model manager test")
    return True

def run_all_tests():
    """Run all tests."""
    
    print("=== Testing Direct Conditional Pattern ===")
    
    direct_success = test_direct_conditional()
    
    # Test with model manager
    asyncio.run(test_model_manager())
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Direct Conditional: {'✅ Passed' if direct_success else '❌ Failed'}")
    print(f"Elapsed time: {time.time() - start_time:.3f} seconds")
    
    if direct_success:
        print("\n✅ ALL TESTS PASSED")
    else:
        print("\n❌ TESTS FAILED")
    
    return direct_success

if __name__ == "__main__":
    start_time = time.time()
    success = run_all_tests()
    sys.exit(0 if success else 1) 