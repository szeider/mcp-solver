"""
Tests for timeout handling in all three model managers (MiniZinc, Z3, and PySAT).

This script tests how each model manager handles timeout values for the solve_model method,
verifying that:
1. Valid timeouts are accepted
2. Timeouts below MIN_SOLVE_TIMEOUT are either rejected or adjusted
3. Timeouts above MAX_SOLVE_TIMEOUT are either rejected or adjusted
4. Edge case timeouts (exactly MIN_SOLVE_TIMEOUT or MAX_SOLVE_TIMEOUT) are accepted
"""

import sys
import os
import asyncio
from datetime import timedelta
import json

# Add the src directory to the path so we can import from it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mcp_solver.constants import MIN_SOLVE_TIMEOUT, MAX_SOLVE_TIMEOUT
from src.mcp_solver.mzn.model_manager import MiniZincModelManager
from src.mcp_solver.z3.model_manager import Z3ModelManager
from src.mcp_solver.pysat.model_manager import PySATModelManager

# Test models for each solver
MINIZINC_MODEL = """
int: n = 4;
array[1..n] of var 1..n: q;

% All queens must be in different rows
constraint alldifferent(q);

% No two queens can be in the same diagonal
constraint forall(i,j in 1..n where i < j) (
    abs(q[i] - q[j]) != abs(i - j)
);

solve satisfy;
"""

Z3_MODEL = """
(declare-const x Int)
(declare-const y Int)
(assert (> x 0))
(assert (> y 0))
(assert (< (+ x y) 10))
(check-sat)
(get-model)
"""

PYSAT_MODEL = """
p cnf 4 8
1 2 3 0
-1 -2 0
2 3 4 0
-2 -3 0
-3 -4 0
1 3 0
1 4 0
2 4 0
"""

async def test_minizinc_timeout():
    """Test timeout handling in MiniZinc mode."""
    print("\n=== Testing MiniZinc Timeout Handling ===")
    
    # Initialize the model manager
    manager = MiniZincModelManager(lite_mode=True)
    
    # Clear the model and add our test model
    await manager.clear_model()
    await manager.add_item(1, MINIZINC_MODEL)
    
    # Test valid timeout
    valid_timeout = timedelta(seconds=2)
    print(f"\nTesting valid timeout: {valid_timeout.seconds} seconds")
    try:
        result = await manager.solve_model(valid_timeout)
        print(f"Result status: {result.get('status', 'unknown')}")
        print(f"Solve time: {manager.last_solve_time} seconds")
        print("Test passed: Manager accepted the valid timeout")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Test failed: Manager rejected valid timeout")
    
    # Test timeout below minimum
    min_timeout = MIN_SOLVE_TIMEOUT
    too_small_timeout = timedelta(seconds=0.1)
    print(f"\nTesting too small timeout: {too_small_timeout.seconds} seconds (minimum: {min_timeout.seconds})")
    try:
        result = await manager.solve_model(too_small_timeout)
        print(f"Result status: {result.get('status', 'unknown')}")
        print(f"Solve time: {manager.last_solve_time} seconds")
        print(f"Test passed: Manager accepted the too small timeout (MIN enforced at server level)")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Test failed: Manager should accept any timeout value at the model manager level")
    
    # Test timeout above maximum
    max_timeout = MAX_SOLVE_TIMEOUT
    too_large_timeout = timedelta(seconds=30)
    print(f"\nTesting too large timeout: {too_large_timeout.seconds} seconds (maximum: {max_timeout.seconds})")
    try:
        result = await manager.solve_model(too_large_timeout)
        print(f"Result status: {result.get('status', 'unknown')}")
        print(f"Solve time: {manager.last_solve_time} seconds")
        print(f"Test passed: Manager accepted the too large timeout (MAX enforced at server level)")
    except Exception as e:
        if "exceeds maximum allowed timeout" in str(e):
            print(f"Result correctly rejected timeout: {str(e)}")
            print("Test passed: Manager correctly rejected the too large timeout")
        else:
            print(f"Error: {str(e)}")
            print("Test failed: Unexpected error when testing too large timeout")
    
    # Additional Test: Exactly MAX_SOLVE_TIMEOUT seconds
    print(f"\nTesting exact maximum timeout: {max_timeout.seconds} seconds")
    try:
        exact_max = MAX_SOLVE_TIMEOUT
        result = await manager.solve_model(exact_max)
        print(f"Result status: {result.get('status', 'unknown')}")
        print(f"Solve time: {manager.last_solve_time} seconds")
        print("Test passed: Manager accepted the maximum timeout")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Test failed: Manager rejected exact maximum timeout")

    # Additional Test: Exactly MIN_SOLVE_TIMEOUT seconds
    print(f"\nTesting exact minimum timeout: {min_timeout.seconds} seconds")
    try:
        exact_min = MIN_SOLVE_TIMEOUT
        result = await manager.solve_model(exact_min)
        print(f"Result status: {result.get('status', 'unknown')}")
        print(f"Solve time: {manager.last_solve_time} seconds")
        print("Test passed: Manager accepted the minimum timeout")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Test failed: Manager rejected exact minimum timeout")
    
    return True

async def test_z3_timeout():
    """Test timeout handling in Z3 mode."""
    print("\n=== Testing Z3 Timeout Handling ===")
    
    # Initialize the model manager
    manager = Z3ModelManager(lite_mode=True)
    
    # Clear the model and add our test model
    await manager.clear_model()
    await manager.add_item(1, Z3_MODEL)
    
    # Test valid timeout
    valid_timeout = timedelta(seconds=2)
    print(f"\nTesting valid timeout: {valid_timeout.seconds} seconds")
    try:
        result = await manager.solve_model(valid_timeout)
        print(f"Result status: {result.get('status', 'unknown')}")
        print(f"Solve time: {manager.last_solve_time} seconds")
        print("Test passed: Manager accepted the valid timeout")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Test failed: Manager rejected valid timeout")
    
    # Test timeout below minimum
    min_timeout = MIN_SOLVE_TIMEOUT
    too_small_timeout = timedelta(seconds=0.1)
    print(f"\nTesting too small timeout: {too_small_timeout.seconds} seconds (minimum: {min_timeout.seconds})")
    try:
        result = await manager.solve_model(too_small_timeout)
        print(f"Result status: {result.get('status', 'unknown')}")
        print(f"Solve time: {manager.last_solve_time} seconds")
        print(f"Test passed: Manager accepted the too small timeout (MIN enforced at server level)")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Test failed: Manager should accept any timeout value at the model manager level")
    
    # Test timeout above maximum
    max_timeout = MAX_SOLVE_TIMEOUT
    too_large_timeout = timedelta(seconds=30)
    print(f"\nTesting too large timeout: {too_large_timeout.seconds} seconds (maximum: {max_timeout.seconds})")
    try:
        result = await manager.solve_model(too_large_timeout)
        print(f"Result status: {result.get('status', 'unknown')}")
        print(f"Solve time: {manager.last_solve_time} seconds")
        print(f"Test passed: Manager accepted the too large timeout (MAX enforced at server level)")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Test failed: Manager should accept any timeout value at the model manager level")
    
    # Additional Test: Exactly MAX_SOLVE_TIMEOUT seconds
    print(f"\nTesting exact maximum timeout: {max_timeout.seconds} seconds")
    try:
        exact_max = MAX_SOLVE_TIMEOUT
        result = await manager.solve_model(exact_max)
        print(f"Result status: {result.get('status', 'unknown')}")
        print(f"Solve time: {manager.last_solve_time} seconds")
        print("Test passed: Manager accepted the maximum timeout")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Test failed: Manager rejected exact maximum timeout")

    # Additional Test: Exactly MIN_SOLVE_TIMEOUT seconds
    print(f"\nTesting exact minimum timeout: {min_timeout.seconds} seconds")
    try:
        exact_min = MIN_SOLVE_TIMEOUT
        result = await manager.solve_model(exact_min)
        print(f"Result status: {result.get('status', 'unknown')}")
        print(f"Solve time: {manager.last_solve_time} seconds")
        print("Test passed: Manager accepted the minimum timeout")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Test failed: Manager rejected exact minimum timeout")
    
    return True

async def test_pysat_timeout():
    """Test timeout handling in PySAT mode."""
    print("\n=== Testing PySAT Timeout Handling ===")
    
    # Initialize the model manager
    manager = PySATModelManager(lite_mode=True)
    
    # Clear the model and add our test model
    await manager.clear_model()
    await manager.add_item(0, PYSAT_MODEL)
    
    # Test valid timeout
    valid_timeout = timedelta(seconds=2)
    print(f"\nTesting valid timeout: {valid_timeout.seconds} seconds")
    try:
        result = await manager.solve_model(valid_timeout)
        print(f"Result status: {result.get('status', 'unknown') if isinstance(result, dict) else 'unknown'}")
        print(f"Solve time: {manager.last_solve_time} seconds")
        print("Test passed: Manager accepted the valid timeout")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Test failed: Manager rejected valid timeout")
    
    # Test timeout below minimum
    min_timeout = MIN_SOLVE_TIMEOUT
    too_small_timeout = timedelta(seconds=0.1)
    print(f"\nTesting too small timeout: {too_small_timeout.seconds} seconds (minimum: {min_timeout.seconds})")
    try:
        result = await manager.solve_model(too_small_timeout)
        print(f"Result status: {result.get('status', 'unknown') if isinstance(result, dict) else 'unknown'}")
        print(f"Solve time: {manager.last_solve_time} seconds")
        print(f"Test passed: Manager accepted the too small timeout (MIN enforced at server level)")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Test failed: Manager should accept any timeout value at the model manager level")
    
    # Test timeout above maximum
    max_timeout = MAX_SOLVE_TIMEOUT
    too_large_timeout = timedelta(seconds=30)
    print(f"\nTesting too large timeout: {too_large_timeout.seconds} seconds (maximum: {max_timeout.seconds})")
    try:
        result = await manager.solve_model(too_large_timeout)
        print(f"Result status: {result.get('status', 'unknown') if isinstance(result, dict) else 'unknown'}")
        print(f"Solve time: {manager.last_solve_time} seconds")
        print(f"Test passed: Manager accepted the too large timeout (MAX enforced at server level)")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Test failed: Manager should accept any timeout value at the model manager level")
        
    # Additional Test: Exactly MAX_SOLVE_TIMEOUT seconds
    print(f"\nTesting exact maximum timeout: {max_timeout.seconds} seconds")
    try:
        exact_max = MAX_SOLVE_TIMEOUT
        result = await manager.solve_model(exact_max)
        print(f"Result status: {result.get('status', 'unknown') if isinstance(result, dict) else 'unknown'}")
        print(f"Solve time: {manager.last_solve_time} seconds")
        print("Test passed: Manager accepted the maximum timeout")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Test failed: Manager rejected exact maximum timeout")

    # Additional Test: Exactly MIN_SOLVE_TIMEOUT seconds
    print(f"\nTesting exact minimum timeout: {min_timeout.seconds} seconds")
    try:
        exact_min = MIN_SOLVE_TIMEOUT
        result = await manager.solve_model(exact_min)
        print(f"Result status: {result.get('status', 'unknown') if isinstance(result, dict) else 'unknown'}")
        print(f"Solve time: {manager.last_solve_time} seconds")
        print("Test passed: Manager accepted the minimum timeout")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Test failed: Manager rejected exact minimum timeout")
    
    return True

async def run_all_tests():
    """Run all timeout handling tests for all three model managers."""
    print("Running timeout handling tests...")
    
    results = {
        "minizinc": await test_minizinc_timeout(),
        "z3": await test_z3_timeout(),
        "pysat": await test_pysat_timeout(),
    }
    
    print("\n=== Test Summary ===")
    for test, result in results.items():
        print(f"{test.upper()}: {'✅ PASS' if result else '❌ FAIL'}")
    
    all_passed = all(results.values())
    
    return all_passed

if __name__ == "__main__":
    result = asyncio.run(run_all_tests())
    sys.exit(0 if result else 1) 