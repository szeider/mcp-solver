"""
Test script to reproduce the satisfiability bug in the PySAT model manager.

The issue is that the PySAT model manager is returning incorrect satisfiability status
in the result. The model correctly solves the problem, but the satisfiability flag
is incorrectly set to False even when the problem is actually satisfiable.
"""

import sys
import os
import asyncio
from datetime import timedelta

# Add the src directory to the path so we can import from it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mcp_solver.pysat.model_manager import PySATModelManager

# A simple satisfiable PySAT problem
SATISFIABLE_MODEL = """
from pysat.formula import CNF
from pysat.solvers import Solver

# Create a simple satisfiable CNF formula
cnf = CNF()
cnf.append([1, 2, 3])  # clause: x1 OR x2 OR x3
cnf.append([-1, 2])    # clause: NOT x1 OR x2
cnf.append([3, 4])     # clause: x3 OR x4

# Create a solver
solver = Solver(name='glucose3')

# Add the formula to the solver
solver.append_formula(cnf)

# Solve the formula
is_sat = solver.solve()
print(f"Is satisfiable: {is_sat}")

# If satisfiable, print the model
if is_sat:
    model = solver.get_model()
    print(f"Model: {model}")
    
    # Map variable IDs to names
    variables = {
        "x1": 1,
        "x2": 2,
        "x3": 3,
        "x4": 4
    }
    
    # Export the solution
    export_solution(solver, variables)
else:
    # Export unsatisfiable result
    export_solution(solver)

# Clean up resources
solver.delete()
"""

# An unsatisfiable PySAT problem
UNSATISFIABLE_MODEL = """
from pysat.formula import CNF
from pysat.solvers import Solver

# Create a simple unsatisfiable CNF formula
cnf = CNF()
cnf.append([1])        # clause: x1
cnf.append([-1])       # clause: NOT x1

# Create a solver
solver = Solver(name='glucose3')

# Add the formula to the solver
solver.append_formula(cnf)

# Solve the formula
is_sat = solver.solve()
print(f"Is satisfiable: {is_sat}")

# If satisfiable, print the model (this won't execute)
if is_sat:
    model = solver.get_model()
    print(f"Model: {model}")
    
    # Map variable IDs to names
    variables = {
        "x1": 1,
    }
    
    # Export the solution
    export_solution(solver, variables)
else:
    # Export unsatisfiable result
    export_solution(solver)

# Clean up resources
solver.delete()
"""

async def test_satisfiable_model():
    """Test a satisfiable PySAT model."""
    print("\n=== Testing satisfiable PySAT model ===")
    
    # Initialize the model manager
    manager = PySATModelManager(lite_mode=True)
    
    # Clear the model and add our test model
    await manager.clear_model()
    await manager.add_item(0, SATISFIABLE_MODEL)
    
    # Solve the model
    timeout = timedelta(seconds=5)
    result = await manager.solve_model(timeout)
    
    # Print the result
    print("\nModel manager result:")
    print(f"Message: {result.get('message', 'No message')}")
    print(f"Success: {result.get('success', False)}")
    print(f"Satisfiable: {result.get('satisfiable', 'Not specified')}")
    print(f"Solve time: {result.get('solve_time', 'Unknown')}")
    
    # Get the solution
    solution = manager.get_solution()
    if solution:
        print("\nSolution details:")
        print(f"Satisfiable: {solution.get('satisfiable', 'Not specified')}")
        print(f"Status: {solution.get('status', 'Unknown')}")
        print(f"Values: {solution.get('values', {})}")
    
    # Validate the result
    is_correct = result.get('satisfiable', False) == True  # Should be True for satisfiable model
    print(f"\nTest result: {'✅ PASS' if is_correct else '❌ FAIL'}")
    return is_correct

async def test_unsatisfiable_model():
    """Test an unsatisfiable PySAT model."""
    print("\n=== Testing unsatisfiable PySAT model ===")
    
    # Initialize the model manager
    manager = PySATModelManager(lite_mode=True)
    
    # Clear the model and add our test model
    await manager.clear_model()
    await manager.add_item(0, UNSATISFIABLE_MODEL)
    
    # Solve the model
    timeout = timedelta(seconds=5)
    result = await manager.solve_model(timeout)
    
    # Print the result
    print("\nModel manager result:")
    print(f"Message: {result.get('message', 'No message')}")
    print(f"Success: {result.get('success', False)}")
    print(f"Satisfiable: {result.get('satisfiable', 'Not specified')}")
    print(f"Solve time: {result.get('solve_time', 'Unknown')}")
    
    # Get the solution
    solution = manager.get_solution()
    if solution:
        print("\nSolution details:")
        print(f"Satisfiable: {solution.get('satisfiable', 'Not specified')}")
        print(f"Status: {solution.get('status', 'Unknown')}")
        print(f"Values: {solution.get('values', {})}")
    
    # Validate the result
    is_correct = result.get('satisfiable', True) == False  # Should be False for unsatisfiable model
    print(f"\nTest result: {'✅ PASS' if is_correct else '❌ FAIL'}")
    return is_correct

async def run_all_tests():
    """Run all tests."""
    print("=== PySAT Satisfiability Bug Reproduction Test ===")
    
    results = {
        "satisfiable_model": await test_satisfiable_model(),
        "unsatisfiable_model": await test_unsatisfiable_model(),
    }
    
    print("\n=== Test Summary ===")
    for test, result in results.items():
        print(f"{test}: {'✅ PASS' if result else '❌ FAIL'}")
    
    all_passed = all(results.values())
    print(f"\nOverall result: {'✅ PASS' if all_passed else '❌ FAIL'}")
    
    return all_passed

if __name__ == "__main__":
    result = asyncio.run(run_all_tests())
    sys.exit(0 if result else 1) 