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
from pysat.solvers import Solver, Glucose3

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

# A course scheduling problem as reported in the bug
COURSE_SCHEDULING_MODEL = """
from pysat.formula import CNF
from pysat.solvers import Glucose3
import itertools

# Define courses and time slots
courses = ["Algorithms", "Database Systems", "Computer Networks", "Operating Systems", "Software Engineering", "Machine Learning"]
time_slots = ["Monday Morning", "Monday Afternoon", "Wednesday Morning", "Wednesday Afternoon"]

# Create variable mapping
# Variable v_{c,t} means course c is scheduled at time slot t
var_map = {}
counter = 1

for course in courses:
    for slot in time_slots:
        var_map[(course, slot)] = counter
        counter += 1

# Create a CNF formula
formula = CNF()

# Constraint 1: Each course must be scheduled in exactly one time slot
for course in courses:
    # At least one time slot per course
    formula.append([var_map[(course, slot)] for slot in time_slots])
    
    # At most one time slot per course (no course can be scheduled twice)
    for slot1, slot2 in itertools.combinations(time_slots, 2):
        formula.append([-var_map[(course, slot1)], -var_map[(course, slot2)]])

# Constraint 2: At most 2 courses per time slot
for slot in time_slots:
    for c1, c2, c3 in itertools.combinations(courses, 3):
        formula.append([-var_map[(c1, slot)], -var_map[(c2, slot)], -var_map[(c3, slot)]])

# Constraint 3: Some specific requirements
# "Algorithms" cannot be scheduled with "Database Systems"
for slot in time_slots:
    formula.append([-var_map[("Algorithms", slot)], -var_map[("Database Systems", slot)]])

# Solve the problem
solver = Glucose3()
solver.append_formula(formula)
satisfiable = solver.solve()
model = solver.get_model() if satisfiable else None

print(f"Is satisfiable: {satisfiable}")
if satisfiable:
    print(f"Model (true variables): {model}")
    
    # Create a schedule from the model
    schedule = {slot: [] for slot in time_slots}
    for (course, slot), var_id in var_map.items():
        if var_id in model:
            schedule[slot].append(course)
    
    # Print the schedule
    print("Course Schedule Solution:")
    for slot, courses_in_slot in schedule.items():
        print(f"{slot}: {', '.join(courses_in_slot)}")
    
    # Export the solution with variables
    export_solution(solver, var_map)
else:
    print("No solution exists with the given constraints.")
    export_solution(solver)

# Clean up
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

async def test_course_scheduling_model():
    """Test the course scheduling model which shows the reported bug."""
    print("\n=== Testing Course Scheduling Model (Bug Reproduction) ===")
    
    # Initialize the model manager
    manager = PySATModelManager(lite_mode=True)
    
    # Clear the model and add our test model
    await manager.clear_model()
    await manager.add_item(0, COURSE_SCHEDULING_MODEL)
    
    # Solve the model
    timeout = timedelta(seconds=5)
    result = await manager.solve_model(timeout)
    
    # Print the result
    print("\nModel manager result:")
    print(f"Message: {result.get('message', 'No message')}")
    print(f"Success: {result.get('success', False)}")
    print(f"Satisfiable: {result.get('satisfiable', 'Not specified')}")
    print(f"Status: {result.get('status', 'Not specified')}")
    print(f"Solve time: {result.get('solve_time', 'Unknown')}")
    
    # Print the output to check if "Is satisfiable: True" appears
    output = result.get('output', '')
    print("\nOutput excerpt:")
    output_lines = output.split('\n')
    for line in output_lines:
        if 'Is satisfiable:' in line or 'Schedule Solution' in line:
            print(line)
    
    # Get the solution
    solution = manager.get_solution()
    if solution:
        print("\nSolution details:")
        print(f"Satisfiable: {solution.get('solution', {}).get('satisfiable', 'Not specified')}")
        print(f"Status: {solution.get('solution', {}).get('status', 'Unknown')}")
    
    # Check for the contradiction described in the bug report
    output_shows_satisfiable = "Is satisfiable: True" in output
    result_says_satisfiable = result.get('satisfiable', False)
    is_contradiction = output_shows_satisfiable and not result_says_satisfiable
    
    print(f"\nContradiction detected: {is_contradiction}")
    print(f"- Output says satisfiable: {output_shows_satisfiable}")
    print(f"- Result says satisfiable: {result_says_satisfiable}")
    
    # For debugging purposes, just check if the issue is reproduced, not if it's fixed
    is_correct = not is_contradiction
    print(f"\nTest result: {'✅ PASS' if is_correct else '❌ FAIL (Bug reproduced)'}")
    return is_correct

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
    print(f"Status: {result.get('status', 'Not specified')}")
    print(f"Solve time: {result.get('solve_time', 'Unknown')}")
    
    # Get the solution
    solution = manager.get_solution()
    if solution:
        print("\nSolution details:")
        print(f"Satisfiable: {solution.get('solution', {}).get('satisfiable', 'Not specified')}")
        print(f"Status: {solution.get('solution', {}).get('status', 'Unknown')}")
    
    # Check for the contradiction
    output_shows_satisfiable = "Is satisfiable: True" in result.get('output', '')
    result_says_satisfiable = result.get('satisfiable', False)
    is_contradiction = output_shows_satisfiable and not result_says_satisfiable
    
    print(f"\nContradiction detected: {is_contradiction}")
    print(f"- Output says satisfiable: {output_shows_satisfiable}")
    print(f"- Result says satisfiable: {result_says_satisfiable}")
    
    # Validate the result
    is_correct = not is_contradiction
    print(f"\nTest result: {'✅ PASS' if is_correct else '❌ FAIL (Bug reproduced)'}")
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
    print(f"Status: {result.get('status', 'Not specified')}")
    print(f"Solve time: {result.get('solve_time', 'Unknown')}")
    
    # Get the solution
    solution = manager.get_solution()
    if solution:
        print("\nSolution details:")
        print(f"Satisfiable: {solution.get('solution', {}).get('satisfiable', 'Not specified')}")
        print(f"Status: {solution.get('solution', {}).get('status', 'Unknown')}")
    
    # Check for the contradiction
    output_shows_satisfiable = "Is satisfiable: True" in result.get('output', '')
    result_says_satisfiable = result.get('satisfiable', False)
    # For unsatisfiable problems, we expect both to be False (no contradiction)
    is_contradiction = output_shows_satisfiable and not result_says_satisfiable
    
    print(f"\nContradiction detected: {is_contradiction}")
    print(f"- Output says satisfiable: {output_shows_satisfiable}")
    print(f"- Result says satisfiable: {result_says_satisfiable}")
    
    # Validate the result - for unsatisfiable problems, we expect result_says_satisfiable to be False
    is_correct = not output_shows_satisfiable and not result_says_satisfiable
    print(f"\nTest result: {'✅ PASS' if is_correct else '❌ FAIL'}")
    return is_correct

async def run_all_tests():
    """Run all tests."""
    print("=== PySAT Satisfiability Bug Reproduction Test ===")
    
    results = {
        "course_scheduling_model": await test_course_scheduling_model(),
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