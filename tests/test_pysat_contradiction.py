"""
Test script to reproduce the specific bug where PySAT model manager reports 
contradictory satisfiability results (showing satisfiable in output but False in results).

This test focuses directly on the reported bug which shows a mismatch between:
1. The raw solver output (showing "Is satisfiable: True" and a valid solution)
2. The satisfiability flag in the function result (showing False)
3. The status in the function result (showing "unsat")
"""

import sys
import os
import asyncio
from datetime import timedelta
import json
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Add the src directory to the path so we can import from it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mcp_solver.pysat.model_manager import PySATModelManager
from src.mcp_solver.pysat.environment import execute_pysat_code

# A course scheduling problem - satisfiable version
COURSE_SCHEDULING_SATISFIABLE = """
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
is_sat = solver.solve()
print(f"Is satisfiable: {is_sat}")

# If satisfiable, print the model
if is_sat:
    model = solver.get_model()
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
else:
    print("No solution exists with the given constraints.")

# Clean up
solver.delete()
"""

# An unsatisfiable version of the course scheduling problem - with contradictory constraints
COURSE_SCHEDULING_UNSATISFIABLE = """
from pysat.formula import CNF
from pysat.solvers import Glucose3
import itertools

# Define courses and time slots - using fewer time slots to make it unsatisfiable
courses = ["Algorithms", "Database Systems", "Computer Networks", "Operating Systems", "Software Engineering", "Machine Learning"]
time_slots = ["Monday Morning", "Monday Afternoon"] # Only 2 time slots for 6 courses with max 2 per slot

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

# Constraint 2: At most 2 courses per time slot - this makes it unsatisfiable (6 courses, 2 slots, max 2 per slot)
for slot in time_slots:
    for c1, c2, c3 in itertools.combinations(courses, 3):
        formula.append([-var_map[(c1, slot)], -var_map[(c2, slot)], -var_map[(c3, slot)]])

# Solve the problem
solver = Glucose3()
solver.append_formula(formula)
is_sat = solver.solve()
print(f"Is satisfiable: {is_sat}")

# If satisfiable, print the model
if is_sat:
    model = solver.get_model()
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
else:
    print("No solution exists with the given constraints.")

# Clean up
solver.delete()
"""

async def test_satisfiable_case():
    """Test a satisfiable case to ensure it's correctly reported."""
    print("\n=== Testing Satisfiable Course Scheduling Model ===")
    
    # Initialize the model manager
    manager = PySATModelManager(lite_mode=True)
    
    # Clear the model and add our test model
    await manager.clear_model()
    await manager.add_item(0, COURSE_SCHEDULING_SATISFIABLE)
    
    # Solve the model
    timeout = timedelta(seconds=5)
    result = await manager.solve_model(timeout)
    
    # Print the result
    print("\nModel manager result:")
    print(f"Success: {result.get('success', False)}")
    print(f"Satisfiable: {result.get('satisfiable', 'Not specified')}")
    print(f"Status: {result.get('status', 'Not specified')}")
    
    # Print the output to check if "Is satisfiable: True" appears
    output = result.get('output', '')
    print("\nOutput excerpt:")
    output_lines = output.split('\n')
    for line in output_lines:
        if 'Is satisfiable:' in line or 'Course Schedule Solution' in line:
            print(line)
    
    # Get the solution
    solution = manager.get_solution()
    if solution and 'solution' in solution:
        print("\nSolution details:")
        print(f"Satisfiable: {solution['solution'].get('satisfiable', 'Not specified')}")
        print(f"Status: {solution['solution'].get('status', 'Unknown')}")
    
    # Check for inconsistencies
    output_shows_satisfiable = "Is satisfiable: True" in output
    result_says_satisfiable = result.get('satisfiable', False)
    solution_says_satisfiable = solution.get('solution', {}).get('satisfiable', False)
    
    # Test passes if no inconsistency is detected
    no_contradiction = (
        output_shows_satisfiable == result_says_satisfiable == solution_says_satisfiable
    )
    
    print(f"\nTest result: {'✅ PASS' if no_contradiction else '❌ FAIL'}")
    return no_contradiction

async def test_unsatisfiable_case():
    """Test an unsatisfiable case to ensure it's correctly reported."""
    print("\n=== Testing Unsatisfiable Course Scheduling Model ===")
    
    # Initialize the model manager
    manager = PySATModelManager(lite_mode=True)
    
    # Clear the model and add our test model
    await manager.clear_model()
    await manager.add_item(0, COURSE_SCHEDULING_UNSATISFIABLE)
    
    # Solve the model
    timeout = timedelta(seconds=5)
    result = await manager.solve_model(timeout)
    
    # Print the result
    print("\nModel manager result:")
    print(f"Success: {result.get('success', False)}")
    print(f"Satisfiable: {result.get('satisfiable', 'Not specified')}")
    print(f"Status: {result.get('status', 'Not specified')}")
    
    # Print the output to check if "Is satisfiable: False" appears
    output = result.get('output', '')
    print("\nOutput excerpt:")
    output_lines = output.split('\n')
    for line in output_lines:
        if 'Is satisfiable:' in line:
            print(line)
    
    # Get the solution
    solution = manager.get_solution()
    if solution and 'solution' in solution:
        print("\nSolution details:")
        print(f"Satisfiable: {solution['solution'].get('satisfiable', 'Not specified')}")
        print(f"Status: {solution['solution'].get('status', 'Unknown')}")
    
    # Check for inconsistencies
    output_shows_unsatisfiable = "Is satisfiable: False" in output
    result_says_unsatisfiable = not result.get('satisfiable', True)
    solution_says_unsatisfiable = not solution.get('solution', {}).get('satisfiable', True)
    
    # Test passes if no inconsistency is detected
    no_contradiction = (
        output_shows_unsatisfiable == result_says_unsatisfiable == solution_says_unsatisfiable
    )
    
    print(f"\nTest result: {'✅ PASS' if no_contradiction else '❌ FAIL'}")
    return no_contradiction

async def run_all_tests():
    """Run all tests and report results."""
    print("Running PySAT satisfiability consistency tests...")
    
    satisfiable_passed = await test_satisfiable_case()
    unsatisfiable_passed = await test_unsatisfiable_case()
    
    all_passed = satisfiable_passed and unsatisfiable_passed
    
    print("\n=== Test Summary ===")
    print(f"Satisfiable Case: {'✅ PASS' if satisfiable_passed else '❌ FAIL'}")
    print(f"Unsatisfiable Case: {'✅ PASS' if unsatisfiable_passed else '❌ FAIL'}")
    print(f"Overall Result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    result = asyncio.run(run_all_tests())
    sys.exit(0 if result else 1) 