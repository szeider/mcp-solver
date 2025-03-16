#!/usr/bin/env python3
"""
University Course Scheduling MaxSAT Example

This example demonstrates the use of the improved MaxSAT functionality
to solve a university course scheduling problem with both hard constraints
and soft preferences.
"""

# To run this example directly, make sure mcp-solver is installed or in the Python path
# Add the src directory to the path if running from the mcp-solver directory
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import the Global MaxSAT functionality
from mcp_solver.pysat.global_maxsat import (
    initialize_maxsat, add_hard_clause, add_soft_clause, 
    solve_maxsat, get_current_wcnf
)

def main():
    """
    Main entry point for the university course scheduling example.
    """
    print("University Course Scheduling Problem (MaxSAT)")
    print("=============================================")
    
    # Initialize a new MaxSAT problem
    initialize_maxsat()
    
    # We'll model a university with 6 courses and 4 time slots
    # Variables: course c in time slot t = 4*(c-1) + t
    # So course 1 in time slot 1 = variable 1, course 1 in timeslot 2 = variable 2, etc.
    
    print("\nProblem Definition:")
    print("- 6 courses to be scheduled in 4 time slots")
    print("- Each course must be scheduled exactly once")
    print("- At most 2 courses can be scheduled in each time slot")
    print("- Various professor and student preferences (soft constraints)")
    
    # Hard constraints
    # Each course must be scheduled exactly once
    
    # At least one time slot per course
    for c in range(1, 7):
        base = 4 * (c - 1)
        add_hard_clause([base + 1, base + 2, base + 3, base + 4])
    
    # At most one time slot per course
    for c in range(1, 7):
        base = 4 * (c - 1)
        for t1 in range(1, 4):
            for t2 in range(t1 + 1, 5):
                add_hard_clause([-(base + t1), -(base + t2)])
    
    # At most 2 courses per time slot
    for t in range(1, 5):
        courses_in_slot_t = []
        for c in range(1, 7):
            base = 4 * (c - 1)
            courses_in_slot_t.append(base + t)
        
        # Add "at most 2" constraint for each combination of 3 courses
        import itertools
        for combo in itertools.combinations(courses_in_slot_t, 3):
            add_hard_clause([-combo[0], -combo[1], -combo[2]])
    
    # Soft constraints (preferences)
    print("\nSoft Constraints (Preferences):")
    
    # Professor A prefers not to teach on Monday afternoon
    print("- Professor A (teaching courses 1, 4) prefers not to teach on Monday afternoon (time slot 2)")
    add_soft_clause([-(4*(1-1)+2)], weight=2)  # Course 1 not in time slot 2
    add_soft_clause([-(4*(4-1)+2)], weight=2)  # Course 4 not in time slot 2
    
    # Professor B prefers not to teach on Wednesday
    print("- Professor B (teaching courses 2, 5) prefers not to teach on Wednesday (time slots 3, 4)")
    for course in [2, 5]:
        for timeslot in [3, 4]:
            add_soft_clause([-(4*(course-1)+timeslot)], weight=3)
    
    # Professor C prefers afternoon slots
    print("- Professor C (teaching courses 3, 6) prefers afternoon slots (time slots 2, 4)")
    for course in [3, 6]:
        for timeslot in [1, 3]:  # Morning slots
            add_soft_clause([-(4*(course-1)+timeslot)], weight=2)
    
    # Student constraint: Algorithms and OS should not be at the same time
    # Let's say course 1 is Algorithms and course 3 is OS
    print("- Students prefer Algorithms (course 1) and OS (course 3) not to be at the same time")
    for t in range(1, 5):
        add_soft_clause([-(4*(1-1)+t), -(4*(3-1)+t)], weight=4)
    
    # Solve the problem
    print("\nSolving...")
    model, cost = solve_maxsat(timeout=10.0)
    
    # Create a more user-friendly representation of the solution
    if model is not None:
        print("\nSolution found!")
        print(f"Objective value (penalty): {cost}")
        
        # Create a schedule representation
        schedule = {}
        for t in range(1, 5):
            schedule[t] = []
        
        for c in range(1, 7):
            for t in range(1, 5):
                var_id = 4 * (c - 1) + t
                if var_id in model:
                    schedule[t].append(c)
        
        # Print the schedule
        print("\nSchedule:")
        for t in range(1, 5):
            slot_name = {
                1: "Monday Morning   (Slot 1)",
                2: "Monday Afternoon (Slot 2)",
                3: "Wednesday Morning   (Slot 3)",
                4: "Wednesday Afternoon (Slot 4)"
            }[t]
            print(f"{slot_name}: {', '.join(f'Course {c}' for c in schedule[t])}")
        
        # Check which soft constraints were violated
        print("\nSoft constraint violations:")
        
        # Professor A: Courses 1, 4 not in slot 2
        if 2 in schedule[2]:
            print("- Professor A has to teach on Monday afternoon")
        
        # Professor B: Courses 2, 5 not in slots 3, 4
        b_violations = []
        for c in [2, 5]:
            for t in [3, 4]:
                if c in schedule[t]:
                    b_violations.append(f"Course {c} in slot {t}")
        if b_violations:
            print(f"- Professor B has to teach on Wednesday: {', '.join(b_violations)}")
        
        # Professor C: Courses 3, 6 prefer slots 2, 4
        c_violations = []
        for c in [3, 6]:
            for t in [1, 3]:
                if c in schedule[t]:
                    c_violations.append(f"Course {c} in slot {t}")
        if c_violations:
            print(f"- Professor C has morning classes: {', '.join(c_violations)}")
        
        # Student constraint: Courses 1 and 3 not at same time
        for t in range(1, 5):
            if 1 in schedule[t] and 3 in schedule[t]:
                print(f"- Algorithms and OS are scheduled at the same time (slot {t})")
    else:
        print("\nNo solution found. The problem may be unsatisfiable.")

if __name__ == "__main__":
    main() 