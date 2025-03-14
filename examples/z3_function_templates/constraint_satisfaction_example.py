"""
Example demonstrating the constraint satisfaction template for Z3.

This example shows how to use the function-based template to solve
a simple constraint satisfaction problem, avoiding common issues like
variable scope problems and import availability.
"""

from z3 import *
from mcp_solver.z3.templates import constraint_satisfaction_template

def build_model():
    """
    Build a model for a simple room allocation problem.
    
    Problem: Assign rooms to people where:
    - Room numbers are between 1 and 10
    - Alice and Bob must be in adjacent rooms
    - Charlie must be in an even-numbered room
    - No two people can share a room
    """
    # [SECTION: VARIABLE DEFINITION]
    # Define room assignment variables
    alice_room = Int('alice_room')
    bob_room = Int('bob_room')
    charlie_room = Int('charlie_room')
    
    # [SECTION: SOLVER CREATION]
    s = Solver()
    
    # [SECTION: DOMAIN CONSTRAINTS]
    # Room numbers are between 1 and 10
    for room in [alice_room, bob_room, charlie_room]:
        s.add(room >= 1, room <= 10)
    
    # [SECTION: CORE CONSTRAINTS]
    # Alice and Bob must be in adjacent rooms
    s.add(Or(alice_room == bob_room + 1, alice_room == bob_room - 1))
    
    # Charlie must be in an even-numbered room
    s.add(charlie_room % 2 == 0)
    
    # No two people can share a room
    s.add(Distinct([alice_room, bob_room, charlie_room]))
    
    # [SECTION: VARIABLES TO EXPORT]
    variables = {
        'alice_room': alice_room,
        'bob_room': bob_room,
        'charlie_room': charlie_room
    }
    
    return s, variables

# Execute the model
if __name__ == "__main__":
    solver, variables = build_model()
    export_solution(solver=solver, variables=variables) 