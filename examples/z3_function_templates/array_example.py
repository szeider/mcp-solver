"""
Example demonstrating the array template for Z3.

This example shows how to use the function-based template to solve
a problem involving arrays, avoiding common issues like variable scope
problems and import availability.
"""

from z3 import *
from mcp_solver.z3.templates import array_template

def build_model():
    """
    Build a model for a simple sequence generation problem.
    
    Problem: Generate a sequence of 5 integers where:
    - Each value is between 1 and 10
    - The sequence is strictly increasing
    - The sum of all values is exactly 30
    - The first value is odd and the last value is even
    """
    # [SECTION: PROBLEM SIZE]
    n = 5  # Size of the sequence
    
    # [SECTION: VARIABLE DEFINITION]
    # Define the sequence as an array
    seq = Array('seq', IntSort(), IntSort())
    
    # [SECTION: SOLVER CREATION]
    s = Solver()
    
    # [SECTION: ARRAY INITIALIZATION]
    # Domain constraints: values between 1 and 10
    for i in range(n):
        s.add(seq[i] >= 1, seq[i] <= 10)
    
    # [SECTION: CONSTRAINTS]
    # Sequence is strictly increasing
    for i in range(n-1):
        s.add(seq[i] < seq[i+1])
    
    # Sum constraint: all values sum to exactly 30
    sum_constraint = seq[0]
    for i in range(1, n):
        sum_constraint = sum_constraint + seq[i]
    s.add(sum_constraint == 30)
    
    # First value is odd, last value is even
    s.add(seq[0] % 2 == 1)  # odd
    s.add(seq[n-1] % 2 == 0)  # even
    
    # [SECTION: VARIABLES TO EXPORT]
    variables = {
        **{f'seq[{i}]': seq[i] for i in range(n)},
        'sequence_sum': sum_constraint
    }
    
    return s, variables

# Execute the model
if __name__ == "__main__":
    solver, variables = build_model()
    export_solution(solver=solver, variables=variables) 