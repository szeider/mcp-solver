"""
Example demonstrating the quantifier template for Z3.

This example shows how to use the function-based template to solve
a problem involving quantifiers, avoiding common issues like variable scope
problems and import availability.
"""

from z3 import *
from mcp_solver.z3.templates import quantifier_template, array_is_sorted, all_distinct

def build_model():
    """
    Build a model for a permutation problem with quantified constraints.
    
    Problem: Find a permutation of integers 1 to n where:
    - Each number appears exactly once
    - The permutation is almost sorted, with at most one pair out of order
    - The sum of values at even indices is greater than the sum at odd indices
    """
    # [SECTION: PROBLEM PARAMETERS]
    n = 6  # Size of permutation
    
    # [SECTION: VARIABLE DEFINITION]
    # Define the permutation as an array
    perm = Array('perm', IntSort(), IntSort())
    
    # Count of inversions (pairs out of order)
    inversion_count = Int('inversion_count')
    
    # [SECTION: SOLVER CREATION]
    s = Solver()
    
    # [SECTION: DOMAIN CONSTRAINTS]
    # Permutation contains values 1 to n, each exactly once
    for i in range(n):
        s.add(perm[i] >= 1, perm[i] <= n)
    
    # Each value appears exactly once (using template helper)
    s.add(all_distinct(perm, n))
    
    # [SECTION: QUANTIFIER CONSTRAINTS]
    # Calculate inversion count using quantifiers
    i, j = Ints('i j')
    
    # Count pairs that are out of order
    inversion_formula = Sum([
        If(And(i < j, perm[i] > perm[j]), 1, 0)
        for i in range(n) for j in range(i+1, n)
    ])
    
    s.add(inversion_count == inversion_formula)
    
    # Allow at most one inversion
    s.add(inversion_count <= 1)
    
    # Sum constraint: sum at even indices > sum at odd indices
    even_sum = Sum([perm[i] for i in range(0, n, 2)])
    odd_sum = Sum([perm[i] for i in range(1, n, 2)])
    s.add(even_sum > odd_sum)
    
    # [SECTION: VARIABLES TO EXPORT]
    variables = {
        **{f'perm[{i}]': perm[i] for i in range(n)},
        'inversion_count': inversion_count,
        'even_indices_sum': even_sum,
        'odd_indices_sum': odd_sum
    }
    
    return s, variables

# Execute the model
if __name__ == "__main__":
    solver, variables = build_model()
    export_solution(solver=solver, variables=variables) 