"""
Example demonstrating the optimization template for Z3.

This example shows how to use the function-based template to solve
an optimization problem, avoiding common issues like variable scope
problems and import availability.
"""

from z3 import *
from mcp_solver.z3.templates import optimization_template

def build_model():
    """
    Build a model for a simple production optimization problem.
    
    Problem: A factory produces two products, A and B:
    - Each unit of A requires 2 hours of labor and yields $5 profit
    - Each unit of B requires 3 hours of labor and yields $7 profit
    - The factory has 40 hours of labor available
    - Due to demand constraints, we must produce at least 5 units of A
    - Due to material constraints, we can produce at most a total of 15 units
    - How many units of each product should be produced to maximize profit?
    """
    # [SECTION: VARIABLE DEFINITION]
    # Define production variables
    a_units = Int('a_units')  # Number of units of product A to produce
    b_units = Int('b_units')  # Number of units of product B to produce
    
    # [SECTION: OPTIMIZER CREATION]
    opt = Optimize()
    
    # [SECTION: DOMAIN CONSTRAINTS]
    # Non-negative production
    opt.add(a_units >= 0)
    opt.add(b_units >= 0)
    
    # [SECTION: CORE CONSTRAINTS]
    # Labor constraint: 2 hours per unit of A, 3 hours per unit of B, 40 hours total
    opt.add(2 * a_units + 3 * b_units <= 40)
    
    # Must produce at least 5 units of A due to demand
    opt.add(a_units >= 5)
    
    # Can produce at most 15 units total due to materials
    opt.add(a_units + b_units <= 15)
    
    # [SECTION: OBJECTIVE FUNCTION]
    # Profit calculation: $5 per unit of A, $7 per unit of B
    profit = 5 * a_units + 7 * b_units
    
    # Maximize profit
    opt.maximize(profit)
    
    # [SECTION: VARIABLES TO EXPORT]
    variables = {
        'a_units': a_units,
        'b_units': b_units,
        'total_profit': profit,
        'labor_hours_used': 2 * a_units + 3 * b_units
    }
    
    return opt, variables

# Execute the model
if __name__ == "__main__":
    solver, variables = build_model()
    export_solution(solver=solver, variables=variables) 