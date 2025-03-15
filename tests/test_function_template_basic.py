"""
Simple test to verify the function templates work correctly.
"""
from z3 import *
from mcp_solver.z3.templates import constraint_satisfaction_template, optimization_template

def test_constraint_satisfaction():
    """Test the basic constraint satisfaction template."""
    print("Testing constraint_satisfaction_template...")
    
    # Call the template
    solver, variables = constraint_satisfaction_template()
    
    # Check that the solver and variables are properly defined
    assert solver is not None, "Solver should not be None"
    assert 'x' in variables, "Variable 'x' should be defined"
    assert 'y' in variables, "Variable 'y' should be defined"
    
    # Check that the solver has the expected constraints
    # We'll solve it to verify it works
    result = solver.check()
    assert result == sat, "Solver should be satisfiable"
    
    # Get the model and check the values
    model = solver.model()
    x_val = model.eval(variables['x']).as_long()
    y_val = model.eval(variables['y']).as_long()
    
    print(f"Found solution: x = {x_val}, y = {y_val}")
    print("Verifying constraints...")
    assert x_val > 0, "x should be positive"
    assert y_val > 0, "y should be positive"
    assert x_val + y_val <= 10, "x + y should be <= 10"
    
    print("Constraint satisfaction template test passed!")

def test_optimization():
    """Test the optimization template."""
    print("\nTesting optimization_template...")
    
    # Call the template
    optimizer, variables = optimization_template()
    
    # Check that the optimizer and variables are properly defined
    assert optimizer is not None, "Optimizer should not be None"
    assert 'x' in variables, "Variable 'x' should be defined"
    assert 'y' in variables, "Variable 'y' should be defined"
    assert 'objective_value' in variables, "Variable 'objective_value' should be defined"
    
    # Check that the optimizer works
    result = optimizer.check()
    assert result == sat, "Optimizer should find a solution"
    
    # Get the model and check the values
    model = optimizer.model()
    x_val = model.eval(variables['x']).as_long()
    y_val = model.eval(variables['y']).as_long()
    obj_val = model.eval(variables['objective_value']).as_long()
    
    print(f"Found solution: x = {x_val}, y = {y_val}, objective = {obj_val}")
    print("Verifying constraints and objective...")
    assert x_val >= 0, "x should be >= 0"
    assert y_val >= 0, "y should be >= 0"
    assert x_val + y_val <= 10, "x + y should be <= 10"
    assert obj_val == x_val + 2*y_val, "objective should be x + 2y"
    
    print("Optimization template test passed!")

if __name__ == "__main__":
    test_constraint_satisfaction()
    test_optimization()
    print("\nAll tests passed successfully!") 