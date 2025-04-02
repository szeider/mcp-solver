"""
Test file for property verification using the Z3 solution module.

This file demonstrates how to use the updated solution module for property verification
where counterexamples should be treated as valid solutions.
"""

import z3
from mcp_solver.z3.solution import export_solution

def test_valid_property():
    """Test verifying a valid property (no counterexample)."""
    print("\nTesting valid property verification:")
    
    # Create solver and variables
    solver = z3.Solver()
    x = z3.Int('x')
    
    # Add constraints to verify the property: "For all x, if x > 0 then x > -5"
    # We negate the property to search for a counterexample
    solver.add(z3.Not(z3.Implies(x > 0, x > -5)))
    
    # Check satisfiability (should be unsat because the property is valid)
    result = solver.check()
    print(f"Solver result: {result}")
    
    # Export solution as a property verification
    variables = {'x': x}
    solution = export_solution(
        solver=solver, 
        variables=variables, 
        is_property_verification=True
    )
    
    # Print the solution
    print("Property verification solution:")
    print(f"  satisfiable: {solution['satisfiable']}")
    print(f"  status: {solution['status']}")
    print(f"  values: {solution['values']}")
    print(f"  output: {solution['output']}")
    
    return solution

def test_invalid_property():
    """Test verifying an invalid property (counterexample exists)."""
    print("\nTesting invalid property verification:")
    
    # Create solver and variables
    solver = z3.Solver()
    x = z3.Int('x')
    
    # Add constraints to verify the property: "For all x, if x > 0 then x > 10"
    # We negate the property to search for a counterexample
    solver.add(z3.Not(z3.Implies(x > 0, x > 10)))
    
    # Check satisfiability (should be sat because we can find a counterexample, e.g., x=5)
    result = solver.check()
    print(f"Solver result: {result}")
    
    # Export solution as a property verification
    variables = {'x': x}
    solution = export_solution(
        solver=solver, 
        variables=variables, 
        is_property_verification=True
    )
    
    # Print the solution
    print("Property verification solution:")
    print(f"  satisfiable: {solution['satisfiable']}")
    print(f"  status: {solution['status']}")
    print(f"  values: {solution['values']}")
    print(f"  output: {solution['output']}")
    
    return solution

def test_property_override():
    """Test explicitly overriding property verification result."""
    print("\nTesting explicit property verification override:")
    
    # Create variables
    variables = {'x': 'dummy_var'}
    
    # Export solution with explicit property_verified=True
    solution1 = export_solution(
        variables=variables, 
        is_property_verification=True, 
        property_verified=True
    )
    
    # Print the solution
    print("Solution with property_verified=True:")
    print(f"  satisfiable: {solution1['satisfiable']}")
    print(f"  status: {solution1['status']}")
    print(f"  values: {solution1['values']}")
    print(f"  output: {solution1['output']}")
    
    # Export solution with explicit property_verified=False
    solution2 = export_solution(
        variables=variables, 
        is_property_verification=True, 
        property_verified=False
    )
    
    # Print the solution
    print("Solution with property_verified=False:")
    print(f"  satisfiable: {solution2['satisfiable']}")
    print(f"  status: {solution2['status']}")
    print(f"  values: {solution2['values']}")
    print(f"  output: {solution2['output']}")
    
    return solution2

def run_all_tests():
    """Run all property verification tests."""
    test_valid_property()
    test_invalid_property()
    test_property_override()
    print("\nAll property verification tests completed successfully!")

if __name__ == "__main__":
    run_all_tests() 