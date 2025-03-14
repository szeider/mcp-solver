"""
Test script for using smallest_subset_with_property through the MCP Solver model.
"""
from z3 import *
from mcp_solver.z3.model_manager import Z3ModelManager
from mcp_solver.z3.environment import execute_z3_code

# Create a model string using the template
model_code = """
from z3 import *
from mcp_solver.z3.templates import smallest_subset_with_property

def build_model():
    # Define a complete graph with 5 nodes (K5)
    # Each state is connected to all other states
    state_neighbors = {
        'A': ['B', 'C', 'D', 'E'],
        'B': ['A', 'C', 'D', 'E'],
        'C': ['A', 'B', 'D', 'E'],
        'D': ['A', 'B', 'C', 'E'],
        'E': ['A', 'B', 'C', 'D']
    }
    
    all_states = list(state_neighbors.keys())
    
    # Property checker: determine if a subset of states requires 4 colors
    def requires_four_colors(subset_states):
        if len(subset_states) < 4:
            return False  # Need at least 4 states to potentially require 4 colors
        
        # Extract the subgraph of just these states
        subgraph = {}
        for state in subset_states:
            # Only include neighbors that are also in our subset
            subgraph[state] = [neighbor for neighbor in state_neighbors[state] if neighbor in subset_states]
        
        # Set up a constraint satisfaction problem
        s = Solver()
        
        # Create variables for the colors of each state (1, 2, or 3)
        colors = {}
        for state in subset_states:
            colors[state] = Int(f"color_{state}")
            # Each state gets a color between 1 and 3
            s.add(colors[state] >= 1, colors[state] <= 3)
        
        # Add constraints: neighboring states must have different colors
        for state, neighbors in subgraph.items():
            for neighbor in neighbors:
                s.add(colors[state] != colors[neighbor])
        
        # If unsatisfiable with 3 colors, then 4 colors are needed
        result = s.check()
        needs_four_colors = result == unsat
        
        return needs_four_colors
    
    # Find the smallest subset that requires 4 colors
    smallest_subset = smallest_subset_with_property(all_states, requires_four_colors, min_size=4)
    
    # Create Z3 variables for the solver
    s = Solver()
    
    # Since we can't directly return Python lists through the Z3 model interface,
    # we'll create string variables to hold the results
    subset_str = ','.join(smallest_subset) if smallest_subset else ""
    subset_size = len(smallest_subset) if smallest_subset else 0
    
    # Create Z3 variables to hold these values
    subset_str_var = String("subset_str")
    subset_size_var = Int("subset_size")
    
    # Constrain the variables to our values
    s.add(subset_str_var == subset_str)
    s.add(subset_size_var == subset_size)
    
    # Check that the solver is satisfied
    if s.check() != sat:
        print("Error: Failed to create satisfiable model")
    
    return s, {
        "subset_str": subset_str_var,
        "subset_size": subset_size_var
    }

# Call the function and export the solution
solver, variables = build_model()
export_solution(solver=solver, variables=variables)
"""

# Execute the code directly using the environment function
print("Testing direct execution through execute_z3_code...")
result = execute_z3_code(model_code, timeout=30.0)
print(f"Status: {result['status']}")
print(f"Output: {result['output']}")
if 'solution' in result and result['solution']:
    solution = result['solution']
    if 'values' in solution:
        values = solution['values']
        if 'subset_str' in values:
            subset_str = values['subset_str']
            subset_size = values['subset_size']
            subset = subset_str.split(',') if subset_str else []
            print(f"Smallest subset: {subset}")
            print(f"Subset size: {subset_size}")
else:
    print("No solution found or error:", result.get('error', 'Unknown error'))

# Test using the model manager
print("\nTesting through Z3ModelManager...")
manager = Z3ModelManager()
import asyncio

async def test_manager():
    await manager.clear_model()
    await manager.add_item(1, model_code)
    solve_result = await manager.solve_model()
    print(f"Solve result status: {solve_result['status']}")
    
    if 'values' in solve_result:
        values = solve_result['values']
        if 'subset_str' in values:
            subset_str = values['subset_str']
            subset_size = values['subset_size']
            subset = subset_str.split(',') if subset_str else []
            print(f"Smallest subset: {subset}")
            print(f"Subset size: {subset_size}")
    else:
        print("Error:", solve_result.get('error', 'Unknown error'))

# Run the async test
try:
    asyncio.run(test_manager())
except Exception as e:
    print(f"Error running test_manager: {e}") 