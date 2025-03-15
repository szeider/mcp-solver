"""
Test script for finding a small subset of states that requires 4 colors.
Uses a complete graph of 5 nodes (K5) which is known to require 4 colors.
"""
from z3 import *
from mcp_solver.z3.templates import smallest_subset_with_property

def test_map_coloring():
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
        """
        Check if a subset of states requires 4 colors for a proper coloring.
        Returns True if 4 colors are needed, False if 3 or fewer colors are sufficient.
        """
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
    
    print(f"Smallest subset requiring 4 colors: {smallest_subset}")
    
    # Verify the result
    if smallest_subset:
        assert requires_four_colors(smallest_subset), "The found subset should require 4 colors!"
        print("Verified: Subset requires 4 colors")
        
        # Check for minimality
        for state in smallest_subset:
            smaller_subset = [s for s in smallest_subset if s != state]
            if requires_four_colors(smaller_subset):
                print(f"Error: Removing {state} still requires 4 colors!")
                assert False, "Subset is not minimal!"
        
        print("Verified: Subset is minimal")
    else:
        print("No subset requiring 4 colors was found.")

if __name__ == "__main__":
    test_map_coloring() 