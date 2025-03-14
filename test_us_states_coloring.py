"""
Test script for finding the smallest subset of US states that requires 4 colors.
Uses the smallest_subset_with_property template from mcp_solver.z3.templates.
"""
from z3 import *
from mcp_solver.z3.templates import smallest_subset_with_property

def test_us_states_coloring():
    # Define all US states and their neighbors
    state_neighbors = {
        'Alabama': ['Florida', 'Georgia', 'Mississippi', 'Tennessee'],
        'Arizona': ['California', 'Colorado', 'Nevada', 'New Mexico', 'Utah'],
        'Arkansas': ['Louisiana', 'Mississippi', 'Missouri', 'Oklahoma', 'Tennessee', 'Texas'],
        'California': ['Arizona', 'Nevada', 'Oregon'],
        'Colorado': ['Arizona', 'Kansas', 'Nebraska', 'New Mexico', 'Oklahoma', 'Utah', 'Wyoming'],
        'Connecticut': ['Massachusetts', 'New York', 'Rhode Island'],
        'Delaware': ['Maryland', 'New Jersey', 'Pennsylvania'],
        'Florida': ['Alabama', 'Georgia'],
        'Georgia': ['Alabama', 'Florida', 'North Carolina', 'South Carolina', 'Tennessee'],
        'Idaho': ['Montana', 'Nevada', 'Oregon', 'Utah', 'Washington', 'Wyoming'],
        'Illinois': ['Indiana', 'Iowa', 'Kentucky', 'Missouri', 'Wisconsin'],
        'Indiana': ['Illinois', 'Kentucky', 'Michigan', 'Ohio'],
        'Iowa': ['Illinois', 'Minnesota', 'Missouri', 'Nebraska', 'South Dakota', 'Wisconsin'],
        'Kansas': ['Colorado', 'Missouri', 'Nebraska', 'Oklahoma'],
        'Kentucky': ['Illinois', 'Indiana', 'Missouri', 'Ohio', 'Tennessee', 'Virginia', 'West Virginia'],
        'Louisiana': ['Arkansas', 'Mississippi', 'Texas'],
        'Maine': ['New Hampshire'],
        'Maryland': ['Delaware', 'Pennsylvania', 'Virginia', 'West Virginia'],
        'Massachusetts': ['Connecticut', 'New Hampshire', 'New York', 'Rhode Island', 'Vermont'],
        'Michigan': ['Indiana', 'Ohio', 'Wisconsin'],
        'Minnesota': ['Iowa', 'North Dakota', 'South Dakota', 'Wisconsin'],
        'Mississippi': ['Alabama', 'Arkansas', 'Louisiana', 'Tennessee'],
        'Missouri': ['Arkansas', 'Illinois', 'Iowa', 'Kansas', 'Kentucky', 'Nebraska', 'Oklahoma', 'Tennessee'],
        'Montana': ['Idaho', 'North Dakota', 'South Dakota', 'Wyoming'],
        'Nebraska': ['Colorado', 'Iowa', 'Kansas', 'Missouri', 'South Dakota', 'Wyoming'],
        'Nevada': ['Arizona', 'California', 'Idaho', 'Oregon', 'Utah'],
        'New Hampshire': ['Maine', 'Massachusetts', 'Vermont'],
        'New Jersey': ['Delaware', 'New York', 'Pennsylvania'],
        'New Mexico': ['Arizona', 'Colorado', 'Oklahoma', 'Texas', 'Utah'],
        'New York': ['Connecticut', 'Massachusetts', 'New Jersey', 'Pennsylvania', 'Vermont'],
        'North Carolina': ['Georgia', 'South Carolina', 'Tennessee', 'Virginia'],
        'North Dakota': ['Minnesota', 'Montana', 'South Dakota'],
        'Ohio': ['Indiana', 'Kentucky', 'Michigan', 'Pennsylvania', 'West Virginia'],
        'Oklahoma': ['Arkansas', 'Colorado', 'Kansas', 'Missouri', 'New Mexico', 'Texas'],
        'Oregon': ['California', 'Idaho', 'Nevada', 'Washington'],
        'Pennsylvania': ['Delaware', 'Maryland', 'New Jersey', 'New York', 'Ohio', 'West Virginia'],
        'Rhode Island': ['Connecticut', 'Massachusetts'],
        'South Carolina': ['Georgia', 'North Carolina'],
        'South Dakota': ['Iowa', 'Minnesota', 'Montana', 'Nebraska', 'North Dakota', 'Wyoming'],
        'Tennessee': ['Alabama', 'Arkansas', 'Georgia', 'Kentucky', 'Mississippi', 'Missouri', 'North Carolina', 'Virginia'],
        'Texas': ['Arkansas', 'Louisiana', 'New Mexico', 'Oklahoma'],
        'Utah': ['Arizona', 'Colorado', 'Idaho', 'Nevada', 'New Mexico', 'Wyoming'],
        'Vermont': ['Massachusetts', 'New Hampshire', 'New York'],
        'Virginia': ['Kentucky', 'Maryland', 'North Carolina', 'Tennessee', 'West Virginia'],
        'Washington': ['Idaho', 'Oregon'],
        'West Virginia': ['Kentucky', 'Maryland', 'Ohio', 'Pennsylvania', 'Virginia'],
        'Wisconsin': ['Illinois', 'Iowa', 'Michigan', 'Minnesota'],
        'Wyoming': ['Colorado', 'Idaho', 'Montana', 'Nebraska', 'South Dakota', 'Utah']
    }
    
    # Convert to a list of state names for easier handling
    all_states = list(state_neighbors.keys())
    
    # Get the states with the most neighbors - they're likely candidates
    state_neighbor_count = {state: len(neighbors) for state, neighbors in state_neighbors.items()}
    sorted_states = sorted(all_states, key=lambda s: state_neighbor_count[s], reverse=True)
    print(f"Top 5 states by number of neighbors:")
    for i, state in enumerate(sorted_states[:5]):
        print(f"{i+1}. {state}: {len(state_neighbors[state])} neighbors")
    
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
    
    # Provide some candidate subsets to speed up the search
    # States with many neighbors are good candidates
    requires_four_colors.candidate_subsets = [
        # Top 4 states by neighbor count
        sorted_states[:4],
        # Missouri and its neighbors
        ['Missouri'] + state_neighbors['Missouri'],
        # Tennessee and its neighbors
        ['Tennessee'] + state_neighbors['Tennessee'],
        # Missouri, Kentucky, Tennessee and their shared neighbors
        list(set(['Missouri', 'Kentucky', 'Tennessee'] + 
             [n for n in state_neighbors['Missouri'] if n in state_neighbors['Kentucky']] +
             [n for n in state_neighbors['Missouri'] if n in state_neighbors['Tennessee']] +
             [n for n in state_neighbors['Kentucky'] if n in state_neighbors['Tennessee']])
        )
    ]
    
    print("\nSearching for the smallest subset of states that requires 4 colors...")
    
    # Find the smallest subset that requires 4 colors
    smallest_subset = smallest_subset_with_property(all_states, requires_four_colors, min_size=4)
    
    if smallest_subset:
        print(f"\nFound smallest subset requiring 4 colors: {smallest_subset}")
        print(f"Size: {len(smallest_subset)} states")
        
        # Draw the subgraph for visualization
        print("\nSubgraph connections:")
        for state in smallest_subset:
            neighbors_in_subset = [n for n in state_neighbors[state] if n in smallest_subset]
            print(f"{state} -> {', '.join(neighbors_in_subset)}")
        
        # Verify that 3 colors are not sufficient
        s = Solver()
        colors = {}
        for state in smallest_subset:
            colors[state] = Int(f"color_{state}")
            s.add(colors[state] >= 1, colors[state] <= 3)
        
        for state in smallest_subset:
            for neighbor in state_neighbors[state]:
                if neighbor in smallest_subset:
                    s.add(colors[state] != colors[neighbor])
        
        print("\nVerifying that 3 colors are not sufficient...")
        result = s.check()
        print(f"Result: {'Unsatisfiable' if result == unsat else 'Satisfiable'}")
        if result == unsat:
            print("Confirmed: This subset requires 4 colors!")
        else:
            print("Error: This subset should require 4 colors but doesn't!")
        
        # Verify minimality
        print("\nVerifying minimality...")
        is_minimal = True
        for state in smallest_subset:
            smaller_subset = [s for s in smallest_subset if s != state]
            if requires_four_colors(smaller_subset):
                print(f"Error: Removing {state} still requires 4 colors!")
                is_minimal = False
        
        if is_minimal:
            print("Confirmed: This subset is minimal!")
    else:
        print("No subset requiring 4 colors was found (this is unexpected).")

if __name__ == "__main__":
    test_us_states_coloring() 