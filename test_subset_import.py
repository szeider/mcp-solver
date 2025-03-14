"""
Simple test script to verify the smallest_subset_with_property template is working.
"""
from z3 import *
from mcp_solver.z3.templates import smallest_subset_with_property

# Define a simple list
items = ['A', 'B', 'C', 'D', 'E']

# Define a simple property checker
def has_property(subset):
    # Just check if 'A' and 'B' are both in the subset
    return 'A' in subset and 'B' in subset

# Find the smallest subset with the property
result = smallest_subset_with_property(items, has_property)

print("Test result:", result)
assert result == ['A', 'B'], f"Expected ['A', 'B'], but got {result}"
print("Test passed!") 