#!/usr/bin/env python3
"""
Tests for the global MaxSAT functionality.
"""

import os
import sys
import unittest
import json
import time
import asyncio
import random
from typing import Dict, Any, List

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mcp_solver.pysat.global_maxsat import (
    initialize_maxsat, add_hard_clause, add_soft_clause, 
    solve_maxsat, get_current_wcnf, add_at_most_k_soft,
    university_course_scheduling_example
)
from mcp_solver.pysat.model_manager import PySATModelManager
from mcp_solver.pysat.environment import execute_pysat_code
from mcp_solver.pysat.solution import export_solution
from mcp_solver.pysat.templates.cardinality_templates import (
    at_most_k, at_least_k
)

class TestGlobalMaxSAT(unittest.TestCase):
    """Test cases for the global MaxSAT functionality."""

    def setUp(self):
        """Set up the test environment."""
        self.model_manager = PySATModelManager(lite_mode=True)
        # Use a more modern way to get an event loop
        try:
            self.event_loop = asyncio.get_running_loop()
        except RuntimeError:
            self.event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.event_loop)

    def test_robust_cardinality_constraints(self):
        """Test the robust cardinality constraint implementation."""
        # Test at_most_k with different inputs
        # Case 1: Edge case where k >= len(variables)
        clauses = at_most_k([1, 2, 3], 3)
        self.assertEqual(len(clauses), 0, "Should return empty list when k >= len(variables)")
        
        # Case 2: Small set using direct encoding
        clauses = at_most_k([1, 2, 3], 1)
        # Should generate clauses for each pair: [-1, -2], [-1, -3], [-2, -3]
        self.assertEqual(len(clauses), 3, "Should generate correct number of clauses for small set")
        
        # Case 3: Verify at_least_k (using De Morgan's law)
        clauses = at_least_k([1, 2, 3], 2)
        # Should be equivalent to at_most_k([-1, -2, -3], 1)
        self.assertEqual(len(clauses), 3, "Should generate correct number of clauses for at_least_k")

    def test_party_planning_with_cardinality(self):
        """Test party planning problem with cardinality constraints."""
        code = """
# Initialize a new MaxSAT problem
initialize_maxsat()

# Variables 1-10 represent dishes (1: pasta, 2: salad, 3: steak, etc.)
dishes = list(range(1, 11))
vegetarian_dishes = [2, 4, 7]  # salad, soup, roasted vegetables
spicy_dishes = [5, 8]  # curry, spicy tacos
pasta_and_bread = [1, 9]  # pasta, garlic bread

# Hard constraints
# 1. At most 5 dishes - unpack clauses correctly
clauses = at_most_k(dishes, 5)
for clause in clauses:
    add_hard_clause(clause)

# 2. At least 1 vegetarian dish - unpack clauses correctly
clauses = at_least_k(vegetarian_dishes, 1)
for clause in clauses:
    add_hard_clause(clause)

# 3. If any spicy dish is served, at least one non-spicy dish must be served
for spicy_dish in spicy_dishes:
    non_spicy = [dish for dish in dishes if dish not in spicy_dishes]
    add_hard_clause([-spicy_dish] + non_spicy)

# 4. Can't serve both pasta and garlic bread
add_hard_clause([-pasta_and_bread[0], -pasta_and_bread[1]])

# Soft constraints (preferences)
# Guest 1 likes steak, curry, and chocolate cake
add_soft_clause([3], weight=2)  # steak
add_soft_clause([5], weight=1)  # curry
add_soft_clause([10], weight=2)  # chocolate cake

# Guest 2 is vegetarian, likes salad, soup, and roasted vegetables
add_soft_clause([2], weight=3)  # salad
add_soft_clause([4], weight=2)  # soup
add_soft_clause([7], weight=1)  # roasted vegetables

# Guest 3 likes pasta, garlic bread, and chicken
add_soft_clause([1], weight=2)  # pasta
add_soft_clause([9], weight=1)  # garlic bread
add_soft_clause([6], weight=2)  # chicken

# Solve the problem
model, cost = solve_maxsat(timeout=5.0)

# Create dish mapping
dish_names = {
    1: "pasta", 2: "salad", 3: "steak", 4: "soup", 5: "curry",
    6: "chicken", 7: "roasted_vegetables", 8: "spicy_tacos",
    9: "garlic_bread", 10: "chocolate_cake"
}

# Extract solution
selected_dishes = [dish_names[d] for d in range(1, 11) if d in model] if model else []

if model is not None:
    result = {
        "satisfiable": True,
        "model": model,
        "objective": cost,
        "solution": {
            "selected_dishes": selected_dishes,
            "cost": cost
        }
    }
    export_solution(result)
else:
    export_solution({"satisfiable": False})
"""
        result = execute_pysat_code(code, timeout=5.0)
        print("Party planning result:", json.dumps(result, indent=2))
        
        # Verify the party planning solution
        self.assertTrue(result["success"], f"Error: {result.get('error')}")
        
        # Only check constraints if we got a solution
        if result.get("solution") is not None and result["solution"].get("satisfiable", False):
            if "solution" in result["solution"]:
                selected_dishes = result["solution"]["solution"]["selected_dishes"]
                self.assertLessEqual(len(selected_dishes), 5, "Should select at most 5 dishes")
                
                # Check if at least one vegetarian dish is selected
                vegetarian_dishes = ["salad", "soup", "roasted_vegetables"]
                has_vegetarian = any(dish in vegetarian_dishes for dish in selected_dishes)
                self.assertTrue(has_vegetarian, "Should include at least one vegetarian dish")
                
                # Check pasta and garlic bread constraint
                has_pasta = "pasta" in selected_dishes
                has_garlic_bread = "garlic_bread" in selected_dishes
                self.assertFalse(has_pasta and has_garlic_bread, "Cannot serve both pasta and garlic bread")
                
                # Check spicy dish constraint
                spicy_dishes = ["curry", "spicy_tacos"]
                has_spicy = any(dish in spicy_dishes for dish in selected_dishes)
                if has_spicy:
                    non_spicy = [d for d in selected_dishes if d not in spicy_dishes]
                    self.assertGreater(len(non_spicy), 0, "If serving spicy, must have non-spicy dish")

    def test_maxsat_timeout_handling(self):
        """Test that timeout handling in solve_maxsat works correctly."""
        code = """
# Create a MaxSAT problem that should finish quickly
initialize_maxsat()

# Import needed modules
import random
import time

# Create 15 variables
variables = list(range(1, 16))

# Add hard constraints that require at most 10 variables to be true
clauses = at_most_k(variables, 10)
for clause in clauses:
    add_hard_clause(clause)

# Add some soft clauses 
random.seed(42)  # Fixed seed for reproducibility
for _ in range(20):
    # Create random soft clauses
    size = random.randint(1, 3)
    clause = random.sample(variables, size)
    add_soft_clause(clause, weight=random.randint(1, 5))

# Solve with a specific timeout (2 second should be enough)
start_time = time.time()
model, cost = solve_maxsat(timeout=2.0)
end_time = time.time()

# Check how long it actually took
elapsed_time = end_time - start_time

if model is not None:
    result = {
        "satisfiable": True,
        "model": model,
        "objective": cost,
        "solution_time": elapsed_time
    }
    export_solution(result)
else:
    export_solution({
        "satisfiable": False,
        "solution_time": elapsed_time
    })
"""
        result = execute_pysat_code(code, timeout=5.0)
        print("Timeout test result:", json.dumps(result, indent=2))
        
        # The test should complete within our timeout
        self.assertTrue(result["success"], f"Error: {result.get('error')}")
        
        # The solution time should be reported if we have a solution
        if result.get("solution") is not None:
            self.assertIn("solution_time", result["solution"])


if __name__ == "__main__":
    unittest.main() 