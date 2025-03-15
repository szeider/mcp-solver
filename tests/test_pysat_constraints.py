"""
Test PySAT constraint helper functions.

This file tests the functionality of the constraint helper functions
implemented in the mcp_solver.pysat.constraints module.
"""

import sys
import os
import unittest
from typing import List, Set

# Add the src directory to the path to import mcp_solver
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import the helper functions
from mcp_solver.pysat.constraints import (
    at_most_k, at_least_k, exactly_k,
    at_most_one, exactly_one,
    implies, mutually_exclusive, if_then_else
)

# Import PySAT dependencies
from pysat.formula import CNF
from pysat.solvers import Glucose3

class TestPySATConstraints(unittest.TestCase):
    """Test cases for PySAT constraint helper functions."""
    
    def check_solutions(self, formula, variables, expected_counts):
        """
        Helper to check the solutions against expected counts.
        
        Args:
            formula: CNF formula with clauses
            variables: List of variables to check
            expected_counts: Dict mapping number of true vars to expected satisfiability
        """
        solver = Glucose3()
        solver.append_formula(formula)
        
        for num_true, expected_satisfiable in expected_counts.items():
            # Force exactly num_true variables to be true
            assumptions = []
            for i, var in enumerate(variables):
                if i < num_true:
                    assumptions.append(var)      # Force true
                else:
                    assumptions.append(-var)     # Force false
            
            result = solver.solve(assumptions=assumptions)
            self.assertEqual(result, expected_satisfiable, 
                            f"Expected {expected_satisfiable} when {num_true} variables are true")
        
        solver.delete()
    
    def test_at_most_k(self):
        """Test the at_most_k function."""
        variables = [1, 2, 3, 4, 5]
        
        # Test at_most_0
        formula = CNF()
        formula.extend(at_most_k(variables, 0))
        expected = {0: True, 1: False, 2: False, 3: False, 4: False, 5: False}
        self.check_solutions(formula, variables, expected)
        
        # Test at_most_1
        formula = CNF()
        formula.extend(at_most_k(variables, 1))
        expected = {0: True, 1: True, 2: False, 3: False, 4: False, 5: False}
        self.check_solutions(formula, variables, expected)
        
        # Test at_most_2
        formula = CNF()
        formula.extend(at_most_k(variables, 2))
        expected = {0: True, 1: True, 2: True, 3: False, 4: False, 5: False}
        self.check_solutions(formula, variables, expected)
        
        # Test at_most_5 (all variables can be true)
        formula = CNF()
        formula.extend(at_most_k(variables, 5))
        expected = {0: True, 1: True, 2: True, 3: True, 4: True, 5: True}
        self.check_solutions(formula, variables, expected)
        
        # Test at_most_10 (exceeds variables, so all can be true)
        formula = CNF()
        formula.extend(at_most_k(variables, 10))
        expected = {0: True, 1: True, 2: True, 3: True, 4: True, 5: True}
        self.check_solutions(formula, variables, expected)
    
    def test_at_least_k(self):
        """Test the at_least_k function."""
        variables = [1, 2, 3, 4, 5]
        
        # Test at_least_0 (always satisfied)
        formula = CNF()
        formula.extend(at_least_k(variables, 0))
        expected = {0: True, 1: True, 2: True, 3: True, 4: True, 5: True}
        self.check_solutions(formula, variables, expected)
        
        # Test at_least_1
        formula = CNF()
        formula.extend(at_least_k(variables, 1))
        expected = {0: False, 1: True, 2: True, 3: True, 4: True, 5: True}
        self.check_solutions(formula, variables, expected)
        
        # Test at_least_3
        formula = CNF()
        formula.extend(at_least_k(variables, 3))
        expected = {0: False, 1: False, 2: False, 3: True, 4: True, 5: True}
        self.check_solutions(formula, variables, expected)
        
        # Test at_least_5 (all variables must be true)
        formula = CNF()
        formula.extend(at_least_k(variables, 5))
        expected = {0: False, 1: False, 2: False, 3: False, 4: False, 5: True}
        self.check_solutions(formula, variables, expected)
    
    def test_exactly_k(self):
        """Test the exactly_k function."""
        variables = [1, 2, 3, 4, 5]
        
        # Test exactly_0
        formula = CNF()
        formula.extend(exactly_k(variables, 0))
        expected = {0: True, 1: False, 2: False, 3: False, 4: False, 5: False}
        self.check_solutions(formula, variables, expected)
        
        # Test exactly_1
        formula = CNF()
        formula.extend(exactly_k(variables, 1))
        expected = {0: False, 1: True, 2: False, 3: False, 4: False, 5: False}
        self.check_solutions(formula, variables, expected)
        
        # Test exactly_3
        formula = CNF()
        formula.extend(exactly_k(variables, 3))
        expected = {0: False, 1: False, 2: False, 3: True, 4: False, 5: False}
        self.check_solutions(formula, variables, expected)
        
        # Test exactly_5 (all variables must be true)
        formula = CNF()
        formula.extend(exactly_k(variables, 5))
        expected = {0: False, 1: False, 2: False, 3: False, 4: False, 5: True}
        self.check_solutions(formula, variables, expected)
        
        # Test invalid k value (greater than number of variables)
        invalid_clauses = exactly_k(variables, 6)
        self.assertEqual(invalid_clauses, [[]], "Should return [[]] for k > len(variables)")
        
        # Test invalid k value (negative)
        invalid_clauses = exactly_k(variables, -1)
        self.assertEqual(invalid_clauses, [[]], "Should return [[]] for negative k")
    
    def test_at_most_one(self):
        """Test the at_most_one function."""
        variables = [1, 2, 3, 4, 5]
        
        formula = CNF()
        formula.extend(at_most_one(variables))
        expected = {0: True, 1: True, 2: False, 3: False, 4: False, 5: False}
        self.check_solutions(formula, variables, expected)
    
    def test_exactly_one(self):
        """Test the exactly_one function."""
        variables = [1, 2, 3, 4, 5]
        
        formula = CNF()
        formula.extend(exactly_one(variables))
        expected = {0: False, 1: True, 2: False, 3: False, 4: False, 5: False}
        self.check_solutions(formula, variables, expected)
    
    def test_implies(self):
        """Test the implies function."""
        # Test a -> b
        formula = CNF()
        formula.extend(implies(1, 2))
        
        # Check all 4 possible assignments
        solver = Glucose3()
        solver.append_formula(formula)
        
        # a=False, b=False: should be satisfiable
        self.assertTrue(solver.solve(assumptions=[-1, -2]))
        
        # a=False, b=True: should be satisfiable
        self.assertTrue(solver.solve(assumptions=[-1, 2]))
        
        # a=True, b=False: should be unsatisfiable
        self.assertFalse(solver.solve(assumptions=[1, -2]))
        
        # a=True, b=True: should be satisfiable
        self.assertTrue(solver.solve(assumptions=[1, 2]))
        
        solver.delete()
    
    def test_mutually_exclusive(self):
        """Test the mutually_exclusive function."""
        variables = [1, 2, 3, 4, 5]
        
        formula = CNF()
        formula.extend(mutually_exclusive(variables))
        expected = {0: True, 1: True, 2: False, 3: False, 4: False, 5: False}
        self.check_solutions(formula, variables, expected)
    
    def test_if_then_else(self):
        """Test the if_then_else function."""
        # Test condition ? then : else
        condition = 1
        then_var = 2
        else_var = 3
        
        formula = CNF()
        formula.extend(if_then_else(condition, then_var, else_var))
        
        solver = Glucose3()
        solver.append_formula(formula)
        
        # condition=True: then_var should be True
        self.assertFalse(solver.solve(assumptions=[condition, -then_var]))
        
        # condition=False: else_var should be True
        self.assertFalse(solver.solve(assumptions=[-condition, -else_var]))
        
        # condition=True: then_var=True and else_var can be anything
        self.assertTrue(solver.solve(assumptions=[condition, then_var, else_var]))
        self.assertTrue(solver.solve(assumptions=[condition, then_var, -else_var]))
        
        # condition=False: else_var=True and then_var can be anything
        self.assertTrue(solver.solve(assumptions=[-condition, then_var, else_var]))
        self.assertTrue(solver.solve(assumptions=[-condition, -then_var, else_var]))
        
        solver.delete()


if __name__ == "__main__":
    unittest.main() 