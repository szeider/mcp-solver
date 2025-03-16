"""
Test that all template functions are properly imported and available in the PySAT execution environment.

This test ensures we don't miss import integration issues in the future.
"""

import unittest
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mcp_solver.pysat.environment import execute_pysat_code

class TestEnvironmentImports(unittest.TestCase):
    """Test that template functions are properly available in the PySAT execution environment."""
    
    def test_cardinality_templates_execution(self):
        """Test that cardinality template functions can be executed."""
        code = """
# Use cardinality template functions
from pysat.formula import CNF
from pysat.solvers import Glucose3

# Create CNF formula using cardinality templates
formula = CNF()

# Use at_most_one constraint with 3 variables
variables = [1, 2, 3]
for clause in at_most_one(variables):
    formula.append(clause)

# Use implies constraint
for clause in implies(1, 2):
    formula.append(clause)

# Solve the formula
solver = Glucose3(bootstrap_with=formula)
is_sat = solver.solve()

print(f"Solution satisfiable: {is_sat}")
solver.delete()
"""
        result = execute_pysat_code(code)
        self.assertTrue(result["success"], f"Cardinality template code failed: {result.get('error')}")
        self.assertIn("Solution satisfiable", result["output"], "Expected output missing")

if __name__ == "__main__":
    unittest.main() 