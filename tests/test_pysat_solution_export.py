"""
Tests for the PySAT solution export functionality.

This module tests the extraction of values from custom dictionaries in PySAT solutions.
"""

import sys
import os
import unittest
from copy import deepcopy

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mcp_solver.pysat.solution import export_solution, _LAST_SOLUTION

class TestPySATSolutionExport(unittest.TestCase):
    """Tests for the PySAT solution export functionality."""
    
    def setUp(self):
        """Reset the _LAST_SOLUTION global before each test."""
        global _LAST_SOLUTION
        _LAST_SOLUTION = None
    
    def test_basic_extraction(self):
        """Test that values are correctly extracted from custom dictionaries."""
        solution = {
            "satisfiable": True,
            "coloring": {"A": "red", "B": "green"},
            "schedule": {"Monday": "Math", "Tuesday": "Science"}
        }
        
        result = export_solution(solution)
        
        # Check that values were extracted correctly
        self.assertIn("values", result)
        self.assertEqual(result["values"]["A"], "red")
        self.assertEqual(result["values"]["B"], "green")
        self.assertEqual(result["values"]["Monday"], "Math")
        self.assertEqual(result["values"]["Tuesday"], "Science")
    
    def test_key_collision_handling(self):
        """Test that key collisions are handled correctly."""
        solution = {
            "satisfiable": True,
            "employees": {"manager": "Alice", "clerk": "Bob"},
            "rooms": {"manager": "Office A", "meeting": "Room B"}
        }
        
        result = export_solution(solution)
        
        # Non-colliding keys should be directly accessible
        self.assertEqual(result["values"]["clerk"], "Bob")
        self.assertEqual(result["values"]["meeting"], "Room B")
        
        # Colliding keys should be prefixed with their parent dictionary name
        self.assertIn("employees.manager", result["values"])
        self.assertIn("rooms.manager", result["values"])
        self.assertEqual(result["values"]["employees.manager"], "Alice")
        self.assertEqual(result["values"]["rooms.manager"], "Office A")
    
    def test_reserved_keys_not_processed(self):
        """Test that reserved keys are not processed as custom dictionaries."""
        solution = {
            "satisfiable": True,
            "model": {"this": "should not be processed"},
            "status": {"neither": "should this"},
            "data": {"this": "should be processed"}
        }
        
        result = export_solution(solution)
        
        # Check that "data" was processed but not "model" or "status"
        self.assertIn("this", result["values"])
        self.assertEqual(result["values"]["this"], "should be processed")
        self.assertNotIn("should not be processed", result["values"].values())
        self.assertNotIn("should this", result["values"].values())
    
    def test_remove_existing_values_dict(self):
        """Test that any existing values dictionary is removed and recreated."""
        solution = {
            "satisfiable": True,
            "values": {"existing": "value"},  # This should be removed
            "data": {"new": "value"}          # This should be extracted
        }
        
        result = export_solution(solution)
        
        # Check that the old values are gone and the new ones are there
        self.assertIn("values", result)
        self.assertNotIn("existing", result["values"])
        self.assertIn("new", result["values"])
        self.assertEqual(result["values"]["new"], "value")
    
    def test_original_not_modified(self):
        """Test that the original dictionary is not modified."""
        solution = {
            "satisfiable": True,
            "data": {"key": "value"}
        }
        
        original = deepcopy(solution)
        result = export_solution(solution)
        
        # Result should have values dict, original should not be modified
        self.assertIn("values", result)
        self.assertEqual(solution, original)
    
    def test_nested_dictionaries_not_extracted(self):
        """Test that nested dictionaries are not extracted."""
        solution = {
            "satisfiable": True,
            "data": {
                "simple": "value",
                "nested": {"should": "not be extracted"}
            }
        }
        
        result = export_solution(solution)
        
        # Simple values should be extracted
        self.assertEqual(result["values"]["simple"], "value")
        
        # Nested values should not be flattened into values dict
        self.assertNotIn("should", result["values"])
        self.assertNotIn("not be extracted", result["values"].values())


if __name__ == "__main__":
    unittest.main() 