#!/usr/bin/env python3
"""
Test file for PySAT error handling functionality.

This file contains tests for the error handling improvements in the PySAT
module, including validation of variables, formula structure, and error
message generation.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from mcp_solver.pysat.error_handling import (
    PySATError,
    pysat_error_handler,
    validate_variables,
    validate_formula,
    format_solution_error
)

class TestPySATErrorHandling(unittest.TestCase):
    """Test suite for PySAT error handling functionality."""
    
    def test_pysat_error_class(self):
        """Test PySATError class with different initialization parameters."""
        # Test basic error
        error = PySATError("Basic error message")
        self.assertEqual(str(error), "Basic error message")
        self.assertIsNone(error.original_error)
        self.assertEqual(error.context, {})
        
        # Test with original error
        original = ValueError("Original message")
        error = PySATError("Enhanced message", original_error=original)
        self.assertIn("Enhanced message", str(error))
        self.assertIn("Original error (ValueError): Original message", str(error))
        self.assertEqual(error.original_error, original)
        
        # Test with context
        context = {"variables": 10, "clauses": 20}
        error = PySATError("Context message", context=context)
        self.assertIn("Context message", str(error))
        self.assertIn("Context:", str(error))
        self.assertIn("variables: 10", str(error))
        self.assertIn("clauses: 20", str(error))
        self.assertEqual(error.context, context)
        
        # Test with both original error and context
        error = PySATError("Full message", original_error=original, context=context)
        self.assertIn("Full message", str(error))
        self.assertIn("Context:", str(error))
        self.assertIn("Original error (ValueError): Original message", str(error))
    
    def test_error_handler_decorator(self):
        """Test the pysat_error_handler decorator."""
        
        # Test function that succeeds
        @pysat_error_handler
        def success_func():
            return "success"
        
        self.assertEqual(success_func(), "success")
        
        # Test function that raises a known error
        @pysat_error_handler
        def value_error_func():
            raise ValueError("variable identifier should be a non-zero integer")
        
        with self.assertRaises(PySATError) as context:
            value_error_func()
        
        error_message = str(context.exception)
        self.assertIn("Variable IDs must be non-zero integers", error_message)
        
        # Test function that raises an unknown error
        @pysat_error_handler
        def unknown_error_func():
            raise RuntimeError("some random error")
        
        with self.assertRaises(PySATError) as context:
            unknown_error_func()
        
        error_message = str(context.exception)
        self.assertIn("Error in PySAT operation: some random error", error_message)
        
        # Test function with formula context
        @pysat_error_handler
        def context_error_func(formula):
            raise ValueError("Test error")
        
        # Create a mock formula
        mock_formula = MagicMock()
        mock_formula.nof_vars.return_value = 10
        mock_formula.nof_clauses.return_value = 20
        mock_formula.clauses = [[1, 2], [3, -4]]
        
        with self.assertRaises(PySATError) as context:
            context_error_func(mock_formula)
        
        error_message = str(context.exception)
        self.assertIn("Context:", error_message)
        self.assertIn("variables: 10", error_message)
        self.assertIn("clauses: 20", error_message)
    
    def test_variable_validation(self):
        """Test the variable validation function."""
        
        # Test empty dictionary
        errors = validate_variables({})
        self.assertEqual(len(errors), 1)
        self.assertIn("Variable dictionary is empty", errors[0])
        
        # Test valid variables
        variables = {"a": 1, "b": 2, "c": 3}
        errors = validate_variables(variables)
        self.assertEqual(errors, [])
        
        # Test non-integer variable
        variables = {"a": 1, "b": "2", "c": 3}
        errors = validate_variables(variables)
        self.assertEqual(len(errors), 1)
        self.assertIn("non-integer ID", errors[0])
        self.assertIn("'b'", errors[0])
        
        # Test non-positive variable
        variables = {"a": 1, "b": 0, "c": -3}
        errors = validate_variables(variables)
        self.assertEqual(len(errors), 2)
        self.assertTrue(any("'b' has non-positive ID: 0" in e for e in errors))
        self.assertTrue(any("'c' has non-positive ID: -3" in e for e in errors))
        
        # Test duplicate variable IDs
        variables = {"a": 1, "b": 2, "c": 1}
        errors = validate_variables(variables)
        self.assertEqual(len(errors), 1)
        self.assertIn("Multiple variables (a, c) share the same ID: 1", errors[0])
    
    def test_formula_validation(self):
        """Test the formula validation function."""
        
        # Test invalid formula object
        errors = validate_formula("not a formula")
        self.assertEqual(len(errors), 1)
        self.assertIn("not appear to be a valid PySAT formula", errors[0])
        
        # Test empty formula
        mock_formula = MagicMock()
        mock_formula.clauses = []
        errors = validate_formula(mock_formula)
        self.assertEqual(len(errors), 1)
        self.assertIn("Formula has no clauses", errors[0])
        
        # Test valid formula
        mock_formula = MagicMock()
        mock_formula.clauses = [[1, 2], [3, -4]]
        mock_formula.nof_vars.return_value = 4
        errors = validate_formula(mock_formula)
        self.assertEqual(errors, [])
        
        # Test formula with empty clause
        mock_formula = MagicMock()
        mock_formula.clauses = [[1, 2], [], [3, -4]]
        mock_formula.nof_vars.return_value = 4
        errors = validate_formula(mock_formula)
        self.assertEqual(len(errors), 1)
        self.assertIn("Clause 2 is empty", errors[0])
        
        # Test formula with non-integer literal
        mock_formula = MagicMock()
        mock_formula.clauses = [[1, 2], [3, "not an int"]]
        mock_formula.nof_vars.return_value = 4
        errors = validate_formula(mock_formula)
        self.assertEqual(len(errors), 1)
        self.assertIn("not an integer", errors[0])
        
        # Test formula with zero literal
        mock_formula = MagicMock()
        mock_formula.clauses = [[1, 2], [3, 0, -4]]
        mock_formula.nof_vars.return_value = 4
        errors = validate_formula(mock_formula)
        self.assertEqual(len(errors), 1)
        self.assertIn("literal 2 is zero", errors[0])
        
        # Test formula with inconsistent variable count
        mock_formula = MagicMock()
        mock_formula.clauses = [[1, 2], [3, -4], [5, 6]]
        mock_formula.nof_vars.return_value = 4  # But we use 6
        errors = validate_formula(mock_formula)
        self.assertEqual(len(errors), 1)
        self.assertIn("Formula reports 4 variables but clauses reference variable 6", errors[0])
    
    def test_format_solution_error(self):
        """Test the solution error formatting function."""
        
        # Test with standard exception
        standard_error = ValueError("standard error")
        result = format_solution_error(standard_error)
        self.assertEqual(result["satisfiable"], False)
        self.assertEqual(result["error_type"], "ValueError")
        self.assertEqual(result["error_message"], "standard error")
        self.assertNotIn("error_context", result)
        
        # Test with PySATError without context
        pysat_error = PySATError("pysat error")
        result = format_solution_error(pysat_error)
        self.assertEqual(result["satisfiable"], False)
        self.assertEqual(result["error_type"], "PySATError")
        self.assertEqual(result["error_message"], "pysat error")
        self.assertNotIn("error_context", result)
        
        # Test with PySATError with context
        pysat_error = PySATError("pysat error with context", context={"key": "value"})
        result = format_solution_error(pysat_error)
        self.assertEqual(result["satisfiable"], False)
        self.assertEqual(result["error_type"], "PySATError")
        self.assertEqual(result["error_message"], str(pysat_error))
        self.assertIn("error_context", result)
        self.assertEqual(result["error_context"]["key"], "value")

if __name__ == "__main__":
    unittest.main() 