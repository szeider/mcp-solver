#!/usr/bin/env python3
"""
Test runner for PySAT error handling and validation tests.

This script runs the tests for PySAT error handling and validation components.
"""

import unittest
import sys

# Test modules
from tests.test_pysat_error_handling import TestPySATErrorHandling
from tests.test_pysat_model_validation import TestModelValidation

if __name__ == "__main__":
    print("Running PySAT error handling and validation tests...")
    
    # Create a test suite with all test cases
    test_suite = unittest.TestSuite()
    
    # Add error handling tests
    test_suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestPySATErrorHandling))
    
    # Add model validation tests
    test_suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestModelValidation))
    
    # Run the tests
    result = unittest.TextTestRunner(verbosity=2).run(test_suite)
    
    # Exit with non-zero code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1) 