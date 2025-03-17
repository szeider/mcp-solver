#!/usr/bin/env python3
"""
Test runner for the PySAT error handling improvements.

This script runs the tests for the error handling module to verify it works correctly.
"""

import os
import sys
import unittest
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

def run_tests():
    """Run the error handling tests."""
    print("\n===== Running PySAT Error Handling Tests =====\n")
    
    # Run the test file
    test_module = "tests.test_pysat_error_handling"
    
    # Use unittest discovery
    test_suite = unittest.defaultTestLoader.loadTestsFromName(test_module)
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Return True if all tests passed
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 