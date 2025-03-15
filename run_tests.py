#!/usr/bin/env python3
"""
Test runner for MCP Solver.

This script runs all tests in the tests directory and reports the results.
"""

import os
import sys
import glob
import subprocess
import time

def run_test(test_path):
    """Run a single test file and return whether it passed."""
    print(f"\n\033[1m=== Running {os.path.basename(test_path)} ===\033[0m")
    start_time = time.time()
    
    # Run the test with uv run python
    result = subprocess.run(
        ["uv", "run", "python", test_path],
        capture_output=True,
        text=True
    )
    
    duration = time.time() - start_time
    
    # Print test output
    print(result.stdout)
    if result.stderr:
        print(f"\033[91mERROR OUTPUT:\033[0m\n{result.stderr}")
    
    # Determine if test passed
    passed = result.returncode == 0
    status = "\033[92mPASSED" if passed else "\033[91mFAILED"
    print(f"{status} in {duration:.2f}s (exit code: {result.returncode})\033[0m")
    
    return passed

def run_all_tests():
    """Run all tests in the tests directory."""
    # Get all test files
    test_files = sorted(glob.glob("tests/test_*.py"))
    
    if not test_files:
        print("No test files found in tests/ directory.")
        return False
    
    print(f"Found {len(test_files)} test files.")
    
    # Run each test
    results = {}
    for test_file in test_files:
        results[test_file] = run_test(test_file)
    
    # Print summary
    print("\n\033[1m=== Test Summary ===\033[0m")
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_file, passed_test in results.items():
        status = "\033[92mPASS" if passed_test else "\033[91mFAIL"
        print(f"{os.path.basename(test_file)}: {status}\033[0m")
    
    print(f"\n\033[1mResults: {passed}/{total} tests passed\033[0m")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 