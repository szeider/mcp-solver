#!/usr/bin/env python
"""
Main Test Runner for MCP Solver
Runs tests for all solvers (MiniZinc, PySAT, Z3)
"""
import os
import sys
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from collections import Counter

# Import test configuration
from test_config import (
    DEFAULT_TIMEOUT,
    MZN_PROBLEMS_DIR,
    PYSAT_PROBLEMS_DIR,
    Z3_PROBLEMS_DIR
)

# Import runners
from run_test_mzn import main as run_mzn_tests
from run_test_pysat import main as run_pysat_tests
from run_test_z3 import main as run_z3_tests

def main():
    """Main function to run all tests"""
    parser = argparse.ArgumentParser(description="Run all MCP Solver tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--timeout", "-t", type=int, default=DEFAULT_TIMEOUT, 
                       help=f"Timeout in seconds (default: {DEFAULT_TIMEOUT})")
    parser.add_argument("--mzn", action="store_true", help="Run only MiniZinc tests")
    parser.add_argument("--pysat", action="store_true", help="Run only PySAT tests")
    parser.add_argument("--z3", action="store_true", help="Run only Z3 tests")
    parser.add_argument("--save", "-s", action="store_true", help="Save test results to files")
    args = parser.parse_args()
    
    # If no solver is specified, run all solvers
    run_all = not (args.mzn or args.pysat or args.z3)
    
    # Track results and metrics
    results = []
    all_problems_count = 0
    success_count = 0
    
    # Log start time
    start_time = datetime.now()
    print(f"\n{'='*60}")
    print(f"STARTING MCP SOLVER TEST RUN - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Run MiniZinc tests
    if run_all or args.mzn:
        print("\n\n=== RUNNING MINIZINC TESTS ===\n")
        sys.argv = ["run_test_mzn.py"]
        if args.verbose:
            sys.argv.append("--verbose")
        if args.timeout != DEFAULT_TIMEOUT:
            sys.argv.extend(["--timeout", str(args.timeout)])
        if args.save:
            sys.argv.append("--save")
        
        mzn_result = run_mzn_tests()
        results.append(("MiniZinc", mzn_result == 0))
    
    # Run PySAT tests
    if run_all or args.pysat:
        print("\n\n=== RUNNING PYSAT TESTS ===\n")
        sys.argv = ["run_test_pysat.py"]
        if args.verbose:
            sys.argv.append("--verbose")
        if args.timeout != DEFAULT_TIMEOUT:
            sys.argv.extend(["--timeout", str(args.timeout)])
        if args.save:
            sys.argv.append("--save")
        
        pysat_result = run_pysat_tests()
        results.append(("PySAT", pysat_result == 0))
    
    # Run Z3 tests
    if run_all or args.z3:
        print("\n\n=== RUNNING Z3 TESTS ===\n")
        sys.argv = ["run_test_z3.py"]
        if args.verbose:
            sys.argv.append("--verbose")
        if args.timeout != DEFAULT_TIMEOUT:
            sys.argv.extend(["--timeout", str(args.timeout)])
        if args.save:
            sys.argv.append("--save")
        
        z3_result = run_z3_tests()
        results.append(("Z3", z3_result == 0))
    
    # Log end time
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Count total problems
    if run_all or args.mzn:
        mzn_problems = len([f for f in os.listdir(MZN_PROBLEMS_DIR) if f.endswith('.md')])
        all_problems_count += mzn_problems
    if run_all or args.pysat:
        pysat_problems = len([f for f in os.listdir(PYSAT_PROBLEMS_DIR) if f.endswith('.md')])
        all_problems_count += pysat_problems
    if run_all or args.z3:
        z3_problems = len([f for f in os.listdir(Z3_PROBLEMS_DIR) if f.endswith('.md')])
        all_problems_count += z3_problems
    
    # Print summary
    print(f"\n\n{'='*60}")
    print(f"MCP SOLVER TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    print(f"\nResults:")
    
    all_passed = True
    for solver, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        all_passed = all_passed and passed
        if passed:
            success_count += 1
        print(f"  {solver}: {status}")
    
    print(f"\nTotal problem sets tested: {len(results)}")
    print(f"Successful problem sets: {success_count}")
    
    if all_passed:
        print("\n🎉 All tests passed! 🎉")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1
    
if __name__ == "__main__":
    sys.exit(main()) 