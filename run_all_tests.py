#!/usr/bin/env python
"""
Master Test Runner
Runs all solver modes' test suites
"""
import subprocess
import sys
import argparse
import os
from datetime import datetime

DEFAULT_TIMEOUT = 300  # 5 minutes default timeout

def run_test_suite(name, script, args=None):
    """Run an individual test suite with optional args"""
    print(f"\n{'='*70}")
    print(f"Running {name} Test Suite")
    print(f"{'='*70}")
    
    # Base command with script path
    cmd = [sys.executable, script]
    
    # Add any provided arguments
    if args and args.verbose:
        cmd.append("--verbose")
    if args and args.timeout:
        cmd.extend(["--timeout", str(args.timeout)])
    if args and args.problem:
        cmd.append(args.problem)
    
    try:
        print(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        
        if result.returncode == 0:
            print(f"\n✅ {name} Test Suite: PASSED")
            return True
        else:
            print(f"\n❌ {name} Test Suite: FAILED (Exit code: {result.returncode})")
            return False
    except Exception as e:
        print(f"\n❌ {name} Test Suite: ERROR - {str(e)}")
        return False

def main():
    """Main function to run all test suites"""
    parser = argparse.ArgumentParser(description="Run solver test suites")
    parser.add_argument("--mode", "-m", choices=["mzn", "pysat", "z3", "all"], default="all",
                       help="Solver mode to test (default: all)")
    parser.add_argument("--problem", "-p", help="Specific problem to run (basename without .md)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--timeout", "-t", type=int, default=DEFAULT_TIMEOUT,
                       help=f"Timeout in seconds (default: {DEFAULT_TIMEOUT})")
    args = parser.parse_args()
    
    start_time = datetime.now()
    print(f"Starting tests at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define test suites
    test_suites = {
        "mzn": ("MiniZinc", "run_mzn_tests.py"),
        "pysat": ("PySAT", "run_pysat_tests.py"),
        "z3": ("Z3", "run_z3_tests.py")
    }
    
    # Select which suites to run
    if args.mode != "all":
        if args.mode not in test_suites:
            print(f"Error: Unknown mode '{args.mode}'")
            return 1
        selected_suites = {args.mode: test_suites[args.mode]}
    else:
        selected_suites = test_suites
    
    # Run selected test suites
    results = {}
    for mode, (name, script) in selected_suites.items():
        script_path = os.path.join(os.getcwd(), script)
        results[name] = run_test_suite(name, script_path, args)
    
    # Print final summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY ({end_time.strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"{'='*70}")
    print(f"Total execution time: {duration:.2f} seconds")
    
    for name in results:
        status = "PASSED" if results[name] else "FAILED"
        print(f"{name} Test Suite: {status}")
    
    # Exit with failure if any test suite failed
    if all(results.values()):
        print("\n🎉 All test suites passed! 🎉")
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main()) 