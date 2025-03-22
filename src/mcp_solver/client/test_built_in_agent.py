#!/usr/bin/env python
"""
Test script to verify the built-in agent works correctly with the MCP client.
This runs a simple test with USE_CUSTOM_AGENT=False.
"""

import os
import sys
import subprocess
from pathlib import Path

# Set environment variable to force using the built-in agent
os.environ["USE_CUSTOM_AGENT"] = "false"

def run_test(solver_type="mzn", problem_name="nqueens", verbose=True):
    """Run a test using the built-in agent."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent.parent.absolute()
    
    # Command to run the test
    cmd = [
        "uv", "run", f"test-client-{solver_type}",
        "--problem", problem_name
    ]
    
    if verbose:
        cmd.append("--verbose")
    
    print(f"Running test with built-in agent (USE_CUSTOM_AGENT=False):")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)
    
    # Run the command and capture output
    result = subprocess.run(
        cmd,
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Print results
    print(f"Exit code: {result.returncode}")
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    return result.returncode == 0

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Test the built-in agent")
    parser.add_argument("--solver", choices=["mzn", "pysat", "z3"], default="mzn",
                        help="Solver type to test (default: mzn)")
    parser.add_argument("--problem", default="nqueens",
                        help="Problem to solve (default: nqueens)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output")
    
    args = parser.parse_args()
    
    # Run the test
    success = run_test(
        solver_type=args.solver,
        problem_name=args.problem,
        verbose=not args.quiet
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 