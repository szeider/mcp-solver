#!/usr/bin/env python
"""
Legacy test runner that forwards to the new test structure.
For backward compatibility.
"""
import os
import sys
import subprocess

# Get the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
tests_dir = os.path.join(script_dir, "tests")

def main():
    # Get command line arguments to forward
    args = sys.argv[1:]
    
    # Build the new command
    cmd = f"cd {tests_dir} && uv run run_all_tests.py {' '.join(args)}"
    
    # Print info
    print("Using new test structure...")
    print(f"Executing: {cmd}")
    
    # Execute the command
    try:
        result = subprocess.run(cmd, shell=True, check=False)
        return result.returncode
    except Exception as e:
        print(f"Error running tests: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 