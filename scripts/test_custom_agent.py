#!/usr/bin/env python
"""
Script to run the custom agent test directly.
"""
import sys
import os
import subprocess
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

def main():
    """Main entry point for the test script wrapper."""
    # Add our project root to Python path
    sys.path.insert(0, str(PROJECT_ROOT))
    
    # Set the default command to run
    command = ["uv", "run", "python", "tests/run_test_custom_agent.py"]
    
    # Add any additional arguments passed to this script
    if len(sys.argv) > 1:
        command.extend(sys.argv[1:])
    
    print(f"Executing command: {' '.join(command)}")
    
    # Run the test process
    process = subprocess.run(command)
    
    # Return the exit code from the test
    return process.returncode

if __name__ == "__main__":
    sys.exit(main()) 