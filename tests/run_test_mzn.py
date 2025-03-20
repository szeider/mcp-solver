#!/usr/bin/env python3
"""
MiniZinc Problem Test Runner
Runs MiniZinc problems through the test-client
"""
import os
import subprocess
import glob
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Import test configuration
from test_config import (
    MCP_CLIENT_DIR, 
    DEFAULT_TIMEOUT, 
    MZN_PROMPT_FILE,
    MZN_PROBLEMS_DIR,
    get_prompt_path
)

def validate_files_exist():
    """Check that required files and directories exist"""
    if not os.path.exists(get_prompt_path(MZN_PROMPT_FILE)):
        print(f"Error: Prompt file '{MZN_PROMPT_FILE}' not found")
        return False
    
    if not os.path.exists(MZN_PROBLEMS_DIR):
        print(f"Error: Problems directory '{MZN_PROBLEMS_DIR}' not found")
        return False
    
    if not os.path.exists(MCP_CLIENT_DIR):
        print(f"Error: MCP client directory '{MCP_CLIENT_DIR}' not found")
        return False
        
    return True

def run_test(problem_file, verbose=False, timeout=DEFAULT_TIMEOUT):
    """Run a single problem through test-client-mzn"""
    problem_name = os.path.basename(problem_file).replace('.md', '')
    print(f"\n{'='*60}")
    print(f"Testing problem: {problem_name}")
    print(f"{'='*60}")
    
    # Get absolute path to the problem file
    abs_problem_path = os.path.abspath(problem_file)
    
    # Create the command with MiniZinc specific args
    cmd = f"cd {MCP_CLIENT_DIR} && uv run test-client-mzn --query {abs_problem_path}"
    
    # Add verbose flag if requested
    if verbose:
        cmd += " --verbose"
    
    # Add timeout if specified
    if timeout and timeout != DEFAULT_TIMEOUT:
        cmd += f" --timeout {timeout}"
    
    # Run the command and capture output
    print(f"Running command: {cmd}")
    start_time = datetime.now()
    try:
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Use communicate with timeout to prevent hanging
        stdout, stderr = process.communicate(timeout=timeout)
        
        # Get exit code
        exit_code = process.returncode
        
    except subprocess.TimeoutExpired:
        print(f"ERROR: Test timed out after {timeout} seconds")
        try:
            process.kill()
        except:
            pass
        return False
        
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Print results
    print(f"Exit code: {exit_code}")
    print(f"Duration: {duration:.2f} seconds")
    
    if exit_code != 0:
        print("ERROR: Test failed with non-zero exit code")
        print("STDERR:")
        print(stderr)
        return False
    
    if verbose:
        print("STDOUT:")
        print(stdout)
    
    print("Test completed successfully")
    return True

def main():
    """Main function to run MiniZinc tests"""
    parser = argparse.ArgumentParser(description="Run MiniZinc problems through test-client")
    parser.add_argument("--problem", help="Specific problem to run (without .md extension)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--timeout", "-t", type=int, default=DEFAULT_TIMEOUT, 
                       help=f"Timeout in seconds (default: {DEFAULT_TIMEOUT})")
    args = parser.parse_args()
    
    # Validate required files exist
    if not validate_files_exist():
        return 1
    
    # Get problem files based on command-line arguments
    if args.problem:
        problem_path = os.path.join(MZN_PROBLEMS_DIR, f"{args.problem}")
        if not problem_path.endswith('.md'):
            problem_path += '.md'
            
        if not os.path.exists(problem_path):
            print(f"Error: Problem file '{problem_path}' not found")
            return 1
        problem_files = [problem_path]
    else:
        problem_files = glob.glob(f"{MZN_PROBLEMS_DIR}/*.md")
    
    if not problem_files:
        print(f"Error: No problem files found in {MZN_PROBLEMS_DIR}")
        return 1
    
    print(f"Found {len(problem_files)} MiniZinc problem(s) to test")
    
    # Track results
    success_count = 0
    failed_tests = []
    
    # Run each test
    for problem_file in sorted(problem_files):
        if run_test(problem_file, verbose=args.verbose, timeout=args.timeout):
            success_count += 1
        else:
            failed_tests.append(os.path.basename(problem_file))
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: MiniZinc Tests ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"{'='*60}")
    print(f"Total tests: {len(problem_files)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(problem_files) - success_count}")
    
    if failed_tests:
        print("\nFailed tests:")
        for test in failed_tests:
            print(f"  - {test}")
    
    if len(problem_files) == success_count:
        print("\n🎉 All MiniZinc tests passed! 🎉")
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main()) 