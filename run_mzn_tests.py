#!/usr/bin/env python3
"""
MiniZinc Problem Test Runner
Runs all MiniZinc problems through the test-client
"""
import os
import subprocess
import glob
import sys
import argparse
from datetime import datetime

# Configuration
PROMPT_FILE = "instructions_prompt_lite.md"  # MiniZinc lite prompt file
MZN_PROBLEMS_DIR = "problems/mzn"
DEFAULT_TIMEOUT = 300  # 5 minutes default timeout

def validate_files_exist():
    """Check that required files and directories exist"""
    if not os.path.exists(PROMPT_FILE):
        print(f"Error: Prompt file '{PROMPT_FILE}' not found")
        return False
    
    if not os.path.exists(MZN_PROBLEMS_DIR):
        print(f"Error: Problems directory '{MZN_PROBLEMS_DIR}' not found")
        return False
        
    return True

def run_test(problem_file, verbose=False, timeout=DEFAULT_TIMEOUT):
    """Run a single problem through test-client"""
    problem_name = os.path.basename(problem_file).replace('.md', '')
    print(f"\n{'='*60}")
    print(f"Testing problem: {problem_name}")
    print(f"{'='*60}")
    
    # Get absolute path to the problem file
    abs_problem_path = os.path.abspath(problem_file)
    
    # Create the command with timeout
    cmd = f"uv run test-client --query {abs_problem_path}"
    if verbose:
        cmd += " --streaming"  # Add streaming output if verbose
    
    # Run the test-client command
    try:
        # Use shell=True to execute cd command before running the actual command
        print(f"\nExecuting: {cmd}\n")
        print(f"{'-'*60}\n[CLIENT OUTPUT START]\n{'-'*60}")
        
        # Always show output in real-time
        if verbose:
            # In streaming mode, use standard subprocess.run to show output in real-time
            process = subprocess.run(cmd, shell=True, check=False)
            success = process.returncode == 0
        else:
            # Capture output but print it immediately
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
            
            # Set a timeout for the process
            try:
                # Print stdout in real-time while capturing it
                stdout_lines = []
                stderr_lines = []

                # Read and process stdout
                for line in process.stdout:
                    print(line, end='')  # Print in real-time
                    stdout_lines.append(line)  # Save for later

                # Make sure to get all remaining output
                stdout, stderr = process.communicate(timeout=timeout)
                if stdout:
                    print(stdout, end='')
                    stdout_lines.append(stdout)

                # Get the return code
                return_code = process.wait(timeout=timeout)
                success = return_code == 0

                # Print stderr if there was an error
                if not success:
                    print("\nSTDERR:")
                    print(stderr)
                else:
                    # Print stderr as information
                    stderr_lines = stderr.splitlines()
                    if stderr_lines:
                        print("\nSTDERR:", end='')
                        for line in stderr_lines:
                            print(line)
                            
            except subprocess.TimeoutExpired:
                # Handle process timeout
                process.kill()
                print(f"\n\n❌ ERROR: Process timed out after {timeout} seconds")
                stdout, stderr = process.communicate()
                if stdout:
                    print("Last stdout:", stdout)
                if stderr:
                    print("Last stderr:", stderr)
                success = False

        print(f"\n{'-'*60}\n[CLIENT OUTPUT END]\n{'-'*60}\n")
        
        if success:
            print(f"✅ Successfully tested: {problem_name}")
            return True
        else:
            print(f"❌ Failed to test: {problem_name}")
            return False
            
    except Exception as e:
        print(f"❌ Error running test: {str(e)}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run MiniZinc problems through test-client")
    parser.add_argument("problem", nargs="?", help="Specific problem to run (basename without .md)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--timeout", "-t", type=int, default=DEFAULT_TIMEOUT, help=f"Timeout in seconds (default: {DEFAULT_TIMEOUT})")
    args = parser.parse_args()
    
    # Check if required files exist
    if not validate_files_exist():
        sys.exit(1)
    
    # Get problems to test
    if args.problem:
        # Test a specific problem
        problem_files = glob.glob(f"{MZN_PROBLEMS_DIR}/{args.problem}.md")
        if not problem_files:
            print(f"Error: Problem file '{args.problem}.md' not found in {MZN_PROBLEMS_DIR}")
            sys.exit(1)
    else:
        # Test all problems
        problem_files = glob.glob(f"{MZN_PROBLEMS_DIR}/*.md")
        if not problem_files:
            print(f"Error: No problem files found in {MZN_PROBLEMS_DIR}")
            sys.exit(1)
        
    print(f"Found {len(problem_files)} MiniZinc problem(s) to test")
    
    # Run tests
    success_count = 0
    for problem_file in problem_files:
        if run_test(problem_file, args.verbose, args.timeout):
            success_count += 1
    
    # Print summary
    print(f"\n{'='*60}")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"SUMMARY: MiniZinc Tests ({current_time})")
    print(f"{'='*60}")
    print(f"Total tests: {len(problem_files)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(problem_files) - success_count}")
    
    if success_count == len(problem_files):
        print("\n🎉 All MiniZinc tests passed! 🎉")
    else:
        print("\n⚠️ Some MiniZinc tests failed ⚠️")
        sys.exit(1)

if __name__ == "__main__":
    main() 