#!/usr/bin/env python
"""
MiniZinc Problem Test Runner
Runs all MiniZinc problems through the mcp-react-os client
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
MCP_CLIENT_DIR = "/Users/stefanszeider/git/mcp-client"
DEFAULT_TIMEOUT = 300  # 5 minutes default timeout

def validate_files_exist():
    """Check that required files and directories exist"""
    if not os.path.exists(PROMPT_FILE):
        print(f"Error: Prompt file '{PROMPT_FILE}' not found")
        return False
    
    if not os.path.exists(MZN_PROBLEMS_DIR):
        print(f"Error: Problems directory '{MZN_PROBLEMS_DIR}' not found")
        return False
    
    if not os.path.exists(MCP_CLIENT_DIR):
        print(f"Error: MCP client directory '{MCP_CLIENT_DIR}' not found")
        return False
        
    return True

def run_test(problem_file, verbose=False, timeout=DEFAULT_TIMEOUT):
    """Run a single problem through mcp-react-os"""
    problem_name = os.path.basename(problem_file).replace('.md', '')
    print(f"\n{'='*60}")
    print(f"Testing problem: {problem_name}")
    print(f"{'='*60}")
    
    # Get absolute path to the problem file
    abs_problem_path = os.path.abspath(problem_file)
    abs_prompt_path = os.path.join(os.getcwd(), PROMPT_FILE)
    
    # Create the command with timeout
    cmd = f"cd {MCP_CLIENT_DIR} && uv run mcp-react-os --query {abs_problem_path} --prompt {abs_prompt_path}"
    if verbose:
        cmd += " --streaming"  # Add streaming output if verbose
    
    # Run the mcp-react-os command
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
                
                # Read and print stdout
                for line in iter(process.stdout.readline, ''):
                    if not line:
                        break
                    print(line, end='', flush=True)
                    stdout_lines.append(line)
                
                # Wait for process to complete
                process.wait(timeout=timeout)
                
                # Read and print any remaining stderr
                for line in iter(process.stderr.readline, ''):
                    if not line:
                        break
                    print(f"STDERR: {line}", end='', flush=True)
                    stderr_lines.append(line)
                
                success = process.returncode == 0
                
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"\n{'-'*60}\n[CLIENT OUTPUT END - TIMED OUT]\n{'-'*60}")
                print(f"\n⏱️ Test timed out after {timeout} seconds: {problem_name}")
                return False
        
        print(f"\n{'-'*60}\n[CLIENT OUTPUT END]\n{'-'*60}")
        
        if success:
            print(f"\n✅ Successfully tested: {problem_name}")
            return True
        else:
            print(f"\n❌ Failed to test {problem_name} (Exit code: {process.returncode})")
            return False
            
    except Exception as e:
        print(f"\n{'-'*60}\n[CLIENT OUTPUT END - ERROR]\n{'-'*60}")
        print(f"\n❌ Failed to test {problem_name}: {str(e)}")
        return False

def main():
    """Main function to run MiniZinc tests"""
    parser = argparse.ArgumentParser(description="Run MiniZinc problems through mcp-react-os")
    parser.add_argument("problem", nargs="?", help="Specific problem to run (basename without .md)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--timeout", "-t", type=int, default=DEFAULT_TIMEOUT, 
                       help=f"Timeout in seconds (default: {DEFAULT_TIMEOUT})")
    args = parser.parse_args()
    
    # Validate required files exist
    if not validate_files_exist():
        return 1
    
    # Get problem files based on command-line arguments
    if args.problem:
        problem_path = os.path.join(MZN_PROBLEMS_DIR, f"{args.problem}.md")
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