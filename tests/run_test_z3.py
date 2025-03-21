#!/usr/bin/env python3
"""
Z3 Problem Test Runner
Runs Z3 problems through the test-client
"""
import os
import subprocess
import glob
import sys
import argparse
from datetime import datetime
from pathlib import Path
from collections import Counter

# Import test configuration
from test_config import (
    MCP_CLIENT_DIR, 
    DEFAULT_TIMEOUT, 
    Z3_PROMPT_FILE,
    Z3_PROBLEMS_DIR,
    get_prompt_path
)

def validate_files_exist():
    """Check that required files and directories exist"""
    prompt_path = get_prompt_path(Z3_PROMPT_FILE)
    if not os.path.exists(prompt_path):
        print(f"Warning: Prompt file '{Z3_PROMPT_FILE}' not found at {prompt_path}")
        print("The test-client-z3 command will attempt to find a suitable prompt file automatically")
    
    if not os.path.exists(Z3_PROBLEMS_DIR):
        print(f"Error: Problems directory '{Z3_PROBLEMS_DIR}' not found")
        return False
    
    if not os.path.exists(MCP_CLIENT_DIR):
        print(f"Error: MCP client directory '{MCP_CLIENT_DIR}' not found")
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
    
    # Create the command with Z3 specific args
    cmd = f"cd {MCP_CLIENT_DIR} && uv run test-client --query {abs_problem_path} --prompt {get_prompt_path(Z3_PROMPT_FILE)} --server 'uv run mcp-solver-z3'"
    if verbose:
        cmd += " --streaming"  # Add streaming output if verbose
    
    # Initialize tool call counter
    tool_calls = Counter()
    
    # Run the test-client-z3 command
    try:
        print(f"\nExecuting: {cmd}\n")
        print(f"{'-'*60}\n[CLIENT OUTPUT START]\n{'-'*60}")
        
        # Always show output in real-time
        if verbose:
            # In streaming mode, use standard subprocess.run to show output in real-time
            process = subprocess.run(cmd, shell=True, check=False)
            success = process.returncode == 0
            # Note: In streaming mode, we cannot track tool calls reliably
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
                    
                    # Track tool usage
                    if "TOOL:" in line:
                        parts = line.split(":", 2)
                        if len(parts) >= 2:
                            tool_name = parts[1].strip()
                            # Extract just the tool name without additional info
                            if " " in tool_name:
                                tool_name = tool_name.split(" ")[0]
                            tool_calls[tool_name] += 1
                
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
                return False, Counter()
        
        print(f"\n{'-'*60}\n[CLIENT OUTPUT END]\n{'-'*60}")
        
        # Display tool usage statistics
        if tool_calls:
            print(f"\n{'-'*60}")
            print(f"Tool Usage Statistics for {problem_name}:")
            print(f"{'-'*60}")
            
            # Sort by most frequently used tools
            for tool, count in sorted(tool_calls.items(), key=lambda x: x[1], reverse=True):
                print(f"  {tool}: {count} calls")
            
            print(f"  Total tool calls: {sum(tool_calls.values())}")
        
        if success:
            print(f"\n✅ Successfully tested: {problem_name}")
            return True, tool_calls
        else:
            print(f"\n❌ Failed to test {problem_name} (Exit code: {process.returncode})")
            return False, tool_calls
            
    except Exception as e:
        print(f"\n{'-'*60}\n[CLIENT OUTPUT END - ERROR]\n{'-'*60}")
        print(f"\n❌ Failed to test {problem_name}: {str(e)}")
        return False, Counter()

def main():
    """Main function to run Z3 tests"""
    parser = argparse.ArgumentParser(description="Run Z3 problems through test-client")
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
        problem_path = os.path.join(Z3_PROBLEMS_DIR, f"{args.problem}")
        if not problem_path.endswith('.md'):
            problem_path += '.md'
            
        if not os.path.exists(problem_path):
            print(f"Error: Problem file '{problem_path}' not found")
            return 1
        problem_files = [problem_path]
    else:
        problem_files = glob.glob(f"{Z3_PROBLEMS_DIR}/*.md")
    
    if not problem_files:
        print(f"Error: No problem files found in {Z3_PROBLEMS_DIR}")
        return 1
    
    print(f"Found {len(problem_files)} Z3 problem(s) to test")
    
    # Track results
    success_count = 0
    failed_tests = []
    all_tool_calls = Counter()
    
    # Run each test
    for problem_file in sorted(problem_files):
        success, tool_counts = run_test(problem_file, verbose=args.verbose, timeout=args.timeout)
        if success:
            success_count += 1
        else:
            failed_tests.append(os.path.basename(problem_file))
        
        # Aggregate tool usage stats
        all_tool_calls.update(tool_counts)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: Z3 Tests ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"{'='*60}")
    print(f"Total tests: {len(problem_files)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(problem_files) - success_count}")
    
    if failed_tests:
        print("\nFailed tests:")
        for test in failed_tests:
            print(f"  - {test}")
    
    # Print overall tool usage statistics
    if all_tool_calls:
        print(f"\n{'-'*60}")
        print(f"OVERALL TOOL USAGE STATISTICS:")
        print(f"{'-'*60}")
        
        # Sort by most frequently used tools
        for tool, count in sorted(all_tool_calls.items(), key=lambda x: x[1], reverse=True):
            print(f"  {tool}: {count} calls")
        
        print(f"  Total tool calls: {sum(all_tool_calls.values())}")
        print(f"  Average tool calls per problem: {sum(all_tool_calls.values()) / len(problem_files):.2f}")
    
    if len(problem_files) == success_count:
        print("\n🎉 All Z3 tests passed! 🎉")
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main()) 