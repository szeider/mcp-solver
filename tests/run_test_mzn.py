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
from collections import Counter

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
    
    # Initialize tool call counter
    tool_calls = Counter()
    
    # Run the command and capture output
    print(f"Running command: {cmd}")
    start_time = datetime.now()
    try:
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Process output in real-time to track tool usage
        for line in iter(process.stdout.readline, ''):
            if verbose:
                print(line, end='')
            
            # Track tool usage
            if "TOOL:" in line:
                parts = line.split(":", 2)
                if len(parts) >= 2:
                    tool_name = parts[1].strip()
                    # Extract just the tool name without additional info
                    if " " in tool_name:
                        tool_name = tool_name.split(" ")[0]
                    tool_calls[tool_name] += 1
        
        # Wait for process to complete with timeout
        exit_code = process.wait(timeout=timeout)
        
        # Read any stderr if needed
        stderr = process.stderr.read()
        
    except subprocess.TimeoutExpired:
        print(f"ERROR: Test timed out after {timeout} seconds")
        try:
            process.kill()
        except:
            pass
        return False, Counter()
        
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Print results
    print(f"Exit code: {exit_code}")
    print(f"Duration: {duration:.2f} seconds")
    
    # Display tool usage statistics
    if tool_calls:
        print(f"\n{'-'*60}")
        print(f"Tool Usage Statistics for {problem_name}:")
        print(f"{'-'*60}")
        
        # Sort by most frequently used tools
        for tool, count in sorted(tool_calls.items(), key=lambda x: x[1], reverse=True):
            print(f"  {tool}: {count} calls")
        
        print(f"  Total tool calls: {sum(tool_calls.values())}")
    
    if exit_code != 0:
        print("ERROR: Test failed with non-zero exit code")
        print("STDERR:")
        print(stderr)
        return False, tool_calls
    
    print("Test completed successfully")
    return True, tool_calls

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
    print(f"SUMMARY: MiniZinc Tests ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
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
        print("\n🎉 All MiniZinc tests passed! 🎉")
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main()) 