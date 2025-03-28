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

def run_test(problem_file, verbose=False, timeout=DEFAULT_TIMEOUT, save_results=False):
    """Run a single problem through test-client-mzn"""
    problem_name = os.path.basename(problem_file).replace('.md', '')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    
    # Set up result saving if enabled
    if save_results:
        # Define output directory for saving results
        output_dir = os.path.join(os.path.dirname(__file__), "results", "mzn")
        os.makedirs(output_dir, exist_ok=True)
        
        # Define output file paths with timestamp
        response_file = os.path.join(output_dir, f"{problem_name}_{timestamp}_response.txt")
        model_file = os.path.join(output_dir, f"{problem_name}_{timestamp}_model.mzn")
    
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
        
        print(f"{'-'*60}\n[CLIENT OUTPUT START]\n{'-'*60}")
        
        # Initialize variables to capture model and response if saving is enabled
        if save_results:
            model_content = ""
            agent_response = ""
            capture_model = False
            capture_response = False
        
        # Process output in real-time to track tool usage
        for line in iter(process.stdout.readline, ''):
            # Always print output
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
            
            # If saving is enabled, parse output to extract model and response
            if save_results:
                # Capture the final model content
                if "Current model:" in line:
                    capture_model = True
                    model_content = ""  # Reset to get only the latest model
                    continue
                elif capture_model:
                    if line.strip().startswith("TOOL:"):
                        capture_model = False
                    else:
                        model_content += line
                
                # Capture agent response
                if "Agent reply received" in line:
                    capture_response = True
                    continue
                elif capture_response and not line.strip().startswith("------------------------------------------------------------"):
                    if "STDERR:" in line:
                        capture_response = False
                    else:
                        agent_response += line
        
        # Wait for process to complete with timeout
        exit_code = process.wait(timeout=timeout)
        
        # Read and print any stderr
        stderr = process.stderr.read()
        if stderr:
            print(f"\nSTDERR:\n{stderr}")
            
        print(f"\n{'-'*60}\n[CLIENT OUTPUT END]\n{'-'*60}")
        
        # Save the model and agent response if enabled
        if save_results:
            if model_content:
                with open(model_file, 'w') as f:
                    f.write(model_content)
                print(f"\nSaved final model to: {model_file}")
                
            if agent_response:
                with open(response_file, 'w') as f:
                    f.write(agent_response)
                print(f"\nSaved agent response to: {response_file}")
        
    except subprocess.TimeoutExpired:
        print(f"\n{'-'*60}\n[CLIENT OUTPUT END - TIMED OUT]\n{'-'*60}")
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
    parser.add_argument("--save", "-s", action="store_true", help="Save test results to files")
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
        success, tool_counts = run_test(problem_file, verbose=args.verbose, timeout=args.timeout, save_results=args.save)
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