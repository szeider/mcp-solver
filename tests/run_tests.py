#!/usr/bin/env python3
"""
Unified Test Runner for MCP Solvers (MiniZinc, PySAT, Z3)
Runs problems through the appropriate test-client.
"""
import os
import subprocess
import glob
import sys
from pathlib import Path

# Add the project root to the Python path to enable imports from mcp_solver
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
from datetime import datetime
from collections import Counter
import threading
import json

# Now import the get_prompt_path function from the module
from src.mcp_solver.core.prompt_loader import get_prompt_path

# Import test configuration constants (excluding prompt files)
from test_config import (
    MCP_CLIENT_DIR, 
    DEFAULT_TIMEOUT, 
    MZN_PROBLEMS_DIR,
    PYSAT_PROBLEMS_DIR,
    Z3_PROBLEMS_DIR,
)

# ====================
# Shared Utility Functions
# ====================

def read_problem_file(file_path):
    """Read content from a problem file"""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading problem file: {str(e)}")
        return ""

def save_json_result(result_path, solver_type, problem_name, problem_content, 
                    model_content, solution_content, result_status):
    """Save test results to a JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(result_path, exist_ok=True)
    
    json_filename = os.path.join(result_path, f"{solver_type}_{problem_name}_{timestamp}.json")
    
    result_data = {
        "problem": problem_content,
        "model": model_content,
        "solution": solution_content,
        "result": result_status
    }
    
    try:
        with open(json_filename, 'w') as f:
            json.dump(result_data, f, indent=2)
        print(f"\nSaved JSON result to: {json_filename}")
    except Exception as e:
        print(f"\nError saving JSON result to {json_filename}: {str(e)}")
        
    return json_filename

def save_text_files(output_dir, problem_name, model_content, agent_response, model_ext=".txt"):
    """Save model and response to text files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    response_file = os.path.join(output_dir, f"{problem_name}_{timestamp}_response.txt")
    # Use the provided model extension
    model_file = os.path.join(output_dir, f"{problem_name}_{timestamp}_model{model_ext}")
    
    # Save model content
    if model_content:
        try:
            with open(model_file, 'w') as f:
                f.write(model_content)
            print(f"\nSaved final model to: {model_file}")
        except Exception as e:
             print(f"\nError saving model file to {model_file}: {str(e)}")

    
    # Save agent response
    if agent_response:
        try:
            with open(response_file, 'w') as f:
                f.write(agent_response)
            print(f"\nSaved agent response to: {response_file}")
        except Exception as e:
            print(f"\nError saving response file to {response_file}: {str(e)}")

    return model_file, response_file

def print_tool_stats(tool_calls, problem_name=None):
    """Print statistics about tool usage in a box"""
    if not tool_calls:
        return

    title = f"Tool Usage Statistics for {problem_name}" if problem_name else "Overall Tool Usage Statistics"
    max_tool_len = max(len(tool) for tool in tool_calls.keys()) if tool_calls else 0
    max_len = max(len(title), max_tool_len + 10) # Adjust 10 for count formatting

    print(f"\n┌{'─' * (max_len + 2)}┐")
    print(f"│ {title.center(max_len)} │")
    print(f"├{'─' * (max_tool_len + 2)}┬{'─' * (max_len - max_tool_len - 1)}┐")
    print(f"│ {'Tool'.ljust(max_tool_len)} │ {'Calls'.rjust(max_len - max_tool_len - 2)} │") # Adjusted alignment
    print(f"├{'─' * (max_tool_len + 2)}┼{'─' * (max_len - max_tool_len - 1)}┤")

    # Sort by most frequently used tools
    for tool, count in sorted(tool_calls.items(), key=lambda x: x[1], reverse=True):
        print(f"│ {tool.ljust(max_tool_len)} │ {str(count).rjust(max_len - max_tool_len - 2)} │")
    
    print(f"├{'─' * (max_tool_len + 2)}┼{'─' * (max_len - max_tool_len - 1)}┤")
    total_calls = sum(tool_calls.values())
    print(f"│ {'TOTAL'.ljust(max_tool_len)} │ {str(total_calls).rjust(max_len - max_tool_len - 2)} │")
    print(f"└{'─' * (max_tool_len + 2)}┴{'─' * (max_len - max_tool_len - 1)}┘")

def print_summary(problem_files, success_count, failed_tests, all_tool_calls, solver_type):
    """Print summary of test results in a box"""
    # Return success code if all tests passed, error code otherwise
    if len(failed_tests) == 0:
        return 0
    else:
        # Print failed tests without the summary box
        if failed_tests:
            print("\nFailed tests:")
            for test in failed_tests:
                print(f"  - {test}")
        return 1

def track_tool_usage(line):
    """Extract tool usage from a line of output"""
    if "TOOL:" in line:
        parts = line.split(":", 2)
        if len(parts) >= 2:
            tool_name = parts[1].strip()
            # Extract just the tool name without additional info (e.g., from "TOOL: solve_model <guid>")
            if " " in tool_name:
                tool_name = tool_name.split(" ")[0]
            return tool_name
    return None

# ====================
# Main Script Logic
# ====================

# --- Solver Configurations --- 
SOLVER_CONFIGS = {
    "mzn": {
        "solver_mode": "mzn", 
        "problems_dir": MZN_PROBLEMS_DIR,
        "command_template": "cd {mcp_client_dir} && uv run test-client-mzn --query {query_path}",
        "model_ext": ".mzn",
        "results_subdir": "mzn",
        "needs_server_arg": False,
    },
    "pysat": {
        "solver_mode": "pysat",
        "problems_dir": PYSAT_PROBLEMS_DIR,
        "command_template": "cd {mcp_client_dir} && uv run test-client --query {query_path} --server \"uv run mcp-solver-pysat\"",
        "model_ext": ".py",
        "results_subdir": "pysat",
        "needs_server_arg": True,
    },
    "z3": {
        "solver_mode": "z3",
        "problems_dir": Z3_PROBLEMS_DIR,
        "command_template": "cd {mcp_client_dir} && uv run test-client --query {query_path} --server \"uv run mcp-solver-z3\"",
        "model_ext": ".py",
        "results_subdir": "z3",
        "needs_server_arg": True,
    }
}

def validate_files_exist(config):
    """Check that required files and directories exist for the selected solver"""
    # Use the get_prompt_path function from prompt_loader.py
    prompt_path = get_prompt_path(config["solver_mode"])
    
    # Z3 can potentially auto-find prompt, so only error if not Z3 or if path doesn't exist
    if config['solver_mode'] != 'z3' and not prompt_path.exists():
        print(f"Error: Prompt file not found at expected location: {prompt_path}")
        return False
    elif config['solver_mode'] == 'z3' and not prompt_path.exists():
        print(f"Warning: Prompt file not found at {prompt_path}. test-client *may* try to find one.")
        # Allow Z3 to continue even if prompt not found initially

    # Check problems directory using the path from test_config
    if not os.path.exists(config["problems_dir"]):
        # Construct absolute path for clarity in error message
        abs_problems_dir = os.path.abspath(config["problems_dir"])
        print(f"Error: Problems directory '{config['problems_dir']}' (abs: {abs_problems_dir}) not found")
        return False
    
    # Check MCP client directory using the path from test_config
    if not os.path.exists(MCP_CLIENT_DIR):
        # Construct absolute path for clarity
        abs_mcp_client_dir = os.path.abspath(MCP_CLIENT_DIR)
        print(f"Error: MCP client directory '{MCP_CLIENT_DIR}' (abs: {abs_mcp_client_dir}) not found")
        return False
        
    return True

def run_test(problem_file, solver_name, config, verbose=False, timeout=DEFAULT_TIMEOUT, save_results=False, result_path=None):
    """Run a single problem through the appropriate test-client"""
    problem_name = os.path.basename(problem_file).replace('.md', '')
    
    # Create a nice box for the header
    header_text = f"Testing problem [{solver_name.upper()}]: {problem_name}"
    box_width = len(header_text) + 2  # Add some padding
    
    print(f"\n┌{'─' * (box_width + 2)}┐")
    print(f"│ {header_text.center(box_width)} │")
    print(f"└{'─' * (box_width + 2)}┘")
    
    abs_problem_path = os.path.abspath(problem_file)
    
    # Build the command with basic parameters
    cmd_params = {
        "mcp_client_dir": MCP_CLIENT_DIR,
        "query_path": abs_problem_path,
    }
    
    # Use the command template directly (no prompt path needed)
    cmd = config["command_template"].format(**cmd_params)

    # Add common args: timeout and verbose if supported
    if verbose:
         print("Verbose flag enabled: Output will be printed in real-time.")
         # If test-client/test-client-mzn support a standard --verbose, it could be added here.
         # cmd += " --verbose" 
    
    if timeout and timeout != DEFAULT_TIMEOUT:
        # Assume test-client and test-client-mzn accept --timeout
        cmd += f" --timeout {timeout}"
        
    # --- Setup for saving results ---
    output_dir = None
    if save_results:
        output_dir = os.path.join(os.path.dirname(__file__), "results", config["results_subdir"])

    # --- Initialize data capture ---
    tool_calls = Counter()
    problem_content = read_problem_file(problem_file)
    model_content = ""
    agent_response = ""
    solution_content = ""
    
    # Define a closure for read_stream to access variables from run_test
    def read_stream(stream, prefix, line_list):
        nonlocal model_content, agent_response, solution_content, tool_calls
        
        capture_model = False
        capture_response = False
        capture_solution = False
        
        for line in iter(stream.readline, ''):
            # Store in line_list for processing
            line_list.append(line)
            
            # Print the line in real-time
            print(line, end='')
            
            # Track tool usage
            tool_name = track_tool_usage(line)
            if tool_name:
                tool_calls[tool_name] += 1
            
            # --- Content Capture Logic --- 
            cleaned_line = line
            
            # Capture model
            if "Current model:" in line:
                capture_model = True
                model_content = "" 
                continue
            elif capture_model:
                # Stop capturing model if a tool call appears
                if line.strip().startswith("TOOL:"):
                    capture_model = False
                else:
                    model_content += cleaned_line 
            
            # Capture agent response
            if "Agent reply received" in line:
                capture_response = True
                agent_response = "" # Reset for final response
                continue
            elif capture_response:
                 # Heuristic: stop capturing response at separators or specific prefixes
                if "------------------------------------------------------------" in line or \
                   line.startswith("STDERR:") or \
                   line.startswith("[CLIENT OUTPUT END]"): 
                    capture_response = False
                else:
                    agent_response += cleaned_line
    
            # Capture solution
            if "=== SOLUTION ===" in line:
                capture_solution = True
                solution_content = ""
                continue
            elif capture_solution:
                if "=== END SOLUTION ===" in line:
                    capture_solution = False
                else:
                    solution_content += cleaned_line
    
    # --- Execute the command ---
    print(f"Running command: {cmd}")
    start_time = datetime.now()
    exit_code = -1 # Default to error
    result_status = "error" # Default status

    try:
        print(f"{'-'*60}\n[CLIENT OUTPUT START]\n{'-'*60}")
        
        # Use Popen with threading for output capture
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            bufsize=1, 
            universal_newlines=True
        )
        
        # --- Threaded Output Reading ---
        stdout_lines = []
        stderr_lines = []
        
        stdout_thread = threading.Thread(target=read_stream, args=(process.stdout, "STDOUT", stdout_lines))
        stderr_thread = threading.Thread(target=read_stream, args=(process.stderr, "STDERR", stderr_lines))
        
        stdout_thread.start()
        stderr_thread.start()
        
        # --- Wait for process completion ---
        try:
            exit_code = process.wait(timeout=timeout)
            result_status = "success" if exit_code == 0 else "failure"
        except subprocess.TimeoutExpired:
            process.kill() # Ensure the process is terminated
            # Give threads a moment to finish reading potentially buffered output
            stdout_thread.join(timeout=1) 
            stderr_thread.join(timeout=1)
            print(f"\n{'-'*60}\n[CLIENT OUTPUT END - TIMED OUT]\n{'-'*60}")
            print(f"\n⏱️ Test timed out after {timeout} seconds: {problem_name}")
            result_status = "timeout"

        # Wait for reading threads to finish processing remaining output
        stdout_thread.join()
        stderr_thread.join()

        # --- Process results --- 
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Save text files if requested
        if save_results and output_dir:
             save_text_files(output_dir, problem_name, model_content, agent_response, config['model_ext'])

        # Save JSON result if path provided (always save, regardless of success/fail/timeout)
        if result_path:
            save_json_result(
                result_path, 
                solver_name, 
                problem_name, 
                problem_content, 
                model_content, 
                solution_content, 
                result_status # Use the determined status
            )

        # Display tool usage statistics
        print_tool_stats(tool_calls, problem_name)

        if result_status == "success":
            print("Test completed successfully")
            return True, tool_calls
        elif result_status == "failure":
             print(f"ERROR: Test failed with non-zero exit code ({exit_code})")
             return False, tool_calls
        else: # Timeout or Error
            return False, tool_calls # Already printed timeout message

    except Exception as e:
        # Catch potential errors during process setup or unexpected issues
        print(f"\n{'-'*60}\n[CLIENT OUTPUT END - ERROR]\n{'-'*60}")
        print(f"\n❌ An unexpected error occurred while testing {problem_name}: {str(e)}")
        result_status = "error"
        # Attempt to save JSON even on error
        if result_path:
             save_json_result(
                 result_path, 
                 solver_name, 
                 problem_name, 
                 problem_content, 
                 model_content, 
                 solution_content, 
                 result_status
             )
        return False, Counter()

def main():
    """Main function to run MCP tests for a specified solver"""
    parser = argparse.ArgumentParser(description="Run MCP problems through test-client")
    parser.add_argument("solver", choices=SOLVER_CONFIGS.keys(), 
                       help="Specify the solver type to test (mzn, pysat, z3)")
    parser.add_argument("--problem", help="Path to specific problem file (.md)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output (prints output in real-time)")
    parser.add_argument("--timeout", "-t", type=int, default=DEFAULT_TIMEOUT, 
                       help=f"Timeout in seconds per problem (default: {DEFAULT_TIMEOUT})")
    parser.add_argument("--save", "-s", action="store_true", 
                        help="Save model and response text files to default results directory")
    parser.add_argument("--resultpath", help="Path to save JSON results for each problem (enables JSON saving)")
    args = parser.parse_args()

    solver_config = SOLVER_CONFIGS[args.solver]
    solver_name = args.solver

    # Ensure MCP_CLIENT_DIR is absolute for cd command robustness
    # Do this early before using it in validate_files_exist or run_test
    global MCP_CLIENT_DIR
    MCP_CLIENT_DIR = os.path.abspath(MCP_CLIENT_DIR)

    # Validate required files exist
    if not validate_files_exist(solver_config):
        return 1
    
    # Get problem files based on command-line arguments
    problems_dir = solver_config["problems_dir"]
    if args.problem:
        # Direct file path only - must exist
        problem_path = args.problem
        if not os.path.exists(problem_path):
            print(f"Error: Problem file not found at '{problem_path}'")
            return 1
        problem_files = [problem_path]
    else:
        problem_files = glob.glob(os.path.join(problems_dir, "*.md"))
    
    if not problem_files:
         # Construct absolute path for clarity in error message
        abs_problems_dir = os.path.abspath(problems_dir)
        print(f"Error: No problem files found in {problems_dir} (abs: {abs_problems_dir})")
        return 1
    
    print(f"Found {len(problem_files)} {solver_name.upper()} problem(s) to test")
    
    # Track results
    success_count = 0
    failed_tests = []
    all_tool_calls = Counter()
    
    # Run each test
    for problem_file in sorted(problem_files):
        success, tool_counts = run_test(
            problem_file,
            solver_name,
            solver_config, 
            verbose=args.verbose, 
            timeout=args.timeout, 
            save_results=args.save,
            result_path=args.resultpath
        )
        if success:
            success_count += 1
        else:
            # Add solver name to failed test for clarity 
            failed_tests.append(f"{solver_name}: {os.path.basename(problem_file)}") 
        
        # Aggregate tool usage stats
        all_tool_calls.update(tool_counts)
    
    # Print summary and return appropriate exit code
    return print_summary(problem_files, success_count, failed_tests, all_tool_calls, solver_name.upper())

if __name__ == "__main__":
    sys.exit(main()) 