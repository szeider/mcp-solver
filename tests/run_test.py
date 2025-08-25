#!/usr/bin/env python3
"""
Unified Test Runner for MCP Solvers (MiniZinc, PySAT, Z3, ASP)
Runs problems through the appropriate test-client.
"""

import glob
import os
import re
import subprocess
import sys
from pathlib import Path


# Add the project root to the Python path to enable imports from mcp_solver
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import json
import threading
from collections import Counter
from datetime import datetime

# Import tool statistics
from src.mcp_solver.client.tool_stats import ToolStats

# Now import the get_prompt_path function from the module
from src.mcp_solver.core.prompt_loader import get_prompt_path

# Import test configuration constants (excluding prompt files)
from tests.test_config import (
    ASP_PROBLEMS_DIR,
    DEFAULT_TIMEOUT,
    MAXSAT_PROBLEMS_DIR,
    MCP_CLIENT_DIR,
    MZN_PROBLEMS_DIR,
    PYSAT_PROBLEMS_DIR,
    Z3_PROBLEMS_DIR,
)


# ====================
# Shared Utility Functions
# ====================


def format_box(text, width=None, style="single"):
    """Format text in a box with unified styling.

    Args:
        text (str): Text to display in box
        width (int, optional): Fixed width for the box. If None, auto-calculate from text length
        style (str): Box style - 'single', 'double', 'header', or 'section'

    Returns:
        str: Formatted box as string
    """
    if width is None:
        width = len(text) + 4  # Default padding

    padded_text = text.center(width - 4)

    if style == "header":
        # Simple header style for major sections
        return f"\n{'=' * width}\n{padded_text.center(width)}\n{'=' * width}"
    elif style == "section":
        # Simple section marker
        return f"\n{'-' * width}\n{padded_text.center(width)}\n{'-' * width}"
    elif style == "double":
        # Double-line box style
        top = f"╔{'═' * (width - 2)}╗"
        middle = f"║ {padded_text} ║"
        bottom = f"╚{'═' * (width - 2)}╝"
        return f"\n{top}\n{middle}\n{bottom}"
    else:
        # Single-line box style (default)
        top = f"┌{'─' * (width - 2)}┐"
        middle = f"│ {padded_text} │"
        bottom = f"└{'─' * (width - 2)}┘"
        return f"\n{top}\n{middle}\n{bottom}"


def read_problem_file(file_path):
    """Read content from a problem file"""
    try:
        with open(file_path) as f:
            return f.read()
    except Exception as e:
        print(f"Error reading problem file: {e!s}")
        return ""


def save_text_files(
    output_dir, problem_name, model_content, agent_response, model_ext=".txt"
):
    """Save model and response to text files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    response_file = os.path.join(output_dir, f"{problem_name}_{timestamp}_response.txt")
    model_file = os.path.join(
        output_dir, f"{problem_name}_{timestamp}_model{model_ext}"
    )

    # Save model content
    if model_content:
        try:
            with open(model_file, "w") as f:
                f.write(model_content)
            print(f"\nSaved final model to: {model_file}")
        except Exception as e:
            print(f"\nError saving model file to {model_file}: {e!s}")

    # Save agent response
    if agent_response:
        try:
            with open(response_file, "w") as f:
                f.write(agent_response)
            print(f"\nSaved agent response to: {response_file}")
        except Exception as e:
            print(f"\nError saving response file to {response_file}: {e!s}")

    return model_file, response_file


def print_tool_stats(tool_calls, problem_name=None):
    """Print statistics about tool usage in a box"""
    if not tool_calls:
        return

    title = (
        f"Tool Usage Statistics for {problem_name}"
        if problem_name
        else "Overall Tool Usage Statistics"
    )
    max_tool_len = max(len(tool) for tool in tool_calls.keys()) if tool_calls else 0
    box_width = max(len(title) + 6, max_tool_len + 15)

    print(f"\n┌{'─' * (box_width - 2)}┐")
    print(f"│ {title.center(box_width - 4)} │")
    print(f"├{'─' * (max_tool_len + 2)}┬{'─' * (box_width - max_tool_len - 5)}┤")
    print(
        f"│ {'Tool'.ljust(max_tool_len)} │ {'Calls'.rjust(box_width - max_tool_len - 7)} │"
    )
    print(f"├{'─' * (max_tool_len + 2)}┼{'─' * (box_width - max_tool_len - 5)}┤")

    # Sort by most frequently used tools
    for tool, count in sorted(tool_calls.items(), key=lambda x: x[1], reverse=True):
        print(
            f"│ {tool.ljust(max_tool_len)} │ {str(count).rjust(box_width - max_tool_len - 7)} │"
        )

    print(f"├{'─' * (max_tool_len + 2)}┼{'─' * (box_width - max_tool_len - 5)}┤")
    total_calls = sum(tool_calls.values())
    print(
        f"│ {'TOTAL'.ljust(max_tool_len)} │ {str(total_calls).rjust(box_width - max_tool_len - 7)} │"
    )
    print(f"└{'─' * (max_tool_len + 2)}┴{'─' * (box_width - max_tool_len - 5)}┘")


def format_section_marker(text):
    """Detect and format section markers in output texts"""
    # Simple markers we directly look for
    formatted_markers = {
        "PROBLEM STATEMENT:": "PROBLEM STATEMENT",
        "CONSTRAINTS:": "CONSTRAINTS",
        "VARIABLES:": "VARIABLES",
        "OBJECTIVE:": "OBJECTIVE",
        "=== SOLUTION ===": "SOLUTION",
        "=== END SOLUTION ===": "END SOLUTION",
    }

    # Check for exact marker matches
    for marker, display_text in formatted_markers.items():
        if marker in text:
            return format_box(display_text)

    # Check for equals-sign wrapped headers (like "======= HEADER =======")
    if text.startswith("=") and "=" * 3 in text:
        # Extract the header text between the equals signs
        parts = text.split("=")
        # Find the non-empty part that likely contains the header
        for part in parts:
            cleaned = part.strip()
            if cleaned and len(cleaned) > 1:  # Avoid single characters
                return format_box(cleaned)

    return None


def print_summary(
    problem_files, success_count, failed_tests, all_tool_calls, solver_type
):
    """Print summary of test results"""
    if failed_tests:
        print("\nFailed tests:")
        for test in failed_tests:
            print(f"  - {test}")
        return 1
    return 0


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

    # Also look for tool calls in the format "system: ▶ clear_model called with args: {}"
    if "system:" in line and "called with args" in line:
        match = re.search(r"system: ▶ (\w+) called with args", line)
        if match:
            return match.group(1)

    return None


# ====================
# Main Script Logic
# ====================

# --- Solver Configurations ---
SOLVER_CONFIGS = {
    "mzn": {
        "solver_mode": "mzn",
        "problems_dir": MZN_PROBLEMS_DIR,
        "command_template": 'cd {mcp_client_dir} && uv run test-client --query {query_path} --server "uv run mcp-solver-mzn"',
        "model_ext": ".mzn",
        "results_subdir": "mzn",
        "needs_server_arg": True,
    },
    "pysat": {
        "solver_mode": "pysat",
        "problems_dir": PYSAT_PROBLEMS_DIR,
        "command_template": 'cd {mcp_client_dir} && uv run test-client --query {query_path} --server "uv run mcp-solver-pysat"',
        "model_ext": ".py",
        "results_subdir": "pysat",
        "needs_server_arg": True,
    },
    "z3": {
        "solver_mode": "z3",
        "problems_dir": Z3_PROBLEMS_DIR,
        "command_template": 'cd {mcp_client_dir} && uv run test-client --query {query_path} --server "uv run mcp-solver-z3"',
        "model_ext": ".py",
        "results_subdir": "z3",
        "needs_server_arg": True,
    },
    "maxsat": {
        "solver_mode": "maxsat",
        "problems_dir": MAXSAT_PROBLEMS_DIR,
        "command_template": 'cd {mcp_client_dir} && uv run test-client --query {query_path} --server "uv run mcp-solver-maxsat"',
        "model_ext": ".py",
        "results_subdir": "maxsat",
        "needs_server_arg": True,
    },
    "asp": {
        "solver_mode": "asp",
        "problems_dir": ASP_PROBLEMS_DIR,
        "command_template": 'cd {mcp_client_dir} && uv run test-client --query {query_path} --server "uv run mcp-solver-asp"',
        "model_ext": ".lp",
        "results_subdir": "asp",
        "needs_server_arg": True,
    },
}


def validate_files_exist(config):
    """Check that required files and directories exist for the selected solver"""
    # Use the get_prompt_path function from prompt_loader.py
    prompt_path = get_prompt_path(config["solver_mode"])

    # Z3 can potentially auto-find prompt, so only error if not Z3 or if path doesn't exist
    if config["solver_mode"] != "z3" and not prompt_path.exists():
        print(f"Error: Prompt file not found at expected location: {prompt_path}")
        return False
    elif config["solver_mode"] == "z3" and not prompt_path.exists():
        print(
            f"Warning: Prompt file not found at {prompt_path}. test-client *may* try to find one."
        )
        # Allow Z3 to continue even if prompt not found initially

    # Check problems directory using the path from test_config
    if not os.path.exists(config["problems_dir"]):
        # Construct absolute path for clarity in error message
        abs_problems_dir = os.path.abspath(config["problems_dir"])
        print(
            f"Error: Problems directory '{config['problems_dir']}' (abs: {abs_problems_dir}) not found"
        )
        return False

    # Check MCP client directory using the path from test_config
    if not os.path.exists(MCP_CLIENT_DIR):
        # Construct absolute path for clarity
        abs_mcp_client_dir = os.path.abspath(MCP_CLIENT_DIR)
        print(
            f"Error: MCP client directory '{MCP_CLIENT_DIR}' (abs: {abs_mcp_client_dir}) not found"
        )
        return False

    return True


def run_test(
    problem_file,
    solver_name,
    config,
    verbose=False,
    timeout=DEFAULT_TIMEOUT,
    save_results=False,
    result_path=None,
    args=None,
):
    """Run a single problem through the appropriate test-client"""
    problem_name = os.path.basename(problem_file).replace(".md", "")

    # Use the unified box formatting
    header_text = f"Testing problem [{solver_name.upper()}]: {problem_name}"
    print(format_box(header_text))

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

    # NOTE: test-client doesn't accept a timeout parameter, but we still use it for overall process timeout
    # Timeout is applied to the subprocess itself, not passed as a command-line parameter

    # If result path provided, forward it to client.py
    if result_path:
        # Create a standardized filename for the JSON output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(result_path, exist_ok=True)
        json_filename = os.path.join(
            result_path, f"{solver_name}_{problem_name}_{timestamp}.json"
        )
        cmd += f" --result-path {json_filename}"

    # Add direct model code parameter if provided
    mc = getattr(args, "mc", None)
    if mc:
        cmd += f' --mc "{mc}"'

    # --- Setup for saving results ---
    output_dir = None
    if save_results:
        output_dir = os.path.join(
            os.path.dirname(__file__), "results", config["results_subdir"]
        )

    # --- Initialize data capture ---
    tool_calls = Counter()
    problem_content = read_problem_file(problem_file)
    model_content = ""
    agent_response = ""
    solution_content = ""
    review_content = ""
    tokens_used = 0
    review_verdict = "unknown"

    # Create a thread-safe dictionary to store output from the threads
    from threading import Lock

    output_lock = Lock()
    shared_output = {
        "tokens_used": 0,
        "solution_content": "",
        "review_content": "",
        "tools_called": 0,
        "review_verdict": "unknown",
    }

    # Define a closure for read_stream to access variables from run_test
    def read_stream(stream, prefix, line_list):
        """Process output stream and format section headers appropriately."""
        nonlocal model_content, agent_response, tool_calls

        capture_model = False
        capture_response = False
        capture_solution = False
        capture_review = False
        local_tokens_used = 0
        local_solution_content = ""
        local_review_content = ""
        local_tools_called = 0
        local_review_verdict = "unknown"

        # Headers that should be formatted as boxes
        section_headers = {
            "PROBLEM STATEMENT:": "PROBLEM STATEMENT",
            "PROBLEM STATEMENT": "PROBLEM STATEMENT",
            "FINAL MODEL:": "FINAL MODEL",
            "FINAL MODEL": "FINAL MODEL",
            "SOLUTION RESULT:": "SOLUTION RESULT",
            "SOLUTION RESULT": "SOLUTION RESULT",
            "REVIEW RESULT:": "REVIEW RESULT",
            "REVIEW RESULT": "REVIEW RESULT",
            "CONSTRAINTS:": "CONSTRAINTS",
            "VARIABLES:": "VARIABLES",
            "OBJECTIVE:": "OBJECTIVE",
            "=== SOLUTION ===": "SOLUTION",
            "=== END SOLUTION ===": "END SOLUTION",
        }

        for line in iter(stream.readline, ""):
            # Store in line_list for processing
            line_list.append(line)

            # Look for mem_ variables in the output
            if "mem_tokens_used:" in line:
                token_match = re.search(r"mem_tokens_used: (\d+)", line)
                if token_match:
                    local_tokens_used = int(token_match.group(1))
                    with output_lock:
                        shared_output["tokens_used"] = local_tokens_used

            if "mem_review_verdict:" in line:
                verdict_match = re.search(r"mem_review_verdict: (\w+)", line)
                if verdict_match:
                    local_review_verdict = verdict_match.group(1)
                    with output_lock:
                        shared_output["review_verdict"] = local_review_verdict

            # Look for token usage information - need to correctly identify the line with COMBINED TOTAL
            if "Token" in line and "COMBINED TOTAL" in line:
                # The token count should be in the last column, which is typically right-aligned
                # Use a regex to extract numeric value with potential 'k' suffix
                token_match = re.search(r"(\d+\.?\d*)k?", line)
                if token_match:
                    token_value = token_match.group(0)

                    if token_value.endswith("k"):
                        # Handle counts like "15k" by converting to thousands
                        try:
                            if "." in token_value:
                                local_tokens_used = int(
                                    float(token_value.replace("k", "")) * 1000
                                )
                            else:
                                local_tokens_used = (
                                    int(token_value.replace("k", "")) * 1000
                                )
                        except ValueError:
                            pass  # Keep as 0 if conversion fails
                    else:
                        # Handle numeric counts
                        try:
                            local_tokens_used = int(token_value)
                        except ValueError:
                            pass  # Keep as 0 if conversion fails

                    # Store in the shared dictionary if we don't already have a value from mem_tokens_used
                    with output_lock:
                        if shared_output["tokens_used"] == 0:
                            shared_output["tokens_used"] = local_tokens_used

            # Look for the total tool count in the statistics table
            if "Tool" in line and "TOTAL" in line and "│" in line:
                # Extract the total tool count from the line
                tool_match = re.search(r"TOTAL\s+\│\s+(\d+)", line)
                if tool_match:
                    try:
                        local_tools_called = int(tool_match.group(1))
                        with output_lock:
                            shared_output["tools_called"] = local_tools_called
                    except ValueError:
                        pass  # Keep as 0 if conversion fails

            # Direct match for agent entry lines (special case)
            if line.strip() == "system: Entering ReAct agent":
                print(format_box("REACT AGENT"))
                continue

            if line.strip() == "system: Entering Review agent":
                print(format_box("REVIEW AGENT"))
                continue

            # Check for section headers
            line_stripped = line.strip()

            # First check for exact matches
            if line_stripped in section_headers:
                print(format_box(section_headers[line_stripped]))

                # Start capturing solution or review based on the header
                if section_headers[line_stripped] == "SOLUTION RESULT":
                    capture_solution = True
                    local_solution_content = ""
                elif section_headers[line_stripped] == "REVIEW RESULT":
                    capture_review = True
                    local_review_content = ""

                continue

            # Look for correctness indicators in review output
            if "Correctness:" in line:
                if "correct" in line:
                    local_review_verdict = "correct"
                elif "incorrect" in line:
                    local_review_verdict = "incorrect"
                else:
                    local_review_verdict = "unknown"

                with output_lock:
                    shared_output["review_verdict"] = local_review_verdict

            # Then check for equals-sign wrapped headers
            if "=" * 10 in line and not line_stripped.startswith("system:"):
                # Skip pure equals-sign separator lines
                if line_stripped == "=" * len(line_stripped):
                    continue

                # Extract header text between equals signs
                text = line_stripped.replace("=", "").strip()
                if text and len(text) > 1:  # Avoid single characters
                    print(format_box(text))
                    continue

            # If no special formatting needed, print the line as is
            print(line, end="")

            # Track tool usage
            tool_name = track_tool_usage(line)
            if tool_name:
                tool_calls[tool_name] += 1

            # --- Content Capture Logic ---
            # Capture model
            if "Current model:" in line:
                capture_model = True
                model_content = ""
                continue
            elif capture_model:
                # Stop capturing model if a tool call appears
                if line_stripped.startswith("TOOL:") or line_stripped.startswith(
                    "system:"
                ):
                    capture_model = False
                else:
                    model_content += line

            # Capture agent response
            if "Agent reply received" in line:
                capture_response = True
                agent_response = ""  # Reset for final response
                continue
            elif capture_response:
                # Heuristic: stop capturing response at separators or specific prefixes
                if (
                    "------------------------------------------------------------"
                    in line
                    or line.startswith("STDERR:")
                    or line.startswith("[CLIENT OUTPUT END]")
                ):
                    capture_response = False
                else:
                    agent_response += line

            # Capture solution content
            if capture_solution:
                # Check for end of solution section - usually a newline or next section header
                if (
                    line_stripped == ""
                    or line_stripped.startswith("system:")
                    or (line_stripped.startswith("=") and "=" * 5 in line_stripped)
                ):
                    capture_solution = False
                    # Store in shared dictionary
                    with output_lock:
                        shared_output["solution_content"] = local_solution_content
                else:
                    local_solution_content += line

            # Capture review content
            if capture_review:
                # Check for end of review section - usually ends with a blank line or
                # starts a new section
                if (
                    line_stripped == ""
                    or line_stripped.startswith("system:")
                    or (line_stripped.startswith("=") and "=" * 5 in line_stripped)
                ):
                    capture_review = False
                    # Store in shared dictionary
                    with output_lock:
                        shared_output["review_content"] = local_review_content
                else:
                    local_review_content += line

    # --- Execute the command ---
    print(f"Running command: {cmd}")
    start_time = datetime.now()
    exit_code = -1  # Default to error
    result_status = "error"  # Default status

    try:
        # Use Popen with threading for output capture
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # --- Threaded Output Reading ---
        stdout_lines = []
        stderr_lines = []

        stdout_thread = threading.Thread(
            target=read_stream, args=(process.stdout, "STDOUT", stdout_lines)
        )
        stderr_thread = threading.Thread(
            target=read_stream, args=(process.stderr, "STDERR", stderr_lines)
        )

        stdout_thread.start()
        stderr_thread.start()

        # --- Wait for process completion ---
        try:
            exit_code = process.wait(timeout=timeout)
            result_status = "success" if exit_code == 0 else "failure"
        except subprocess.TimeoutExpired:
            process.kill()  # Ensure the process is terminated
            # Give threads a moment to finish reading potentially buffered output
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)
            print(f"\n⏱️ Test timed out after {timeout} seconds: {problem_name}")
            result_status = "timeout"

        # Wait for reading threads to finish processing remaining output
        stdout_thread.join()
        stderr_thread.join()

        # Retrieve collected data from the shared dictionary
        tokens_used = shared_output["tokens_used"]
        solution_content = shared_output["solution_content"]
        review_content = shared_output["review_content"]
        tools_called = shared_output["tools_called"]
        review_verdict = shared_output["review_verdict"]

        # --- Process results ---
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Save text files if requested
        if save_results and output_dir:
            save_text_files(
                output_dir,
                problem_name,
                model_content,
                agent_response,
                config["model_ext"],
            )

        # Map review verdict to a more standard result status if it exists
        if review_verdict and review_verdict != "unknown":
            if review_verdict == "correct":
                result_status = "correct"
            elif review_verdict == "incorrect":
                result_status = "incorrect"

        # Display tool usage statistics
        print_tool_stats(tool_calls, problem_name)

        if result_status == "success" or result_status == "correct":
            print("Test completed successfully")
            return True, tool_calls
        elif result_status == "failure" or result_status == "incorrect":
            print(f"ERROR: Test failed with non-zero exit code ({exit_code})")
            return False, tool_calls
        else:  # Timeout or Error
            return False, tool_calls  # Already printed timeout message

    except Exception as e:
        # Catch potential errors during process setup or unexpected issues
        print(f"\n❌ An unexpected error occurred while testing {problem_name}: {e!s}")
        result_status = "error"

        return False, Counter()


def main():
    """Main function to run MCP tests for a specified solver"""
    parser = argparse.ArgumentParser(description="Run MCP problems through test-client")
    parser.add_argument(
        "solver",
        choices=SOLVER_CONFIGS.keys(),
        help="Specify the solver type to test (mzn, pysat, z3, asp)",
    )
    parser.add_argument("--problem", help="Path to specific problem file (.md)")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output (prints output in real-time)",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout in seconds per problem (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--save",
        "-s",
        action="store_true",
        help="Save model and response text files to default results directory",
    )
    parser.add_argument(
        "--result", help="Path to folder where JSON results should be saved"
    )
    parser.add_argument(
        "--mc",
        help="Direct model code specification (e.g., AT:claude-3-7-sonnet-20250219)",
    )
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
        # Check if test.md exists in the problem directory, use it as default
        default_test_path = os.path.join(problems_dir, "test.md")
        if os.path.exists(default_test_path):
            print(f"No problem specified, using default test: {default_test_path}")
            problem_files = [default_test_path]
        else:
            # Fall back to all problems if test.md doesn't exist
            problem_files = glob.glob(os.path.join(problems_dir, "*.md"))

    if not problem_files:
        # Construct absolute path for clarity in error message
        abs_problems_dir = os.path.abspath(problems_dir)
        print(
            f"Error: No problem files found in {problems_dir} (abs: {abs_problems_dir})"
        )
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
            result_path=args.result,
            args=args,
        )
        if success:
            success_count += 1
        else:
            # Add solver name to failed test for clarity
            failed_tests.append(f"{solver_name}: {os.path.basename(problem_file)}")

        # Aggregate tool usage stats
        all_tool_calls.update(tool_counts)

    # Print summary and return appropriate exit code
    return print_summary(
        problem_files, success_count, failed_tests, all_tool_calls, solver_name.upper()
    )


if __name__ == "__main__":
    sys.exit(main())
