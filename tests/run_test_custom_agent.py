#!/usr/bin/env python
"""
Test script for the custom agent implementation with MCP tools.
"""
import asyncio
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import necessary modules
from mcp_solver.client.client import mcp_solver_node
from mcp_solver.client.react_agent import normalize_state

# Import test configuration
from tests.test_config import (
    DEFAULT_TIMEOUT,
    RESULTS_DIR,
)

# Problem definitions by solver type
MZN_PROBLEMS = {
    "nqueens": "mzn/nqueens.md",
    "sudoku": "mzn/sudoku.md",
}

PYSAT_PROBLEMS = {
    "graph_coloring": "pysat/graph_coloring.md",
    "scheduling": "pysat/scheduling.md",
    "furniture-arrangement": "pysat/furniture-arrangement.md",
    "mine-sweeper-hard": "pysat/mine-sweeper-hard.md",
    "sudoku-16x16": "pysat/sudoku-16x16.md",
}

Z3_PROBLEMS = {
    "bounded_sum": "z3/bounded_sum.md",
    "cryptarithmetic": "z3/cryptarithmetic.md",
}

ALL_PROBLEMS = {**MZN_PROBLEMS, **PYSAT_PROBLEMS, **Z3_PROBLEMS}
DEFAULT_PROBLEM = "nqueens"

# Mapping from problem to solver
PROBLEM_TO_SERVER = {
    **{p: "mcp-solver-mzn" for p in MZN_PROBLEMS},
    **{p: "mcp-solver-pysat" for p in PYSAT_PROBLEMS},
    **{p: "mcp-solver-z3" for p in Z3_PROBLEMS},
}

# Mapping from problem to instruction prompt
PROBLEM_TO_PROMPT = {
    **{p: "instructions_prompt_mzn.md" for p in MZN_PROBLEMS},
    **{p: "instructions_prompt_pysat.md" for p in PYSAT_PROBLEMS}, 
    **{p: "instructions_prompt_z3.md" for p in Z3_PROBLEMS},
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test the improved custom agent implementation")
    parser.add_argument(
        "--problem", 
        choices=list(ALL_PROBLEMS.keys()),
        default=DEFAULT_PROBLEM,
        help=f"Problem to solve (default: {DEFAULT_PROBLEM})"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--timeout", "-t", 
        type=int, 
        default=DEFAULT_TIMEOUT,
        help=f"Timeout in seconds (default: {DEFAULT_TIMEOUT})"
    )
    parser.add_argument(
        "--save", "-s", 
        action="store_true",
        help="Save test results to the results directory"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="MC1",
        help="Model code to use (default: MC1 - Claude 3.7 Sonnet)"
    )
    return parser.parse_args()

async def run_test(args):
    """Run the test with the improved custom agent."""
    problem_name = args.problem
    problem_path = ALL_PROBLEMS.get(problem_name)
    server_name = PROBLEM_TO_SERVER.get(problem_name)
    prompt_file = PROBLEM_TO_PROMPT.get(problem_name)
    
    if not problem_path or not server_name or not prompt_file:
        print(f"Error: Problem '{problem_name}' not found.")
        return False
    
    # Set up paths
    base_dir = Path(__file__).parent
    project_root = base_dir.parent  # Root directory of the project
    problem_file = base_dir / "problems" / problem_path
    prompt_file = project_root / prompt_file  # Look in project root for prompt files
    
    if not problem_file.exists():
        print(f"Error: Problem file '{problem_file}' not found.")
        return False
    
    if not prompt_file.exists():
        print(f"Error: Prompt file '{prompt_file}' not found.")
        return False
    
    # Print test information
    print(f"\n{'='*60}")
    print(f"Testing improved custom agent with {problem_name} problem")
    print(f"Using server: {server_name}")
    print(f"{'='*60}\n")
    
    # Create initial state
    start_time = datetime.now()
    
    try:
        # Load system prompt
        with open(prompt_file, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
        
        # Load problem definition
        with open(problem_file, "r", encoding="utf-8") as f:
            problem_query = f.read().strip()
        
        # Initialize state
        state = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": problem_query}
            ],
            "server_command": "uv",
            "server_args": ["run", server_name],
            "is_pure_mode": True,
            "start_time": start_time,
        }
        
        # Force enable custom agent
        from mcp_solver.client.client import USE_CUSTOM_AGENT
        import mcp_solver.client.client as client_module
        
        # Set custom agent flag to True
        setattr(client_module, "USE_CUSTOM_AGENT", True)
        print("Enabled custom agent implementation.")
        
        # Run the solver node
        print(f"Running test with problem: {problem_name}")
        print(f"Using model: {args.model}")
        
        # Execute the solver
        updated_state = await mcp_solver_node(state, args.model)
        
        # Check if the test completed successfully
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Check for assistant response
        if len(updated_state["messages"]) > 2:
            print(f"\nTest completed in {duration:.2f} seconds")
            print(f"Final message count: {len(updated_state['messages'])}")
            
            # Save results if requested
            if args.save:
                save_results(updated_state, problem_name, args.model, duration)
            
            return True
        else:
            print("\nTest failed: No response from the assistant.")
            return False
        
    except Exception as e:
        print(f"Error during test execution: {str(e)}")
        print(traceback.format_exc())
        return False

def save_results(state, problem_name, model_name, duration):
    """Save test results to a file."""
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = results_dir / f"{problem_name}_{model_name}_{timestamp}.json"
    
    # Prepare results for saving
    results = {
        "problem": problem_name,
        "model": model_name,
        "duration_seconds": duration,
        "timestamp": timestamp,
        "messages": [
            {
                "role": msg.get("role", "unknown"),
                "content": msg.get("content", "")
            } for msg in state["messages"]
        ]
    }
    
    # Save to file
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {result_file}")

def main():
    """Main entry point for the test script."""
    args = parse_args()
    
    try:
        result = asyncio.run(run_test(args))
        return 0 if result else 1
    except KeyboardInterrupt:
        print("\nTest interrupted.")
        return 130
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 