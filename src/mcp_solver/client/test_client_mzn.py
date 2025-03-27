#!/usr/bin/env python
"""Entry point for test-client-mzn command."""
import os
import sys
from pathlib import Path
from mcp_solver.client.client import main_cli as client_main
import asyncio

def main():
    """Run the test client with MiniZinc defaults."""
    # Get base directory and prompt directory
    base_dir = Path(__file__).parent.parent.parent.parent
    prompt_dir = base_dir / "prompts" / "mzn"
    
    # Check if the required prompt files exist
    instructions_path = prompt_dir / "instructions.md"
    review_path = prompt_dir / "review.md"
    
    # Verify both files exist
    if not instructions_path.exists() or not review_path.exists():
        missing = []
        if not instructions_path.exists(): missing.append("instructions.md")
        if not review_path.exists(): missing.append("review.md")
        print(f"Warning: Missing required prompt files in {prompt_dir}: {', '.join(missing)}", flush=True)
    
    # Set default server to MiniZinc
    server_cmd = "uv run mcp-solver-mzn"
    
    # Insert defaults into sys.argv if not overridden
    args = sys.argv[1:]
    modified_args = list(args)  # Create a copy to modify
    
    # Support old --problem syntax
    for i, arg in enumerate(args):
        if arg == "--problem" and i + 1 < len(args):
            # Convert --problem to --query with proper path
            modified_args[i] = "--query"
            problem_name = args[i + 1]
            if not problem_name.endswith('.md'):
                problem_name += '.md'
            
            # Set the full path to the problem
            problem_path = base_dir / "tests" / "problems" / "mzn" / problem_name
            modified_args[i + 1] = str(problem_path)
    
    # Add --server if not specified
    if not any(arg.startswith("--server") for arg in args):
        modified_args.extend(["--server", server_cmd])
        
    # Filter out flags that are meant for the test runner but not supported by the client
    # (We just silently ignore them to maintain compatibility with test scripts)
    filtered_args = []
    skip_next = False
    for i, arg in enumerate(modified_args):
        if skip_next:
            skip_next = False
            continue
            
        if arg == "--verbose" or arg == "-v":
            # Just skip verbose flag
            continue
        elif arg.startswith("--timeout") or arg == "-t":
            # Skip timeout flag and its value
            if i < len(modified_args) - 1 and not modified_args[i+1].startswith("--"):
                skip_next = True
            continue
        else:
            filtered_args.append(arg)
    
    # Replace sys.argv with our filtered version
    sys.argv = [sys.argv[0]] + filtered_args
    
    # Call the main function
    return client_main()

if __name__ == "__main__":
    sys.exit(main()) 