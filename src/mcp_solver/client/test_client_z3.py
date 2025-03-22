#!/usr/bin/env python
"""Entry point for test-client-z3 command."""
import os
import sys
from pathlib import Path
from mcp_solver.client.client import main_cli as client_main

def main():
    """Run the test client with Z3 defaults."""
    # Find the standard Z3 prompt
    base_dir = Path(__file__).parent.parent.parent.parent
    prompt_paths = [
        base_dir / "docs" / "standard_prompt_z3.md",     # Try docs directory first
        base_dir / "instructions_prompt_z3.md",      # Try root directory
    ]
    
    # Find the first prompt file that exists
    default_prompt = None
    for path in prompt_paths:
        if path.exists():
            default_prompt = str(path)
            print(f"Using Z3 prompt: {path}")
            break
    
    if not default_prompt:
        print("Warning: Could not find default Z3 prompt file")
    
    # Set default server to Z3
    server_cmd = "uv run mcp-solver-z3"
    
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
            problem_path = base_dir / "tests" / "problems" / "z3" / problem_name
            modified_args[i + 1] = str(problem_path)
    
    # Add --prompt if not specified and we found a default
    if not any(arg.startswith("--prompt") for arg in args) and default_prompt:
        modified_args.extend(["--prompt", default_prompt])
    
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
            # Keep verbose flag for the ReAct agent
            filtered_args.append(arg)
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
    
    # Call the main function from test_client.py
    return client_main()

if __name__ == "__main__":
    main() 