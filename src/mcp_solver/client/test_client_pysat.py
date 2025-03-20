#!/usr/bin/env python
"""Entry point for test-client-pysat command."""
import os
import sys
from pathlib import Path
from mcp_solver.client.client import main_cli

def main():
    """Run the test client with PySAT defaults."""
    # Find the standard PySAT prompt
    base_dir = Path(__file__).parent.parent.parent.parent
    prompt_paths = [
        base_dir / "docs" / "standard_prompt_pysat.md",  # Try docs directory first
        base_dir / "instructions_prompt_pysat_lite.md",  # Try root directory
    ]
    
    # Find the first prompt file that exists
    default_prompt = None
    for path in prompt_paths:
        if path.exists():
            default_prompt = str(path)
            break
    
    if not default_prompt:
        print("Warning: Could not find default PySAT prompt file")
    
    # Set default server to PySAT
    server_cmd = "uv run mcp-solver-pysat"
    
    # Insert defaults into sys.argv if not overridden
    args = sys.argv[1:]
    modified_args = list(args)  # Create a copy to modify
    
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
            # Ignore verbose flag
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
    
    # Call the main client function
    return main_cli()

if __name__ == "__main__":
    main() 