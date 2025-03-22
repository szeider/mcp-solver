#!/usr/bin/env python
"""Custom test client for MCP Solver using the ReAct pattern."""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

from .llm_factory import LLMFactory, ModelInfo
from .react_agent import create_mcp_react_agent

# Model codes mapping - copied from client.py for consistency
MODEL_CODES = {
    "MC1": "AT:claude-3-7-sonnet-20250219",  # Anthropic Claude 3.7 direct
    "MC2": "OR:openai/o3-mini-high",         # OpenAI o3-mini-high via OpenRouter
    "MC3": "OR:openai/o3-mini",              # OpenAI o3-mini via OpenRouter
    "MC4": "OA:o3-mini:high",                # OpenAI o3-mini with high reasoning effort
    "MC5": "OA:o3-mini",                     # OpenAI o3-mini with default (medium) reasoning effort
    "MC6": "OA:gpt-4o",                      # OpenAI GPT-4o direct via OpenAI API
    "MC7": "OR:openai/gpt-4o"                # OpenAI GPT-4o via OpenRouter
}

DEFAULT_MODEL = "MC1"
DEFAULT_SERVER_COMMAND = "uv run mcp-solver-mzn"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MCP Solver Test Client using ReAct")
    
    parser.add_argument(
        '--query',
        type=str,
        required=True,
        help='Path to query file'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        help='Path to custom prompt file'
    )
    parser.add_argument(
        '--server',
        type=str,
        help='Server command to use instead of defaults. Format: "command arg1 arg2 arg3..."'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=list(MODEL_CODES.keys()),
        default=DEFAULT_MODEL,
        help=f'Model to use (default: {DEFAULT_MODEL}). Available models: {", ".join(MODEL_CODES.keys())}'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    return parser.parse_args()

def load_file_content(file_path):
    """Load content from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        sys.exit(1)

def main():
    """Main function to run the client."""
    # Parse arguments
    args = parse_args()
    
    # Look for default prompts if --prompt is not specified
    prompt_path = args.prompt
    if not prompt_path:
        base_dir = Path(__file__).parent.parent.parent.parent
        prompt_paths = [
            base_dir / "docs" / "standard_prompt_mzn.md",  # Try docs directory first
            base_dir / "instructions_prompt_mzn.md",       # Try root directory
        ]
        
        # Find the first prompt file that exists
        for path in prompt_paths:
            if path.exists():
                prompt_path = str(path)
                print(f"Using default prompt: {path}")
                break
        
        if not prompt_path:
            print("Error: No prompt file specified and could not find a default prompt")
            sys.exit(1)
    
    # Set default server command if not specified
    server_cmd = args.server or DEFAULT_SERVER_COMMAND
    
    # Load prompt and query
    system_message = load_file_content(prompt_path)
    query = load_file_content(args.query)
    
    # Create model
    model_code = MODEL_CODES.get(args.model, MODEL_CODES[DEFAULT_MODEL])
    llm = LLMFactory.create_model(model_code)
    model_info = LLMFactory.get_model_info(llm)
    model_str = f"{model_info.platform}:{model_info.model_name}" if model_info else "Unknown"
    print(f"Using model: {model_str}")
    print(f"Using custom server command: {server_cmd}")
    
    # Create ReAct agent
    react_agent = create_mcp_react_agent(
        llm=llm,
        server_command=server_cmd,
        system_message=system_message,
        verbose=args.verbose
    )
    
    # Invoke the agent
    try:
        result = react_agent(query)
        
        # Print the result
        print("Agent reply received, length:", len(result) if result else 0)
        print(result)
        
        return 0
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 