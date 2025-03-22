#!/usr/bin/env python3
"""
Test client for MCP Solver with options for different agent implementations.

This script allows testing MCP solvers with either:
1. The standard implementation (using client.py)
2. Our custom ReAct agent implementation (using react_agent.py)
"""

import sys
import os
import argparse
import logging
from typing import Dict, Any, List
from pathlib import Path

from .llm_factory import LLMFactory, ModelInfo
from .client import log_system, set_system_title, load_file_content
from .react_agent import create_mcp_react_agent

# Model codes mapping - single source of truth for available models
MODEL_CODES = {
    "MC1": "AT:claude-3-7-sonnet-20250219",  # Anthropic Claude 3.7 direct
    "MC2": "OR:openai/o3-mini-high",         # OpenAI o3-mini-high via OpenRouter
    "MC3": "OR:openai/o3-mini",              # OpenAI o3-mini via OpenRouter
    "MC4": "OA:o3-mini:high",                # OpenAI o3-mini with high reasoning effort
    "MC5": "OA:o3-mini",                     # OpenAI o3-mini with default (medium) reasoning effort
    "MC6": "OA:gpt-4o",                      # OpenAI GPT-4o direct via OpenAI API
    "MC7": "OR:openai/gpt-4o"                # OpenAI GPT-4o via OpenRouter
}
DEFAULT_MODEL = "MC1"  # Default model to use

# Default server configuration
DEFAULT_SERVER_COMMAND = "uv run mcp-solver-mzn"

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MCP Solver Test Client')
    parser.add_argument(
        '--query',
        type=str,
        required=True,
        help='Path to the file containing the query'
    )
    parser.add_argument(
        '--server',
        type=str,
        default=DEFAULT_SERVER_COMMAND,
        help=f'Server command to use (default: {DEFAULT_SERVER_COMMAND})'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=list(MODEL_CODES.keys()),
        default=DEFAULT_MODEL,
        help=f'Model to use (default: {DEFAULT_MODEL}). Available models: {", ".join(MODEL_CODES.keys())}'
    )
    parser.add_argument(
        '--implementation',
        type=str,
        choices=['standard', 'react'],
        default='standard',
        help='Which implementation to use: standard (current) or react (new ReAct agent)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    return parser.parse_args()

def run_standard_implementation(query_path, server_command, model_code, verbose=False):
    """
    Run the standard implementation using the existing client.py script.
    
    This effectively just calls the main function from client.py with the appropriate arguments.
    """
    # Import the main function from client.py
    from .client import main_cli
    
    # Set environment variables for the standard implementation
    os.environ["MCP_SERVER_COMMAND"] = server_command
    os.environ["MCP_MODEL"] = model_code
    os.environ["MCP_QUERY_PATH"] = query_path
    os.environ["MCP_VERBOSE"] = "1" if verbose else "0"
    
    # Call the main function
    main_cli()

def run_react_implementation(query_path, server_command, model_code, verbose=False):
    """
    Run the new ReAct agent implementation.
    
    This uses our custom MCPReactAgent class to handle the query.
    """
    # Load the query content
    query = load_file_content(query_path)
    
    # Create the LLM
    llm = LLMFactory.create_model(model_code)
    
    # Create the React agent
    log_system(f"Using model: {model_code}")
    log_system(f"Using custom server command: {server_command}")
    
    # Create the system message based on query context
    system_message = f"""You are an expert constraint solver, trained to help users solve constraint programming problems.
Use the available tools to interact with a constraint solver and solve the problem described in the user's query.
Think step by step and use the tools effectively to build and solve the constraint model.
"""
    
    # Create the React agent
    agent = create_mcp_react_agent(
        llm=llm,
        server_command=server_command,
        system_message=system_message,
        verbose=verbose
    )
    
    # Run the agent on the query
    response = agent(query)
    
    # Print the response
    log_system("Agent reply received, length: " + str(len(response)))
    print(response)

def main():
    """Main function to run the test client."""
    args = parse_arguments()
    
    # Get the model code from the model name
    model_code = MODEL_CODES[args.model]
    
    if args.implementation == 'standard':
        run_standard_implementation(args.query, args.server, model_code, args.verbose)
    else:
        run_react_implementation(args.query, args.server, model_code, args.verbose)

if __name__ == "__main__":
    main() 