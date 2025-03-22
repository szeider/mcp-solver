import sys
import os
import asyncio
import argparse
import json
import re
import textwrap
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

# Core dependencies
from .llm_factory import LLMFactory, ModelInfo
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.runnables.config import RunnableConfig
from rich.console import Console

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
DEFAULT_SERVER_COMMAND = "uv"
DEFAULT_SERVER_ARGS = ["run", "mcp-solver-mzn"]

# Global Rich Console instance with color support
console = Console(color_system="truecolor")
_current_title = None  # Stores the current title for system messages

class ToolError:
    """Class to represent tool errors in a format that LangGraph can process correctly."""
    def __init__(self, message: str):
        self.content = f"ERROR: {message}"
        
    def __str__(self) -> str:
        return self.content

# Custom callback handler for tool tracking
class SimpleToolTracker(BaseCallbackHandler):
    def on_tool_end(self, output, **kwargs):
        # We no longer need to output here since we're already showing the output
        # in the wrapper functions with system: toolname output: result
        pass

def set_system_title(title: str) -> None:
    """Set a title for the system message."""
    global _current_title
    _current_title = title

def log_system(msg: str, title: str = None) -> None:
    """Log a system message with optional title."""
    global _current_title
    
    if title is not None:
        _current_title = title
    
    if _current_title:
        console.print(f"[bold blue]{_current_title}[/bold blue]: {msg}")
    else:
        console.print(f"[bold blue]system[/bold blue]: {msg}")
    
    # Ensure output is flushed immediately
    sys.stdout.flush()
    console.file.flush() if hasattr(console, 'file') else None

class ClientError(Exception):
    """Client related errors."""
    pass

def load_system_prompt(prompt_file: str = "system_prompt.md") -> str:
    """Load system prompt from a markdown file located at the project root."""
    try:
        if not os.path.exists(prompt_file):
            raise ClientError(f"System prompt file not found: {prompt_file}")
        
        with open(prompt_file, encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        raise ClientError(f"Failed to load system prompt: {e}")

def parse_arguments():
    """Parse command line arguments focusing on the essential parameters."""
    parser = argparse.ArgumentParser(description='MCP Solver Client')
    parser.add_argument(
        '--query', 
        type=str,
        required=True,
        help='Path to the file containing the problem query'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        required=True,
        help='Path to the file containing the system prompt'
    )
    parser.add_argument(
        '--server',
        type=str,
        help='Server command to use. Format: "command arg1 arg2 arg3..."'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=list(MODEL_CODES.keys()),
        default=DEFAULT_MODEL,
        help=f'Model to use (default: {DEFAULT_MODEL})'
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

def load_initial_state(custom_prompt_path, query_path) -> dict:
    """Initialize state with the custom system prompt and the query from file."""
    # Load custom prompt
    custom_prompt = load_file_content(custom_prompt_path)
    
    # Load query
    query = load_file_content(query_path)
    
    # Create initial messages with only the custom prompt
    messages = [{"role": "system", "content": custom_prompt}]

    # Add the user query
    messages.append({"role": "user", "content": query})

    return {
        "messages": messages,
        "is_pure_mode": True,
        "start_time": datetime.now()
    }

def format_tool_output(result):
    """Format tool outputs into readable text."""
    # Handle error objects
    if hasattr(result, "content"):
        return result.content
    
    # Handle dictionary responses
    if isinstance(result, dict):
        if result.get("isError") is True:
            return f"ERROR: {result.get('content', 'Unknown error')}"
        return result.get("content", str(result))
    
    # Handle string responses
    if isinstance(result, str):
        return result.replace("\\n", "\n")
    
    # Default: convert to string
    return str(result).replace("\\n", "\n")

def wrap_tool(tool):
    """Wrap a tool for logging with tidier output."""
    # Remove any "[Tool]" prefix from the tool's name if present.
    tool_name = tool.name
    if tool_name.startswith("[Tool]"):
        tool_name = tool_name.replace("[Tool]", "").strip()
    
    updates = {}
    
    # Define a wrapper function for both sync and async invocation
    def log_and_call(func):
        def wrapper(call_args, config=None):
            args_only = call_args.get("args", {})
            # Use one consolidated message for tool call
            log_system(f"▶ {tool_name} called with args: {json.dumps(args_only, indent=2)}")
            sys.stdout.flush()  # Force flush to ensure immediate output
            
            try:
                result = func(call_args, config)
                formatted = format_tool_output(result)
                log_system(f"{tool_name} output: {formatted}")
                sys.stdout.flush()  # Force flush again after tool completes
                return result
            except Exception as e:
                error_msg = f"Tool execution failed: {str(e)}"
                log_system(f"✖ Error: {error_msg}")
                sys.stdout.flush()  # Force flush error messages too
                return ToolError(error_msg)
        return wrapper
    
    # Apply wrappers to sync and async methods if they exist
    if hasattr(tool, "invoke"):
        updates["invoke"] = log_and_call(tool.invoke)
    
    if hasattr(tool, "ainvoke"):
        orig_ainvoke = tool.ainvoke
        async def ainvoke_wrapper(call_args, config=None):
            args_only = call_args.get("args", {})
            # Use one consolidated message for tool call
            log_system(f"▶ {tool_name} called with args: {json.dumps(args_only, indent=2)}")
            sys.stdout.flush()  # Force flush to ensure immediate output
            
            try:
                result = await orig_ainvoke(call_args, config)
                formatted = format_tool_output(result)
                log_system(f"{tool_name} output: {formatted}")
                sys.stdout.flush()  # Force flush after tool completes
                return result
            except Exception as e:
                error_msg = f"Tool execution failed: {str(e)}"
                log_system(f"✖ Error: {error_msg}")
                sys.stdout.flush()  # Force flush error messages
                return ToolError(error_msg)
        updates["ainvoke"] = ainvoke_wrapper
    
    if updates:
        return tool.copy(update=updates)
    else:
        log_system(f"Warning: {tool_name} has no invoke or ainvoke method; cannot wrap for logging.")
        return tool

async def mcp_solver_node(state: dict, model_name: str) -> dict:
    """Processes the conversation via the MCP solver with direct tool calling."""
    state["solver_visit_count"] = state.get("solver_visit_count", 0) + 1

    # Get the model code from the model name
    if model_name not in MODEL_CODES:
        console.print(f"[bold red]Error: Unknown model '{model_name}'. Using default: {DEFAULT_MODEL}[/bold red]")
        model_name = DEFAULT_MODEL
    
    model_code = MODEL_CODES[model_name]
    
    # Check if required API key is present
    try:
        model_info = LLMFactory.parse_model_code(model_code)
        api_key_name = model_info.api_key_name
        api_key = os.environ.get(api_key_name)
        
        if not api_key:
            error_msg = f"Error: {api_key_name} not found in environment variables. Please set {api_key_name} in your .env file."
            console.print(f"[bold red]{error_msg}[/bold red]")
            state["messages"].append({
                "role": "assistant",
                "content": error_msg
            })
            return state
    except Exception as e:
        console.print(f"[bold red]Error parsing model code: {str(e)}[/bold red]")
        state["messages"].append({
            "role": "assistant",
            "content": f"Error parsing model code: {str(e)}"
        })
        return state
    
    # Create model and get model info
    SOLVE_MODEL = LLMFactory.create_model(model_code)
    model_info = LLMFactory.get_model_info(SOLVE_MODEL)
    model_str = f"{model_info.platform}:{model_info.model_name}" if model_info else "Unknown"
    
    state["solve_llm"] = model_str
    print(f"Using model: {model_str}", flush=True)

    # Get server command and args from command line or use defaults
    if state.get("server_command") and state.get("server_args"):
        # Use server command from command line
        mcp_command = state["server_command"]
        mcp_args = state["server_args"]
        print(f"Using custom server command: {mcp_command} {' '.join(mcp_args)}", flush=True)
    else:
        # Use default server command
        mcp_command = DEFAULT_SERVER_COMMAND
        mcp_args = DEFAULT_SERVER_ARGS
        print(f"Using default server command: {mcp_command} {' '.join(mcp_args)}", flush=True)

    # Set up server parameters for stdio connection
    server_params = StdioServerParameters(
        command=mcp_command,
        args=mcp_args
    )

    try:
        # Create a direct client session and initialize MCP tools
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                try:
                    # Initialize the connection
                    await session.initialize()
                    
                    # Get tools directly using the langchain-mcp-adapters
                    raw_tools = await load_mcp_tools(session)
                    
                    # Wrap tools for better logging
                    wrapped_tools = [wrap_tool(tool) for tool in raw_tools]
                    
                    # Print tools for debugging
                    print("\n📋 Available tools:", flush=True)
                    for i, tool in enumerate(wrapped_tools):
                        print(f"  {i+1}. {tool.name}", flush=True)
                    print("", flush=True)
                    sys.stdout.flush()

                    # Configure the agent with tool tracker
                    config = RunnableConfig(
                        recursion_limit=100,
                        callbacks=[SimpleToolTracker()]
                    )

                    # Initialize the agent with tools
                    agent = create_react_agent(SOLVE_MODEL, wrapped_tools)

                    # Process the request
                    print(f"\n{'='*60}")
                    print("Sending request to LLM...")
                    print(f"{'='*60}\n")
                    sys.stdout.flush()
                    try:
                        response = await agent.ainvoke({"messages": state["messages"]}, config=config)
                        print(f"\n{'='*60}")
                        print("Received response from LLM.")
                        print(f"{'='*60}\n")
                        sys.stdout.flush()

                        # Extract and add the agent's response to state
                        if response.get("messages") and len(response["messages"]) > 0:
                            agent_reply = response["messages"][-1].content
                            print(f"Agent reply received, length: {len(agent_reply)}", flush=True)
                            state["messages"].append({"role": "assistant", "content": agent_reply})
                        else:
                            print("Warning: No message content found in response", flush=True)
                    except Exception as e:
                        error_msg = f"Error during LLM invocation: {str(e)}"
                        print(error_msg, flush=True)
                        state["messages"].append({
                            "role": "assistant", 
                            "content": f"I encountered an error while processing your request: {str(e)}. Please try again with a simpler query or check the model."
                        })
                except Exception as e:
                    error_msg = f"Error initializing MCP session: {str(e)}"
                    console.print(f"[bold red]{error_msg}[/bold red]")
                    state["messages"].append({
                        "role": "assistant",
                        "content": error_msg
                    })
    except Exception as e:
        error_msg = f"Error connecting to MCP server: {str(e)}"
        console.print(f"[bold red]{error_msg}[/bold red]")
        state["messages"].append({
            "role": "assistant",
            "content": error_msg
        })

    return state

def main_cli():
    """Entry point for the command-line interface."""
    # Run the main async function
    asyncio.run(main())

async def main():
    """Main async function for one-shot execution."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load initial state with custom prompt and query
    try:
        state = load_initial_state(args.prompt, args.query)
    except Exception as e:
        console.print(f"[bold red]Error loading files: {str(e)}[/bold red]")
        sys.exit(1)
    
    # If server command is provided, parse it into command and args
    if args.server:
        command_parts = args.server.split()
        state["server_command"] = command_parts[0]
        state["server_args"] = command_parts[1:] if len(command_parts) > 1 else []
    
    # Run the solver once
    await mcp_solver_node(state, args.model)

if __name__ == "__main__":
    main_cli() 