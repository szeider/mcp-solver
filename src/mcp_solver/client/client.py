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
    """Helper function to format tool outputs."""
    if hasattr(result, "content"):
        out = result.content
    elif isinstance(result, dict):
        # Check if it's an MCP error response
        if result.get("isError") is True:
            out = f"ERROR: {result.get('content', 'Unknown error')}"
        else:
            out = result.get("content", str(result))
    elif isinstance(result, str):
        m = re.search(r"content='([^']*)'", result)
        out = m.group(1) if m else result
    else:
        out = str(result)
    return out.replace("\\n", "\n")

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
            log_system(f"{tool_name} called with args: {json.dumps(args_only, indent=2)}")
            
            try:
                result = func(call_args, config)
                formatted = format_tool_output(result)
                log_system(f"{tool_name} output: {formatted}")
                return result
            except Exception as e:
                error_msg = f"Tool execution failed: {str(e)}"
                log_system(f"Error: {error_msg}")
                return ToolError(error_msg)
        return wrapper
    
    # Apply wrappers to sync and async methods if they exist
    if hasattr(tool, "invoke"):
        updates["invoke"] = log_and_call(tool.invoke)
    
    if hasattr(tool, "ainvoke"):
        orig_ainvoke = tool.ainvoke
        async def ainvoke_wrapper(call_args, config=None):
            args_only = call_args.get("args", {})
            log_system(f"{tool_name} called with args: {json.dumps(args_only, indent=2)}")
            
            try:
                result = await orig_ainvoke(call_args, config)
                formatted = format_tool_output(result)
                log_system(f"{tool_name} output: {formatted}")
                return result
            except Exception as e:
                error_msg = f"Tool execution failed: {str(e)}"
                log_system(f"Error: {error_msg}")
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
    
    SOLVE_MODEL = LLMFactory.create_model(model_code)
    model_info = LLMFactory.get_model_info(SOLVE_MODEL)
    model_str = f"{model_info.platform}:{model_info.model_name}" if model_info else "Unknown"

    state["solve_llm"] = model_str

    print(f"Using model: {model_str}")

    # Get server command and args from command line or use defaults
    if state.get("server_command") and state.get("server_args"):
        # Use server command from command line
        mcp_command = state["server_command"]
        mcp_args = state["server_args"]
        print(f"Using custom server command: {mcp_command} {' '.join(mcp_args)}")
    else:
        # Use default server command
        mcp_command = DEFAULT_SERVER_COMMAND
        mcp_args = DEFAULT_SERVER_ARGS
        print(f"Using default server command: {mcp_command} {' '.join(mcp_args)}")

    try:
        # Set up server parameters for stdio connection
        server_params = StdioServerParameters(
            command=mcp_command,
            args=mcp_args
        )

        # Create a direct client session
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()
                
                # Get tools directly using the langchain-mcp-adapters
                raw_tools = await load_mcp_tools(session)
                
                # Wrap tools for better logging
                wrapped_tools = [wrap_tool(tool) for tool in raw_tools]
                
                # Print tools for debugging
                print(f"Available tools ({len(wrapped_tools)}):")
                for i, tool in enumerate(wrapped_tools):
                    print(f"  {i+1}. {tool.name}")

                # Custom callback handler for tool tracking
                class SimpleToolTracker(BaseCallbackHandler):
                    def on_tool_end(self, output, **kwargs):
                        tool_name = kwargs.get("name", "unknown_tool")
                        print(f"Tool executed: {tool_name}")

                # Create agent with higher recursion limit and our tool tracker
                config = RunnableConfig(
                    recursion_limit=100,
                    callbacks=[SimpleToolTracker()]
                )

                # Initialize the solver LLM and create the agent
                solver_llm = SOLVE_MODEL
                agent = create_react_agent(solver_llm, wrapped_tools)

                try:
                    # Regular invoke method
                    try:
                        print("Sending request to LLM...")
                        response = await agent.ainvoke({"messages": state["messages"]}, config=config)
                        print("Received response from LLM.")

                        # Extract the agent's response content
                        if response.get("messages") and len(response["messages"]) > 0:
                            agent_reply = response["messages"][-1].content
                            print("Agent reply received, length:", len(agent_reply))
                            print(agent_reply)

                            # Add the response to state messages
                            state["messages"].append({"role": "assistant", "content": agent_reply})
                        else:
                            print("Warning: No message content found in response")
                    except Exception as invoke_error:
                        print(f"Error during LLM invocation: {str(invoke_error)}")
                        raise invoke_error

                except Exception as e:
                    print(f"Agent error: {str(e)}")
                    import traceback
                    print(traceback.format_exc())

                    state["messages"].append({
                        "role": "assistant",
                        "content": f"I encountered an error while processing your request: {str(e)}. Please try again with a simpler query or check the model."
                    })
    except Exception as e:
        console.print(f"[bold red]Error connecting to MCP server: {str(e)}[/bold red]")
        sys.exit(1)

    return state

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
    state = await mcp_solver_node(state, args.model)

def main_cli():
    """Entry point for the command-line interface."""
    # Modified to use the main_wrapper for compatibility with previous mcp_react_os.py behavior
    asyncio.run(main_wrapper())

def main_wrapper():
    """
    Wrapper function to provide backward compatibility with mcp_react_os.py.
    This function sets default values for arguments if they're not provided.
    """
    # Modify the arguments to include defaults that mcp_react_os.py provided
    args = sys.argv[1:]
    modified_args = list(args)  # Create a copy to modify
    
    # Find the standard MiniZinc prompt
    base_dir = Path(__file__).parent.parent.parent.parent
    prompt_paths = [
        base_dir / "docs" / "standard_prompt_mzn.md",  # Try docs directory first
        base_dir / "instructions_prompt_mzn.md",  # Try root directory
    ]
    
    # Find the first prompt file that exists
    default_prompt = None
    for path in prompt_paths:
        if path.exists():
            default_prompt = str(path)
            break
    
    # Add --prompt if not specified and we found a default
    if not any(arg.startswith("--prompt") for arg in args) and default_prompt:
        modified_args.extend(["--prompt", default_prompt])
    
    # Set default server if not specified
    if not any(arg.startswith("--server") for arg in args):
        modified_args.extend(["--server", "uv run mcp-solver-mzn"])
        
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
    
    # Call the main function
    return main()

if __name__ == "__main__":
    main_cli() 