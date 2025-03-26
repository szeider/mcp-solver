import sys
import os
import asyncio
import argparse
import json
from datetime import datetime
from typing import Dict, Any
from rich.console import Console
import traceback

# Core dependencies
from .llm_factory import LLMFactory
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent

# Import tool stats tracking
from .tool_stats import ToolStats

# Check LangGraph version for compatibility
import importlib
import importlib.metadata
try:
    langgraph_version = importlib.metadata.version("langgraph")
    LANGGRAPH_VERSION = tuple(map(int, langgraph_version.split(".")[:3]))
    USING_NEW_LANGGRAPH = LANGGRAPH_VERSION >= (0, 3, 18)
except (importlib.metadata.PackageNotFoundError, ValueError):
    # If we can't determine version, assume older version
    LANGGRAPH_VERSION = (0, 0, 0)
    USING_NEW_LANGGRAPH = False

# Custom agent implementation
from mcp_solver.client.react_agent import create_custom_react_agent, run_agent, normalize_state

# For testing purposes - force using the custom agent
USE_CUSTOM_AGENT = True

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
    
    # Ensure output is flushed immediately
    sys.stdout.flush()
    console.file.flush() if hasattr(console, 'file') else None

class ClientError(Exception):
    """Client related errors."""
    pass

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
    # Load custom prompt and query
    custom_prompt = load_file_content(custom_prompt_path)
    query = load_file_content(query_path)
    
    # Create initial messages
    messages = [
        {"role": "system", "content": custom_prompt},
        {"role": "user", "content": query}
    ]

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
    # Clean tool name if needed
    tool_name = tool.name
    if tool_name.startswith("[Tool]"):
        tool_name = tool_name.replace("[Tool]", "").strip()
    
    updates = {}
    
    # Define a wrapper function for both sync and async invocation
    def log_and_call(func):
        def wrapper(call_args, config=None):
            args_only = call_args.get("args", {})
            log_system(f"▶ {tool_name} called with args: {json.dumps(args_only, indent=2)}")
            sys.stdout.flush()
            
            # Record tool usage statistics
            tool_stats = ToolStats.get_instance()
            tool_stats.record_tool_call(tool_name)
            
            try:
                result = func(call_args, config)
                formatted = format_tool_output(result)
                log_system(f"◀ {tool_name} output: {formatted}")
                sys.stdout.flush()
                
                # Capture solve_model result
                if tool_name == "solve_model":
                    # Create a global variable to store the solution
                    wrap_tool.mem_solution = formatted
                    print(f"DEBUG: Captured solve_model result: {formatted[:100]}...", file=sys.stderr)
                    
                    # We also want to capture the model state right after solve_model is called
                    # Find the get_model tool in the available tools
                    for available_tool in getattr(wrap_tool, "available_tools", []):
                        if available_tool.name == "get_model":
                            try:
                                # Call get_model right after solve_model to capture the model state
                                print("Calling get_model right after solve_model to capture model state", file=sys.stderr)
                                get_result = available_tool.invoke({})
                                get_formatted = format_tool_output(get_result)
                                
                                # Store the model
                                wrap_tool.mem_model = get_formatted
                                    
                                print(f"DEBUG: Captured model after solve_model: {get_formatted[:100]}...", file=sys.stderr)
                                
                                # Record this call in tool stats
                                tool_stats.record_tool_call("get_model")
                            except Exception as e:
                                print(f"Error calling get_model after solve_model: {e}", file=sys.stderr)
                            
                            break
                
                # Capture get_model result
                if tool_name == "get_model":
                    # Create a global variable to store the model
                    wrap_tool.mem_model = formatted
                    print(f"DEBUG: Captured get_model result: {formatted[:100]}...", file=sys.stderr)
                
                return result
            except Exception as e:
                error_msg = f"Tool execution failed: {str(e)}"
                log_system(f"✖ Error: {error_msg}")
                sys.stdout.flush()
                return ToolError(error_msg)
        return wrapper
    
    # Store this tool in the available_tools list for cross-tool access
    if not hasattr(wrap_tool, "available_tools"):
        wrap_tool.available_tools = []
    wrap_tool.available_tools.append(tool)
    
    # Apply wrappers to sync and async methods if they exist
    if hasattr(tool, "invoke"):
        updates["invoke"] = log_and_call(tool.invoke)
    
    if hasattr(tool, "ainvoke"):
        orig_ainvoke = tool.ainvoke
        async def ainvoke_wrapper(call_args, config=None):
            args_only = call_args.get("args", {})
            log_system(f"▶ {tool_name} called with args: {json.dumps(args_only, indent=2)}")
            sys.stdout.flush()
            
            # Record tool usage statistics
            tool_stats = ToolStats.get_instance()
            tool_stats.record_tool_call(tool_name)
            
            try:
                result = await orig_ainvoke(call_args, config)
                formatted = format_tool_output(result)
                log_system(f"◀ {tool_name} output: {formatted}")
                sys.stdout.flush()
                
                # Capture solve_model result
                if tool_name == "solve_model":
                    # Create a global variable to store the solution
                    wrap_tool.mem_solution = formatted
                    print(f"DEBUG: Captured solve_model result: {formatted[:100]}...", file=sys.stderr)
                    
                    # We also want to capture the model state right after solve_model is called
                    # Find the get_model tool in the available tools
                    for available_tool in getattr(wrap_tool, "available_tools", []):
                        if available_tool.name == "get_model":
                            try:
                                # Call get_model right after solve_model to capture the model state
                                print("Calling get_model right after solve_model to capture model state", file=sys.stderr)
                                if hasattr(available_tool, "ainvoke"):
                                    get_result = await available_tool.ainvoke({})
                                else:
                                    get_result = available_tool.invoke({})
                                    
                                get_formatted = format_tool_output(get_result)
                                
                                # Store the model
                                wrap_tool.mem_model = get_formatted
                                    
                                print(f"DEBUG: Captured model after solve_model: {get_formatted[:100]}...", file=sys.stderr)
                                
                                # Record this call in tool stats
                                tool_stats.record_tool_call("get_model")
                            except Exception as e:
                                print(f"Error calling get_model after solve_model: {e}", file=sys.stderr)
                            
                            break
                
                # Capture get_model result
                if tool_name == "get_model":
                    # Create a global variable to store the model
                    wrap_tool.mem_model = formatted
                    print(f"DEBUG: Captured get_model result: {formatted[:100]}...", file=sys.stderr)
                
                return result
            except Exception as e:
                error_msg = f"Tool execution failed: {str(e)}"
                log_system(f"✖ Error: {error_msg}")
                sys.stdout.flush()
                return ToolError(error_msg)
        updates["ainvoke"] = ainvoke_wrapper
    
    if updates:
        return tool.copy(update=updates)
    else:
        log_system(f"Warning: {tool_name} has no invoke or ainvoke method; cannot wrap for logging.")
        return tool

def parse_arguments():
    """Parse command line arguments."""
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
    parser.add_argument(
        '--no-stats',
        action='store_true',
        help='Disable tool usage statistics'
    )
    parser.add_argument(
        '--recursion-limit',
        type=int,
        help='Set the recursion limit for the agent'
    )
    return parser.parse_args()

async def mcp_solver_node(state: dict, model_name: str) -> dict:
    """Processes the conversation via the MCP solver with direct tool calling."""
    state["solver_visit_count"] = state.get("solver_visit_count", 0) + 1
    
    # Initialize mem_solution and mem_model if they don't exist
    if "mem_solution" not in state:
        state["mem_solution"] = "No solution generated yet"
    
    if "mem_model" not in state:
        state["mem_model"] = "No model captured yet"

    # Get the model name
    SOLVE_MODEL = model_name.lower()
    print(f"Using model: {SOLVE_MODEL}")
    
    # Extract args from state if available, otherwise use None
    args = state.get("args")
    
    # Set up the connection to the MCP server
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

    # Set up server command and args
    if state.get("server_command") and state.get("server_args"):
        mcp_command = state["server_command"]
        mcp_args = state["server_args"]
        print(f"Using custom server command: {mcp_command} {' '.join(mcp_args)}", flush=True)
    else:
        mcp_command = DEFAULT_SERVER_COMMAND
        mcp_args = DEFAULT_SERVER_ARGS
        print(f"Using default server command: {mcp_command} {' '.join(mcp_args)}", flush=True)

    # Set up server parameters for stdio connection
    server_params = StdioServerParameters(command=mcp_command, args=mcp_args)

    try:
        # Create a direct client session and initialize MCP tools
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                try:
                    # Initialize the connection and get tools
                    await session.initialize()
                    raw_tools = await load_mcp_tools(session)
                    wrapped_tools = [wrap_tool(tool) for tool in raw_tools]
                    
                    # Print tools for debugging
                    print("\n📋 Available tools:", flush=True)
                    for i, tool in enumerate(wrapped_tools):
                        print(f"  {i+1}. {tool.name}", flush=True)
                    print("", flush=True)
                    sys.stdout.flush()
                    
                    # Initialize the agent with tools
                    if USE_CUSTOM_AGENT:
                        print("Using custom React agent implementation...")
                        # Extract system prompt from the messages list if present
                        system_prompt = None
                        for msg in state["messages"]:
                            if msg.get("role") == "system":
                                system_prompt = msg.get("content")
                                print(f"Using system prompt with {len(system_prompt)} characters")
                                break
                                
                        if not system_prompt:
                            print("Warning: No system prompt found in messages!")
                            
                        agent = create_custom_react_agent(SOLVE_MODEL, wrapped_tools, system_prompt)
                    else:
                        print("Using built-in React agent implementation...")
                        agent = create_react_agent(SOLVE_MODEL, wrapped_tools)
                    
                    # Get recursion_limit from args (if provided), then from config, or default to 200
                    recursion_limit = args.recursion_limit if args is not None else None
                    if recursion_limit is None:
                        # Try to get from config
                        try:
                            import tomli
                            with open(os.path.join(os.path.dirname(__file__), "../../../pyproject.toml"), "rb") as f:
                                config_data = tomli.load(f)
                            recursion_limit = config_data.get("tool", {}).get("test_client", {}).get("recursion_limit", 200)
                        except Exception:
                            recursion_limit = 200
                    
                    print(f"Using recursion limit: {recursion_limit}")
                    config = RunnableConfig(recursion_limit=recursion_limit)

                    # Process the request
                    print(f"\n{'='*60}")
                    print("Sending request to LLM...")
                    print(f"{'='*60}\n")
                    sys.stdout.flush()
                    
                    try:
                        # Execute the agent
                        if USE_CUSTOM_AGENT:
                            print("Executing custom React agent...")
                            # Create initial state with messages
                            human_message = HumanMessage(content=state["messages"][-1]["content"])
                            
                            try:
                                # Run the custom agent
                                final_state = await run_agent(agent, human_message.content, config)
                                
                                # Normalize the state for consistent format handling
                                if USING_NEW_LANGGRAPH:
                                    final_state = normalize_state(final_state)
                                
                                # Extract and add the agent's response to state
                                if isinstance(final_state, dict) and final_state.get("messages") and len(final_state["messages"]) > 0:
                                    last_message = final_state["messages"][-1]
                                    if hasattr(last_message, "content"):
                                        agent_reply = last_message.content
                                    else:
                                        agent_reply = str(last_message)
                                    print(f"Agent reply received, length: {len(str(agent_reply))}", flush=True)
                                    state["messages"].append({"role": "assistant", "content": agent_reply})
                                else:
                                    print("Warning: No message content found in custom agent response", flush=True)
                            except Exception as e:
                                error_msg = f"Error running custom agent: {str(e)}"
                                print(error_msg, flush=True)
                                state["messages"].append({"role": "assistant", "content": error_msg})
                        else:
                            print("Executing built-in React agent implementation...")
                            # Execute the built-in agent
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

    # Check if solve_model was called and update mem_solution
    if hasattr(wrap_tool, "mem_solution"):
        state["mem_solution"] = wrap_tool.mem_solution
        print(f"Updated state mem_solution with solve_model result", file=sys.stderr)

    # Update state with any get_model results we captured during execution
    if hasattr(wrap_tool, "mem_model"):
        state["mem_model"] = wrap_tool.mem_model
        print(f"Updated state mem_model with get_model result", file=sys.stderr)
    
    return state

async def main():
    """Main entry point for the client."""
    # Parse arguments
    args = parse_arguments()
    
    # Configure tool stats
    tool_stats_enabled = not args.no_stats
    tool_stats = ToolStats.get_instance()
    tool_stats.enabled = tool_stats_enabled
    
    # Load initial state with custom prompt and query
    try:
        state = load_initial_state(args.prompt, args.query)
    except Exception as e:
        console.print(f"[bold red]Error loading files: {str(e)}[/bold red]")
        sys.exit(1)
    
    # Store args in state for later use
    state["args"] = args
    
    # Initialize mem_solution and mem_model
    state["mem_solution"] = "No solution generated yet"
    state["mem_model"] = "No model captured yet"
    
    # If server command is provided, parse it into command and args
    if args.server:
        command_parts = args.server.split()
        state["server_command"] = command_parts[0]
        state["server_args"] = command_parts[1:] if len(command_parts) > 1 else []
    
    # Run the solver once
    await mcp_solver_node(state, args.model)
    
    # Return the state for printing in main_cli
    return state

def main_cli():
    """Command line entrypoint."""
    try:
        state = asyncio.run(main())
        
        # After the main function completes, print tool usage statistics
        tool_stats = ToolStats.get_instance()
        tool_stats.print_stats()
        
        # Print mem_solution if available
        if state and isinstance(state, dict) and "mem_solution" in state:
            print("\n" + "="*60)
            print("SOLUTION RESULT:")
            print("="*60)
            print(state["mem_solution"])
            print("="*60 + "\n")
        
        # Print mem_model if available
        if state and isinstance(state, dict) and "mem_model" in state:
            print("\n" + "="*60)
            print("FINAL MODEL:")
            print("="*60)
            print(state["mem_model"])
            print("="*60 + "\n")
        
        return 0
    except KeyboardInterrupt:
        print("\nOperation interrupted by user", file=sys.stderr)
        
        # Still try to print statistics in case of interruption
        try:
            tool_stats = ToolStats.get_instance()
            tool_stats.print_stats()
        except Exception:
            pass
            
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1

if __name__ == "__main__":
    main_cli() 