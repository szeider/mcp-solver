import sys
import os
import asyncio
import argparse
import json
import re
import textwrap
from datetime import datetime
from typing import Dict, Any, List

# Core dependencies
from .llm_factory import LLMFactory, ModelInfo
from langchain_mcp_adapters.client import MultiServerMCPClient
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
DEFAULT_SERVER_ARGS = ["run", "mcp-solver", "--lite"]

# Global Rich Console instance with color support
console = Console(color_system="truecolor")
_current_title = None  # Stores the current title for system messages

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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MCP Solver Client')
    parser.add_argument(
        '--query',
        type=str,
        required=True,
        help='Path to the file containing the query'
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
        help='Server command to use instead of defaults. Format: "command arg1 arg2 arg3..."'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=list(MODEL_CODES.keys()),
        default=DEFAULT_MODEL,
        help=f'Model to use (default: {DEFAULT_MODEL}). Available models: {", ".join(MODEL_CODES.keys())}'
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
    if hasattr(tool, "invoke"):
        orig_invoke = tool.invoke

        def new_invoke(call_args, config=None):
            args_only = call_args.get("args", {})
            args_str = json.dumps(args_only, indent=2).strip()
            if args_str.startswith("{") and args_str.endswith("}"):
                args_str = args_str[1:-1].strip()
            set_system_title(f"tool: {tool_name}")
            log_system(f"{tool_name} called with args: {args_str}")
            result = orig_invoke(call_args, config)
            formatted = format_tool_output(result)
            log_system(f"{tool_name} output: {formatted}")
            return result

        updates["invoke"] = new_invoke
    if hasattr(tool, "ainvoke"):
        orig_ainvoke = tool.ainvoke

        async def new_ainvoke(call_args, config=None):
            args_only = call_args.get("args", {})
            args_str = json.dumps(args_only, indent=2).strip()
            if args_str.startswith("{") and args_str.endswith("}"):
                args_str = args_str[1:-1].strip()
            set_system_title(f"TOOL: {tool_name}")
            log_system(f"{tool_name} called with args: {args_str}")
            result = await orig_ainvoke(call_args, config)
            formatted = format_tool_output(result)
            log_system(f"{tool_name} output: {formatted}")
            return result

        updates["ainvoke"] = new_ainvoke
    if updates:
        return tool.copy(update=updates)
    else:
        set_system_title("warning")
        log_system(f"Warning: {tool_name} has no invoke or ainvoke method; cannot wrap for logging.")
        return tool

async def mcp_solver_node(state: dict, model_name: str) -> dict:
    """Processes the conversation via the MCP solver with direct tool calling."""
    state["solver_visit_count"] = state.get("solver_visit_count", 0) + 1

    # Initialize these keys so they're always available
    if "get_model_result" not in state:
        state["get_model_result"] = "No model generated yet"

    if "solve_model_result" not in state:
        state["solve_model_result"] = "No solution generated yet"

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
            state["solver_error"] = True
            return state
    except Exception as e:
        console.print(f"[bold red]Error parsing model code: {str(e)}[/bold red]")
        state["solver_error"] = True
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
        async with MultiServerMCPClient() as client:
            await client.connect_to_server(
                server_name="MCP Solver",
                command=mcp_command,
                args=mcp_args,
            )

            # Get all tools and wrap them for the agent
            tools = client.get_tools()
            wrapped_tools = [wrap_tool(tool) for tool in tools]
            
            # Print tools for debugging
            print(f"Available tools ({len(wrapped_tools)}):")
            for i, tool in enumerate(wrapped_tools):
                print(f"  {i+1}. {tool.name}")

            # Tracking function that can be attached to any tool
            def track_tool_call(tool_name, tool_input, tool_output):
                if tool_name == "get_model":
                    state["get_model_result"] = getattr(tool_output, "content", str(tool_output))

                    if "Empty Model" in str(tool_output) or "Model is empty" in str(tool_output):
                        state["empty_model_error"] = True
                        print("WARNING: Model appears to be empty or incomplete")

                elif tool_name == "solve_model":
                    state["solve_model_result"] = getattr(tool_output, "content", str(tool_output))
                    # Mark that we've attempted to solve the model
                    state["solve_model_called"] = True

                    # Check for errors
                    if isinstance(tool_output, dict) and "error" in tool_output:
                        error_code = tool_output.get("error", {}).get("code", "")
                        error_msg = tool_output.get("error", {}).get("message", "")
                        state["solver_error"] = True
                        state["solver_error_code"] = error_code
                        state["solver_error_message"] = error_msg
                        print(f"Solver error: {error_code} - {error_msg}")

                    elif isinstance(tool_output, dict):
                        if "status" in tool_output and tool_output["status"] == "SAT":
                            if "solution" in tool_output:
                                solution = tool_output["solution"]
                                state["solution"] = solution
                                # Flag as valid solution
                                state["sat_solution_found"] = True
                                print("SAT solution found! This solution satisfies all constraints.")

            # Custom callback handler for tool tracking
            class SimpleToolTracker(BaseCallbackHandler):
                def on_tool_end(self, output, **kwargs):
                    tool_name = kwargs.get("name", "unknown_tool")
                    tool_input = kwargs.get("input", {})
                    track_tool_call(tool_name, tool_input, output)

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
                
                # Set error flag
                state["solver_error"] = True
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
    asyncio.run(main())

if __name__ == "__main__":
    main_cli() 