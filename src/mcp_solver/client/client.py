import argparse
import asyncio
import json
import os
import re
import sys
import traceback
from datetime import datetime
from string import Template
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig

# LangGraph agent implementation
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from rich.console import Console

from mcp_solver.client.mcp_tool_adapter import load_mcp_tools
from mcp_solver.core.prompt_loader import load_prompt

# Core dependencies
from .llm_factory import LLMFactory
from .token_callback import TokenUsageCallbackHandler
from .token_counter import TokenCounter

# Import tool stats tracking
from .tool_stats import ToolStats


def format_token_count(count):
    """
    Format token count with compact magnitude representation using perceptually
    significant digits.

    Parameters:
        count (int): The token count to format

    Returns:
        str: Formatted representation with appropriate magnitude suffix
    """
    if count < 1000:
        return str(count)
    elif count < 1000000:
        # Scale to thousands with conditional precision
        scaled = count / 1000
        if scaled < 10:
            # For 1k-9.9k, maintain single decimal precision
            return f"{scaled:.1f}".rstrip("0").rstrip(".") + "k"
        else:
            # For ≥10k, use integer representation
            return f"{int(scaled)}k"
    else:
        # Scale to millions with conditional precision
        scaled = count / 1000000
        if scaled < 10:
            # For 1M-9.9M, maintain single decimal precision
            return f"{scaled:.1f}".rstrip("0").rstrip(".") + "M"
        else:
            # For ≥10M, use integer representation
            return f"{int(scaled)}M"


# Default model to use
DEFAULT_MODEL = "AT:claude-sonnet-4-20250514"  # Anthropic Claude Sonnet 4

# Default server configuration
DEFAULT_SERVER_COMMAND = "uv"
DEFAULT_SERVER_ARGS = ["run", "mcp-solver-mzn"]

# Mode-specific server arguments
MODE_SERVER_ARGS = {
    "mzn": ["run", "mcp-solver-mzn"],
    "z3": ["run", "mcp-solver-z3"],
    "pysat": ["run", "mcp-solver-pysat"],
    "maxsat": ["run", "mcp-solver-maxsat"],
    "asp": ["run", "mcp-solver-asp"],
}

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
    console.file.flush() if hasattr(console, "file") else None


class ClientError(Exception):
    """Client related errors."""

    pass


def load_file_content(file_path):
    """Load content from a file."""
    try:
        with open(file_path, encoding="utf-8") as file:
            return file.read().strip()
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        sys.exit(1)


def load_initial_state(query_path, mode):
    """Initialize state with the instructions/review prompts and the query."""
    # Load query
    query = load_file_content(query_path)

    try:
        # Load both required prompts using the centralized loader
        instructions_prompt = load_prompt(mode, "instructions")
        review_prompt = load_prompt(mode, "review")

        print(f"Loaded prompts for {mode} mode:")
        print(f"- Instructions: {len(instructions_prompt)} characters")
        print(f"- Review: {len(review_prompt)} characters")
    except Exception as e:
        console.print(f"[bold red]Error loading prompts: {e!s}[/bold red]")
        sys.exit(1)

    # Create initial messages with ONLY the instructions prompt as system message
    messages = [
        {"role": "system", "content": instructions_prompt},
        {"role": "user", "content": query},
    ]

    # Return state with review_prompt stored separately for future use
    return {
        "messages": messages,
        "review_prompt": review_prompt,  # Store for future use but don't use in agent messages
        "start_time": datetime.now(),
        "mem_problem": query,
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
            log_system(
                f"▶ {tool_name} called with args: {json.dumps(args_only, indent=2)}"
            )
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

                    # We also want to capture the model state right after solve_model is called
                    # Find the get_model tool in the available tools
                    for available_tool in getattr(wrap_tool, "available_tools", []):
                        if available_tool.name == "get_model":
                            try:
                                # Call get_model right after solve_model to capture the model state
                                get_result = available_tool.invoke({})
                                get_formatted = format_tool_output(get_result)

                                # Store the model
                                wrap_tool.mem_model = get_formatted

                                # Record this call in tool stats
                                tool_stats.record_tool_call("get_model")
                            except Exception as e:
                                log_system(
                                    f"Warning: Failed to capture model after solve: {e!s}"
                                )
                                pass  # Handle errors in get_model capture

                            break

                    # Capture get_model result
                    if tool_name == "get_model":
                        # Create a global variable to store the model
                        wrap_tool.mem_model = formatted

                return result
            except Exception as e:
                error_msg = f"Tool execution failed: {e!s}"
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
            log_system(
                f"▶ {tool_name} called with args: {json.dumps(args_only, indent=2)}"
            )
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

                    # We also want to capture the model state right after solve_model is called
                    # Find the get_model tool in the available tools
                    for available_tool in getattr(wrap_tool, "available_tools", []):
                        if available_tool.name == "get_model":
                            try:
                                # Call get_model right after solve_model to capture the model state
                                if hasattr(available_tool, "ainvoke"):
                                    get_result = await available_tool.ainvoke({})
                                else:
                                    get_result = available_tool.invoke({})

                                get_formatted = format_tool_output(get_result)

                                # Store the model
                                wrap_tool.mem_model = get_formatted

                                # Record this call in tool stats
                                tool_stats.record_tool_call("get_model")
                            except Exception as e:
                                log_system(
                                    f"Warning: Failed to capture model after solve: {e!s}"
                                )
                                pass  # Handle errors in get_model capture

                            break

                    # Capture get_model result
                    if tool_name == "get_model":
                        # Create a global variable to store the model
                        wrap_tool.mem_model = formatted

                return result
            except Exception as e:
                error_msg = f"Tool execution failed: {e!s}"
                log_system(f"✖ Error: {error_msg}")
                sys.stdout.flush()
                return ToolError(error_msg)

        updates["ainvoke"] = ainvoke_wrapper

    if updates:
        return tool.copy(update=updates)
    else:
        log_system(
            f"Warning: {tool_name} has no invoke or ainvoke method; cannot wrap for logging."
        )
        return tool


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MCP Solver Client")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Path to the file containing the problem query",
    )
    parser.add_argument(
        "--server",
        type=str,
        help='Server command to use. Format: "command arg1 arg2 arg3..."',
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["mzn", "z3", "pysat", "maxsat", "asp"],
        help="Solver mode to use (overrides automatic detection)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model code to use (default: {DEFAULT_MODEL}). Format: '<platform>:<model>' or '<platform>:<provider>/<model>'. "
        "Examples: 'OA:gpt-4.1-2025-04-14', 'AT:claude-3-7-sonnet-20250219', 'OR:google/gemini-2.5-pro-preview'",
    )
    parser.add_argument(
        "--mc",
        type=str,
        help="Alternative way to specify model code. Same format as --model. "
        "Supported platforms: OR (OpenRouter), AT (Anthropic), OA (OpenAI), GO (Google), LM (LM Studio). "
        "For LM Studio use format 'LM:model@url'. Overrides --model if provided.",
    )
    parser.add_argument(
        "--no-stats", action="store_true", help="Disable tool usage statistics"
    )
    parser.add_argument(
        "--recursion-limit", type=int, help="Set the recursion limit for the agent"
    )
    parser.add_argument(
        "--result-path",
        type=str,
        help="Path to save JSON results file (includes directory and filename)",
    )
    return parser.parse_args()


def call_reviewer(state: dict, model: Any) -> dict:
    """Call a reviewer to assess the solution.

    This function takes the current state, which includes the solution,
    and calls a reviewer to assess its correctness.

    Args:
        state: The current state
        model: The language model to use

    Returns:
        Updated state with review result added
    """
    print("Starting solution review process...", flush=True)

    # Extract necessary information from state
    mem_problem = state.get("mem_problem", "No problem statement provided")
    mem_model = state.get("mem_model", "No model code provided")
    solution = state.get("mem_solution", "No solution generated yet")

    # Process the solution to make it more readable if it's in JSON format
    processed_solution = solution
    try:
        if isinstance(solution, str) and "solution" in solution:
            # Try using ast.literal_eval which is safer than eval for parsing Python literals
            import ast

            try:
                # Convert the string representation of a dict to an actual dict
                solution_dict = ast.literal_eval(solution)

                if isinstance(solution_dict, dict) and "solution" in solution_dict:
                    solution_values = solution_dict["solution"]

                    processed_solution = "Solution values found:\n"
                    for var, value in solution_values.items():
                        processed_solution += f"{var} = {value}\n"

                    # Add a clear statement about satisfiability
                    if solution_dict.get("satisfiable", False):
                        processed_solution += "\nThe model is satisfiable."
                    else:
                        processed_solution += "\nThe model is NOT satisfiable."
            except (SyntaxError, ValueError):
                # If literal_eval fails, try regex approach
                import json

                # Extract the solution part
                solution_match = re.search(r"'solution'\s*:\s*({[^}]+})", solution)
                if solution_match:
                    try:
                        # Try to parse just the solution part
                        solution_part = solution_match.group(1).replace("'", '"')
                        solution_values = json.loads(solution_part)
                        processed_solution = "Solution values found:\n"
                        for var, value in solution_values.items():
                            processed_solution += f"{var} = {value}\n"

                        # Check for satisfiability
                        if "'satisfiable': True" in solution:
                            processed_solution += "\nThe model is satisfiable."
                    except json.JSONDecodeError:
                        # If JSON parsing fails, extract manually
                        value_matches = re.findall(
                            r"'([^']+)':\s*(\d+)", solution_match.group(1)
                        )
                        if value_matches:
                            processed_solution = "Solution values found:\n"
                            for var, value in value_matches:
                                processed_solution += f"{var} = {value}\n"
    except Exception as e:
        # Log errors in solution processing but continue
        print(f"Note: Solution preprocessing encountered an issue: {e!s}", flush=True)
        pass

    # Create reviewer prompt using Template
    reviewer_template = Template(state.get("review_prompt", ""))
    reviewer_prompt = reviewer_template.substitute(
        PROBLEM=mem_problem, MODEL=mem_model, SOLUTION=processed_solution
    )

    # For debug, show just a preview of the prompt
    print(f"Preparing review prompt ({len(reviewer_prompt)} characters)...", flush=True)

    try:
        # Get token counter instance
        token_counter = TokenCounter.get_instance()

        # Create a human message with the prompt
        review_message = HumanMessage(content=reviewer_prompt)

        print("Calling model for review...", flush=True)

        # Create a callback handler for the reviewer
        reviewer_callback = TokenUsageCallbackHandler(
            token_counter, agent_type="reviewer"
        )

        # Invoke the model with the callback handler
        response = model.invoke(
            [review_message], config={"callbacks": [reviewer_callback]}
        )
        response_text = response.content

        # Track reviewer tokens (fallback for providers without exact counts)
        if not token_counter.reviewer_is_exact:
            token_counter.count_reviewer_input([review_message])
            token_counter.count_reviewer_output([response])

        # Parse the verdict from the response using regex
        verdict_match = re.search(
            r"<verdict>(correct|incorrect|unknown)</verdict>",
            response_text,
            re.IGNORECASE,
        )

        if verdict_match:
            # Extract the verdict
            verdict = verdict_match.group(1).lower()
            print(f"Found verdict tag: {verdict}", flush=True)

            # Get the explanation (everything before the verdict tag)
            explanation_parts = response_text.split("<verdict>")
            explanation = explanation_parts[0].strip()

            # Create the review result
            review_result = {"correctness": verdict, "explanation": explanation}
        else:
            # No explicit verdict tag, use "unknown"
            verdict = "unknown"
            print(f"No verdict tag found, using default: {verdict}", flush=True)

            # Use the full response as the explanation
            review_result = {"correctness": verdict, "explanation": response_text}

        # Store extracted values
        mem_review_verdict = review_result["correctness"]
        mem_review_text = response_text

        print(f"Review complete: verdict is '{mem_review_verdict}'", flush=True)

        # Return both the review result and keep existing messages
        return {
            "review_result": review_result,
            "mem_review_verdict": mem_review_verdict,
            "mem_review_text": mem_review_text,
            "messages": state.get("messages", []),  # Preserve existing messages
        }

    except Exception as e:
        print(f"Review process error: {e!s}", flush=True)
        import traceback

        print(f"Traceback: {traceback.format_exc()}", flush=True)

        # Return a default review result and keep existing messages
        return {
            "review_result": {
                "correctness": "unknown",
                "explanation": f"Error running reviewer: {e!s}",
            },
            "mem_review_verdict": "unknown",
            "mem_review_text": f"Error running reviewer: {e!s}",
            "messages": state.get("messages", []),  # Preserve existing messages
        }


async def mcp_solver_node(state: dict, model_code: str) -> dict:
    """Processes the conversation via the MCP solver with direct tool calling."""
    state["solver_visit_count"] = state.get("solver_visit_count", 0) + 1

    # Initialize mem_solution and mem_model if they don't exist
    if "mem_solution" not in state:
        state["mem_solution"] = "No solution generated yet"

    if "mem_model" not in state:
        state["mem_model"] = "No model captured yet"

    # Use the model code directly
    print(f"Using model code: {model_code}")

    # Extract args from state if available, otherwise use None
    args = state.get("args")

    # Set up the connection to the MCP server
    console.print(f"[bold green]Using model code: {model_code}[/bold green]")

    # Check if required API key is present
    try:
        model_info = LLMFactory.parse_model_code(model_code)

        # LM Studio doesn't require an API key
        if model_info.platform == "LM":
            # Skip API key check for LM Studio
            pass
        else:
            api_key_name = model_info.api_key_name
            api_key = os.environ.get(api_key_name)

            if not api_key:
                error_msg = f"Error: {api_key_name} not found in environment variables. Please set {api_key_name} in your .env file."
                console.print(f"[bold red]{error_msg}[/bold red]")
                state["messages"].append({"role": "assistant", "content": error_msg})
                return state
    except Exception as e:
        console.print(f"[bold red]Error parsing model code: {e!s}[/bold red]")
        state["messages"].append(
            {"role": "assistant", "content": f"Error parsing model code: {e!s}"}
        )
        return state

    # Create model and get model info
    SOLVE_MODEL = LLMFactory.create_model(model_code)
    model_info = LLMFactory.get_model_info(SOLVE_MODEL)
    model_str = (
        f"{model_info.platform}:{model_info.model_name}" if model_info else "Unknown"
    )

    state["solve_llm"] = model_str
    print(f"Using model: {model_str}", flush=True)

    # Set up server command and args
    if state.get("server_command") and state.get("server_args"):
        # Custom server command specified via --server argument
        mcp_command = state["server_command"]
        mcp_args = state["server_args"]
        print(
            f"Using custom server command: {mcp_command} {' '.join(mcp_args)}",
            flush=True,
        )
    else:
        # Use mode-specific server args
        mode = state.get("mode", "mzn")
        mcp_command = DEFAULT_SERVER_COMMAND
        mcp_args = MODE_SERVER_ARGS.get(mode, DEFAULT_SERVER_ARGS)
        print(
            f"Using {mode} mode server command: {mcp_command} {' '.join(mcp_args)}",
            flush=True,
        )

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
                        print(f"  {i + 1}. {tool.name}", flush=True)
                    print("", flush=True)
                    sys.stdout.flush()

                    # Get recursion_limit from args (if provided), then from config, or default to 200
                    recursion_limit = (
                        args.recursion_limit
                        if args is not None and hasattr(args, "recursion_limit")
                        else None
                    )
                    if recursion_limit is None:
                        # Try to get from config
                        try:
                            import tomli

                            with open(
                                os.path.join(
                                    os.path.dirname(__file__), "../../../pyproject.toml"
                                ),
                                "rb",
                            ) as f:
                                config_data = tomli.load(f)
                            recursion_limit = (
                                config_data.get("tool", {})
                                .get("test_client", {})
                                .get("recursion_limit", 200)
                            )
                        except Exception:
                            recursion_limit = 200

                    print(f"Using recursion limit: {recursion_limit}", flush=True)

                    # Ensure token counter is initialized before agent runs
                    token_counter = TokenCounter.get_instance()
                    # Do not reset token counts - this prevents proper tracking
                    # token_counter.main_input_tokens = 0
                    # token_counter.main_output_tokens = 0

                    # Create a callback handler for the main agent
                    main_callback = TokenUsageCallbackHandler(
                        token_counter, agent_type="main"
                    )

                    # Create config with callbacks
                    config = RunnableConfig(
                        recursion_limit=recursion_limit, callbacks=[main_callback]
                    )

                    # Simplified agent start message
                    log_system("Entering ReAct Agent")
                    sys.stdout.flush()

                    try:
                        # Initialize the agent with tools
                        agent = create_react_agent(SOLVE_MODEL, wrapped_tools)

                        # Execute the agent
                        response = await agent.ainvoke(
                            {"messages": state["messages"]}, config=config
                        )

                        # Debug output - print response structure
                        print(
                            f"Agent response keys: {list(response.keys())}", flush=True
                        )

                        # Extract and add the agent's response to state
                        if response.get("messages") and len(response["messages"]) > 0:
                            agent_reply = response["messages"][-1].content
                            print(
                                f"Agent reply received, length: {len(agent_reply)}",
                                flush=True,
                            )
                            state["messages"].append(
                                {"role": "assistant", "content": agent_reply}
                            )

                            # Pass all relevant state updates from the response to the state object
                            # This ensures token counts and other metadata are preserved
                            for key, value in response.items():
                                if key != "messages" and key.startswith("mem_"):
                                    state[key] = value
                                    # Debug print only if needed
                                    # print(f"State update: {key} = {value}", flush=True)

                            # If we didn't get explicit token counts and we don't have exact counts from callbacks, estimate them
                            if (
                                not token_counter.main_is_exact
                                and token_counter.main_input_tokens == 0
                                and response.get("messages")
                            ):
                                # Estimate tokens from input messages
                                # Don't print annoying message: print("No token counts found, estimating from messages...", flush=True)
                                input_msgs = [
                                    msg
                                    for msg in response.get("messages", [])
                                    if not isinstance(msg, AIMessage)
                                ]
                                token_counter.count_main_input(input_msgs)

                                # Estimate tokens from output messages
                                output_msgs = [
                                    msg
                                    for msg in response.get("messages", [])
                                    if isinstance(msg, AIMessage)
                                ]
                                token_counter.count_main_output(output_msgs)
                        else:
                            print(
                                "Warning: No message content found in response",
                                flush=True,
                            )
                    except Exception as e:
                        error_msg = f"Error during LLM invocation: {e!s}"
                        print(error_msg, flush=True)
                        state["messages"].append(
                            {
                                "role": "assistant",
                                "content": f"I encountered an error while processing your request: {e!s}. Please try again with a simpler query or check the model.",
                            }
                        )
                except Exception as e:
                    error_msg = f"Error initializing MCP session: {e!s}"
                    console.print(f"[bold red]{error_msg}[/bold red]")
                    state["messages"].append(
                        {"role": "assistant", "content": error_msg}
                    )
    except Exception as e:
        error_msg = f"Error connecting to MCP server: {e!s}"
        console.print(f"[bold red]{error_msg}[/bold red]")
        state["messages"].append({"role": "assistant", "content": error_msg})

    # Check if solve_model was called and update mem_solution
    if hasattr(wrap_tool, "mem_solution"):
        state["mem_solution"] = wrap_tool.mem_solution

    # Update state with any get_model results we captured during execution
    if hasattr(wrap_tool, "mem_model"):
        state["mem_model"] = wrap_tool.mem_model

    # Always print the current state's mem_solution before review
    # This ensures the value is available in the logs even if tools were never called
    # print(f"mem_solution: {state.get('mem_solution', 'No solution generated yet')}")
    # sys.stdout.flush()

    # Display problem, model, and result before reviewing
    display_problem_model_result(state)

    # Now that we have the updated state, run the reviewer directly
    # This ensures that the reviewer has access to the current state with the correct model and solution
    if SOLVE_MODEL and state.get("review_prompt"):
        try:
            # Simplified reviewer start message
            log_system("Entering Review Agent")

            # Create a state object that mimics the AgentState expected by call_reviewer
            reviewer_state = {
                "mem_problem": state.get("mem_problem", ""),
                "mem_model": state.get("mem_model", ""),
                "mem_solution": state.get("mem_solution", ""),
                "review_prompt": state.get("review_prompt", ""),
                "messages": state.get("messages", []),
            }

            # Call the reviewer directly
            review_result = call_reviewer(reviewer_state, SOLVE_MODEL)

            # Add the review result to the state
            if isinstance(review_result, dict) and "review_result" in review_result:
                state["review_result"] = review_result["review_result"]
                correctness = review_result["review_result"].get(
                    "correctness", "unknown"
                )
                # Add symbols based on correctness status
                if correctness == "correct":
                    symbol = "✅"  # Green checkmark
                elif correctness == "incorrect":
                    symbol = "❌"  # Red X
                else:
                    symbol = "❓"  # Question mark

                print(f"Review complete: {symbol} {correctness}", flush=True)

                # Display review results immediately after completion
                display_review_result(state)
            else:
                print("Review result: ❓ Unknown format", flush=True)
        except Exception as e:
            print(f"Error running reviewer: {e!s}", flush=True)
            state["review_result"] = {
                "correctness": "unknown",
                "explanation": f"Error running reviewer: {e!s}",
            }

            # Display review results even if there was an error
            display_review_result(state)

    return state


def display_problem_model_result(state):
    """Display problem, model, and result output."""
    # Print mem_problem if available
    if isinstance(state, dict) and "mem_problem" in state:
        print("\n" + "=" * 60)
        print("PROBLEM STATEMENT:")
        print("=" * 60)
        print(state["mem_problem"])
        print("=" * 60 + "\n")

    # Print mem_model if available
    if isinstance(state, dict) and "mem_model" in state:
        print("\n" + "=" * 60)
        print("FINAL MODEL:")
        print("=" * 60)
        print(state["mem_model"])
        print("=" * 60 + "\n")

    # Print mem_solution if available
    if isinstance(state, dict) and "mem_solution" in state:
        print("\n" + "=" * 60)
        print("SOLUTION RESULT:")
        print("=" * 60)
        print(state["mem_solution"])
        print("=" * 60 + "\n")


def display_review_result(state):
    """Display review result output."""
    # Print review_result if available
    if isinstance(state, dict) and "review_result" in state:
        print("\n" + "=" * 60)
        print("REVIEW RESULT:")
        print("=" * 60)
        correctness = state["review_result"].get("correctness", "unknown")
        explanation = state["review_result"].get(
            "explanation", "No explanation provided"
        )

        # Add symbols based on correctness status
        if correctness == "correct":
            symbol = "✅"  # Green checkmark
        elif correctness == "incorrect":
            symbol = "❌"  # Red X
        else:
            symbol = "❓"  # Question mark

        print(f"Correctness: {symbol} {correctness}")
        print(f"Explanation: {explanation}")
        print("=" * 60 + "\n")


def display_combined_stats():
    """Display combined tool usage and token usage statistics."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Get tool statistics
    tool_stats = ToolStats.get_instance()
    if tool_stats.enabled and tool_stats.total_calls > 0:
        tool_table = Table(title="Tool Usage and Token Statistics")

        # Add tool usage section
        tool_table.add_column("Category", style="cyan")
        tool_table.add_column("Item", style="green")
        tool_table.add_column("Count/Value", style="yellow")

        # Add a header row for tool usage
        tool_table.add_row("TOOL USAGE", "", "", style="bold")

        # List of all standard tools
        standard_tools = [
            "clear_model",
            "add_item",
            "replace_item",
            "delete_item",
            "get_model",
            "solve_model",
        ]

        # Make sure all standard tools are represented in tool_calls
        for tool_name in standard_tools:
            if tool_name not in tool_stats.tool_calls:
                tool_stats.tool_calls[tool_name] = 0

        # Sort tools by number of calls (descending)
        sorted_tools = sorted(
            tool_stats.tool_calls.items(), key=lambda x: x[1], reverse=True
        )

        for tool_name, count in sorted_tools:
            tool_table.add_row("Tool", tool_name, str(count))

        # Add a total row for tools
        tool_table.add_row("Tool", "TOTAL", str(tool_stats.total_calls), style="bold")

        # Get token statistics
        token_counter = TokenCounter.get_instance()

        # Add a header row for token usage
        tool_table.add_row("TOKEN USAGE", "", "", style="bold")

        # Main agent tokens
        main_total = token_counter.main_input_tokens + token_counter.main_output_tokens
        main_type = " (exact)" if token_counter.main_is_exact else " (approx)"
        tool_table.add_row(
            "Token",
            f"ReAct Agent Input{main_type}",
            format_token_count(token_counter.main_input_tokens),
        )
        tool_table.add_row(
            "Token",
            f"ReAct Agent Output{main_type}",
            format_token_count(token_counter.main_output_tokens),
        )
        tool_table.add_row(
            "Token",
            f"ReAct Agent Total{main_type}",
            format_token_count(main_total),
            style="bold",
        )

        # Reviewer agent tokens
        reviewer_total = (
            token_counter.reviewer_input_tokens + token_counter.reviewer_output_tokens
        )
        if reviewer_total > 0:
            reviewer_type = (
                " (exact)" if token_counter.reviewer_is_exact else " (approx)"
            )
            tool_table.add_row(
                "Token",
                f"Reviewer Input{reviewer_type}",
                format_token_count(token_counter.reviewer_input_tokens),
            )
            tool_table.add_row(
                "Token",
                f"Reviewer Output{reviewer_type}",
                format_token_count(token_counter.reviewer_output_tokens),
            )
            tool_table.add_row(
                "Token",
                f"Reviewer Total{reviewer_type}",
                format_token_count(reviewer_total),
                style="bold",
            )

        # Grand total
        grand_total = main_total + reviewer_total
        tool_table.add_row(
            "Token", "COMBINED TOTAL", format_token_count(grand_total), style="bold"
        )

        console.print("\n")
        console.print(tool_table)
    else:
        console.print(
            "[yellow]No tool usage or token statistics available for this run.[/yellow]"
        )


def generate_result_json(state, json_path):
    """Generate and save JSON results file from state."""
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # Get token statistics
    token_counter = TokenCounter.get_instance()
    tokens = {
        "react_input": token_counter.main_input_tokens,
        "react_output": token_counter.main_output_tokens,
        "reviewer_input": token_counter.reviewer_input_tokens,
        "reviewer_output": token_counter.reviewer_output_tokens,
    }

    # Get tool usage statistics
    tool_stats = ToolStats.get_instance()
    tool_calls = tool_stats.tool_calls if tool_stats.enabled else {}

    # Get the full explanation from the review result
    review_explanation = ""
    if isinstance(state.get("review_result"), dict):
        review_explanation = state["review_result"].get("explanation", "")

    # If we have the full review text available, use that as a backup
    if not review_explanation and state.get("mem_review_text"):
        # Try to extract just the explanation part (before any verdict tag)
        verdict_match = re.search(
            r"<verdict>(correct|incorrect|unknown)</verdict>",
            state.get("mem_review_text", ""),
            re.IGNORECASE,
        )
        if verdict_match:
            parts = state["mem_review_text"].split("<verdict>")
            review_explanation = parts[0].strip()
        else:
            # Use the full text as explanation
            review_explanation = state.get("mem_review_text", "")

    # Extract the problem name from the query file path
    problem_name = ""
    if state.get("args") and hasattr(state["args"], "query"):
        query_path = state["args"].query
        problem_name = os.path.basename(query_path).replace(".md", "")

    # Prepare JSON data
    json_data = {
        "problem": state.get("mem_problem", ""),
        "model": state.get("mem_model", "No model captured"),
        "solution": state.get("mem_solution", "No solution generated"),
        "review_text": review_explanation,
        "review_verdict": state.get("review_result", {}).get("correctness", "unknown"),
        "result": (
            "correct"
            if state.get("review_result", {}).get("correctness") == "correct"
            else "incorrect"
        ),
        "tool_calls": tool_calls,
        "tokens": tokens,
        "mode": state.get("mode", "unknown"),
        "problem_name": problem_name,
    }

    # Save to file
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
        print(f"\nSaved JSON result to: {json_path}")
    except Exception as e:
        print(f"\nError saving JSON result: {e!s}")


async def main():
    """Main entry point for the client."""
    # Parse arguments
    args = parse_arguments()

    # Configure tool stats
    tool_stats_enabled = not args.no_stats
    tool_stats = ToolStats.get_instance()
    tool_stats.enabled = tool_stats_enabled

    # Initialize token counter (always enabled regardless of --no-stats flag)
    _ = TokenCounter.get_instance()

    # Determine mode from arguments or server command
    if args.mode:
        # Use explicit mode if provided
        mode = args.mode
    elif args.server:
        # Otherwise detect from server command
        mode = "mzn"  # Default to MiniZinc
        server_cmd = args.server.lower()
        if "z3" in server_cmd:
            mode = "z3"
        elif "maxsat" in server_cmd:
            mode = "maxsat"
        elif "pysat" in server_cmd:
            mode = "pysat"
        elif "asp" in server_cmd:
            mode = "asp"
    else:
        # Default to MiniZinc if no mode or server specified
        mode = "mzn"

    print(f"Detected mode: {mode}")

    # Log the selected model code for reference
    print(f"Model code: {args.mc if args.mc else DEFAULT_MODEL}")

    # Load initial state with prompts and query
    try:
        state = load_initial_state(args.query, mode)
    except Exception as e:
        console.print(f"[bold red]Error loading files: {e!s}[/bold red]")
        sys.exit(1)

    # Store args in state for later use
    state["args"] = args

    # Store mode in state for later use in JSON output and server selection
    state["mode"] = mode

    # Initialize mem_solution and mem_model
    state["mem_solution"] = "No solution generated yet"
    state["mem_model"] = "No model captured yet"

    # If server command is provided, parse it into command and args
    if args.server:
        command_parts = args.server.split()
        state["server_command"] = command_parts[0]
        state["server_args"] = command_parts[1:] if len(command_parts) > 1 else []

    # Run the solver once with model code
    model_code = args.mc if args.mc else DEFAULT_MODEL
    await mcp_solver_node(state, model_code)

    # Return the state for printing in main_cli
    return state


def main_cli():
    """Command line entrypoint."""
    try:
        state = asyncio.run(main())

        # Get the parsed arguments to access result_path
        args = parse_arguments()

        # Display statistics at the end
        display_combined_stats()

        # Generate and save JSON if result path provided
        if args.result_path:
            generate_result_json(state, args.result_path)

        return 0
    except KeyboardInterrupt:
        print("\nOperation interrupted by user", file=sys.stderr)

        # Still try to print statistics in case of interruption
        try:
            display_combined_stats()
        except Exception:
            pass

        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    main_cli()
