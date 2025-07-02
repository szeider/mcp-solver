import asyncio
import json
import logging
import sys
from datetime import timedelta
from importlib.metadata import version
from typing import Any

import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.shared.exceptions import McpError

from .constants import (
    ITEM_CHARS,
    MAX_SOLVE_TIMEOUT,
    MIN_SOLVE_TIMEOUT,
)
from .prompt_loader import load_prompt


# Global flags for mode selection
Z3_MODE = False
PYSAT_MODE = False
MAXSAT_MODE = False
ASP_MODE = False

try:
    version_str = version("mcp-solver")
    logging.getLogger(__name__).info(f"Loaded version: {version_str}")
except Exception:
    version_str = "0.0.0"
    logging.getLogger(__name__).warning(
        "Failed to load version from package, using default: 0.0.0"
    )


def format_model_items(
    items: list[tuple[int, str]], max_chars: int | None = None
) -> str:
    """Format model items with optional truncation."""
    if not items:
        return "Model is empty"

    def truncate(text: str) -> str:
        if max_chars is None or len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."

    return "\n".join(f"{i} | {truncate(content)}" for i, content in items)


async def serve() -> None:
    server = Server("mcp-solver")
    server.capabilities = {"prompts": {}}

    # Initialize the appropriate model manager based on mode
    if Z3_MODE:
        from ..z3.model_manager import Z3ModelManager

        model_mgr = Z3ModelManager()
        logging.getLogger(__name__).info("Using Z3 model manager")
    elif PYSAT_MODE:
        from ..pysat.model_manager import PySATModelManager

        model_mgr = PySATModelManager()
        logging.getLogger(__name__).info("Using PySAT model manager")
    elif MAXSAT_MODE:
        from ..maxsat.model_manager import MaxSATModelManager

        model_mgr = MaxSATModelManager()
        logging.getLogger(__name__).info("Using MaxSAT model manager")
    elif ASP_MODE:
        from ..asp.model_manager import ASPModelManager

        model_mgr = ASPModelManager()
        logging.getLogger(__name__).info("Using ASP model manager")
    else:
        from ..mzn.model_manager import MiniZincModelManager

        model_mgr = MiniZincModelManager()
        logging.getLogger(__name__).info("Using MiniZinc model manager")

    # Helper function to get mode-specific descriptions
    def get_description(descriptions: dict) -> str:
        """
        Get the appropriate description based on the current mode.

        Args:
            descriptions: A dictionary of descriptions keyed by mode ('z3', 'pysat', 'maxsat', 'mzn', 'asp')
                          or a string for a common description across all modes

        Returns:
            The appropriate description string for the current mode
        """
        if isinstance(descriptions, str):
            return descriptions

        if Z3_MODE and "z3" in descriptions:
            return descriptions["z3"]
        elif PYSAT_MODE and "pysat" in descriptions:
            return descriptions["pysat"]
        elif MAXSAT_MODE and "maxsat" in descriptions:
            return descriptions["maxsat"]
        elif ASP_MODE and "asp" in descriptions:
            return descriptions["asp"]
        elif (
            not Z3_MODE
            and not PYSAT_MODE
            and not MAXSAT_MODE
            and not ASP_MODE
            and "mzn" in descriptions
        ):
            return descriptions["mzn"]
        elif "default" in descriptions:
            return descriptions["default"]
        else:
            # Return the first available description if no matching mode is found
            return next(iter(descriptions.values()))

    @server.list_prompts()
    async def handle_list_prompts() -> list[types.Prompt]:
        return [
            types.Prompt(
                name="instructions",
                title="MCP Solver Instructions",
                description="Instructions for using the MCP Solver",
            )
        ]

    @server.get_prompt()
    async def handle_get_prompt(
        name: str, arguments: dict[str, str] | None
    ) -> types.GetPromptResult:
        # Choose the appropriate instruction prompt based on mode
        if name == "instructions":
            # Determine mode subfolder
            if Z3_MODE:
                mode_folder = "z3"
            elif PYSAT_MODE:
                mode_folder = "pysat"
            elif MAXSAT_MODE:
                mode_folder = "maxsat"
            elif ASP_MODE:
                mode_folder = "asp"
            else:
                mode_folder = "mzn"

            logging.getLogger(__name__).info(
                f"Loading {name} prompt for {mode_folder} mode"
            )

            try:
                content = load_prompt(mode_folder, "instructions")

                # Add debugging logs
                logging.getLogger(__name__).info("Prompt loaded successfully")
                logging.getLogger(__name__).info(
                    f"Prompt content length: {len(content)}"
                )
                logging.getLogger(__name__).info(
                    f"Prompt content first 100 chars: {content[:100]}"
                )
            except Exception as e:
                error_msg = f"Critical Error: {e!s}"
                logging.getLogger(__name__).error(error_msg)
                raise McpError(error_msg)

            # Return with the new format
            return types.GetPromptResult(
                messages=[
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(type="text", text=content),
                    )
                ]
            )
        else:
            error_msg = f"Unknown prompt: {name}"
            logging.getLogger(__name__).error(error_msg)
            raise McpError(error_msg)

    def format_array_access(variable_name: str, indices: list[int]) -> str:
        return (
            variable_name
            if not indices
            else f"{variable_name}[{','.join(str(i) for i in indices)}]"
        )

    def get_array_value(array: Any, indices: list[int]) -> Any:
        if not indices:
            return array
        if not hasattr(array, "__getitem__"):
            raise ValueError("Variable is not an array")
        try:
            return (
                array[indices[0] - 1]
                if len(indices) == 1
                else get_array_value(array[indices[0] - 1], indices[1:])
            )
        except IndexError:
            raise ValueError(f"Index {indices[0]} is out of bounds")
        except TypeError:
            raise ValueError("Invalid index type")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        # Tools for all modes
        tools = [
            types.Tool(
                name="clear_model",
                description=get_description(
                    {
                        "mzn": "Remove all items from the minizinc model, effectively resetting it.",
                        "z3": "Remove all items from the Z3 Python model, effectively resetting it.",
                        "pysat": "Remove all items from the PySAT Python model, effectively resetting it.",
                        "maxsat": "Remove all items from the MaxSAT optimization model, effectively resetting it.",
                        "asp": "Remove all items from the ASP model, effectively resetting it.",
                    }
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="add_item",
                description=get_description(
                    {
                        "mzn": "Add new minizinc item to the model at a specific index (indices start at 0). Required parameters: 'index' and 'content'.",
                        "z3": "Add new Python code to the Z3 model at a specific index (indices start at 0). Required parameters: 'index' and 'content'.",
                        "pysat": "Add new Python code to the PySAT model at a specific index (indices start at 0). Required parameters: 'index' and 'content'.",
                        "maxsat": "Add new Python code to the MaxSAT optimization model at a specific index (indices start at 0). Required parameters: 'index' and 'content'.",
                        "asp": "Add new ASP item to the model at a specific index (indices start at 0). Required parameters: 'index' and 'content'.",
                    }
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "content": {"type": "string"},
                    },
                    "required": ["index", "content"],
                },
            ),
            types.Tool(
                name="replace_item",
                description=get_description(
                    {
                        "mzn": "Replace an existing item in the minizinc model at a specified index with new content. Required parameters: 'index' and 'content'.",
                        "z3": "Replace an existing item in the Z3 Python model at a specified index with new content. Required parameters: 'index' and 'content'.",
                        "pysat": "Replace an existing item in the PySAT Python model at a specified index with new content. Required parameters: 'index' and 'content'.",
                        "maxsat": "Replace an existing item in the MaxSAT optimization model at a specified index with new content. Required parameters: 'index' and 'content'.",
                        "asp": "Replace an existing item in the ASP model at a specified index with new content. Required parameters: 'index' and 'content'.",
                    }
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "content": {"type": "string"},
                    },
                    "required": ["index", "content"],
                },
            ),
            types.Tool(
                name="delete_item",
                description=get_description(
                    {
                        "mzn": "Delete an item from the minizinc model at the specified index. Required parameter: 'index'.",
                        "z3": "Delete an item from the Z3 Python model at the specified index. Required parameter: 'index'.",
                        "pysat": "Delete an item from the PySAT Python model at the specified index. Required parameter: 'index'.",
                        "maxsat": "Delete an item from the MaxSAT optimization model at the specified index. Required parameter: 'index'.",
                        "asp": "Delete an item from the ASP model at the specified index. Required parameter: 'index'.",
                    }
                ),
                inputSchema={
                    "type": "object",
                    "properties": {"index": {"type": "integer"}},
                    "required": ["index"],
                },
            ),
            types.Tool(
                name="get_model",
                description=get_description(
                    {
                        "mzn": "Fetch the current content of the minizinc model, listing each item with its index.",
                        "z3": "Fetch the current content of the Z3 Python model, listing each item with its index.",
                        "pysat": "Fetch the current content of the PySAT Python model, listing each item with its index.",
                        "maxsat": "Fetch the current content of the MaxSAT optimization model, listing each item with its index.",
                        "asp": "Fetch the current content of the ASP model, listing each item with its index.",
                    }
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="solve_model",
                description=get_description(
                    {
                        "mzn": "Solve the current minizinc model with a timeout parameter. Required parameter: 'timeout'.",
                        "z3": "Solve the current Z3 Python model with a timeout parameter. Required parameter: 'timeout'.",
                        "pysat": "Solve the current PySAT Python model with a timeout parameter. Required parameter: 'timeout'.",
                        "maxsat": "Solve the current MaxSAT optimization model with a timeout parameter. Required parameter: 'timeout'.",
                        "asp": "Solve the current ASP model with a timeout parameter. Required parameter: 'timeout'.",
                    }
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "timeout": {
                            "description": f"Solve timeout in seconds (minimum: {MIN_SOLVE_TIMEOUT.seconds}, maximum: {MAX_SOLVE_TIMEOUT.seconds})",
                            "type": "number",
                        }
                    },
                    "required": ["timeout"],
                },
            ),
        ]
        return tools

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        try:
            match name:
                case "get_model":
                    items = model_mgr.get_model()
                    if not items:
                        return [types.TextContent(type="text", text="Model is empty")]
                    return [
                        types.TextContent(
                            type="text",
                            text="\n".join(f"{i} | {content}" for i, content in items),
                        )
                    ]
                case "add_item":
                    # Check if required parameters are provided
                    if "index" not in arguments:
                        return [
                            types.TextContent(
                                type="text",
                                text="Tool execution failed: Missing required parameter 'index'",
                            )
                        ]
                    if "content" not in arguments:
                        return [
                            types.TextContent(
                                type="text",
                                text="Tool execution failed: Missing required parameter 'content'",
                            )
                        ]

                    result = await model_mgr.add_item(
                        arguments["index"], arguments["content"]
                    )

                    # Check if the operation was successful
                    if result.get("success", True):
                        items = model_mgr.get_model()
                        return [
                            types.TextContent(
                                type="text",
                                text=f"Item added\nCurrent model:\n{format_model_items(items, ITEM_CHARS)}",
                            )
                        ]
                    else:
                        # Return the error message
                        error_msg = result.get("error", "Failed to add item")
                        return [
                            types.TextContent(
                                type="text",
                                text=f"Failed to add item: {error_msg}",
                            )
                        ]
                case "delete_item":
                    if "index" not in arguments:
                        return [
                            types.TextContent(
                                type="text",
                                text="Tool execution failed: Missing required parameter 'index'",
                            )
                        ]

                    result = await model_mgr.delete_item(arguments["index"])

                    # Check if the operation was successful
                    if result.get("success", True):
                        items = model_mgr.get_model()
                        return [
                            types.TextContent(
                                type="text",
                                text=f"Item deleted\nCurrent model:\n{format_model_items(items, ITEM_CHARS)}",
                            )
                        ]
                    else:
                        # Return the error message
                        error_msg = result.get("error", "Failed to delete item")
                        return [
                            types.TextContent(
                                type="text",
                                text=f"Failed to delete item: {error_msg}",
                            )
                        ]
                case "replace_item":
                    # Check if required parameters are provided
                    if "index" not in arguments:
                        return [
                            types.TextContent(
                                type="text",
                                text="Tool execution failed: Missing required parameter 'index'",
                            )
                        ]
                    if "content" not in arguments:
                        return [
                            types.TextContent(
                                type="text",
                                text="Tool execution failed: Missing required parameter 'content'",
                            )
                        ]

                    result = await model_mgr.replace_item(
                        arguments["index"], arguments["content"]
                    )

                    # Check if the operation was successful
                    if result.get("success", True):
                        items = model_mgr.get_model()
                        return [
                            types.TextContent(
                                type="text",
                                text=f"Item replaced\nCurrent model:\n{format_model_items(items, ITEM_CHARS)}",
                            )
                        ]
                    else:
                        # Return the error message
                        error_msg = result.get("error", "Failed to replace item")
                        return [
                            types.TextContent(
                                type="text",
                                text=f"Failed to replace item: {error_msg}",
                            )
                        ]
                case "clear_model":
                    result = await model_mgr.clear_model()
                    return [types.TextContent(type="text", text="Model cleared")]
                case "solve_model":
                    # Check if required parameter is provided
                    if "timeout" not in arguments:
                        return [
                            types.TextContent(
                                type="text",
                                text="Tool execution failed: Missing required parameter 'timeout'",
                            )
                        ]

                    timeout_val = None

                    # Parse and validate timeout parameter
                    try:
                        raw_timeout = arguments.get("timeout")
                        # Default to MIN_SOLVE_TIMEOUT if missing or not a number
                        if raw_timeout is None or not isinstance(
                            raw_timeout, (int, float)
                        ):
                            timeout_val = MIN_SOLVE_TIMEOUT
                        else:
                            # Cap at MAX_SOLVE_TIMEOUT if too large
                            if raw_timeout > MAX_SOLVE_TIMEOUT.seconds:
                                timeout_val = MAX_SOLVE_TIMEOUT
                            # Use MIN_SOLVE_TIMEOUT if too small
                            elif raw_timeout < MIN_SOLVE_TIMEOUT.seconds:
                                timeout_val = MIN_SOLVE_TIMEOUT
                            else:
                                timeout_val = timedelta(seconds=float(raw_timeout))
                    except (ValueError, TypeError):
                        # Use MIN_SOLVE_TIMEOUT for any parsing errors
                        timeout_val = MIN_SOLVE_TIMEOUT

                    # Log that we're about to solve with a timeout
                    logging.getLogger(__name__).info(
                        f"Solving model with timeout {timeout_val.total_seconds()} seconds"
                    )

                    # Create a safe timeout task to avoid hanging the server
                    try:
                        # Create a task with a slightly longer timeout than requested to allow for graceful handling
                        safe_timeout = (
                            timeout_val.total_seconds() + 5.0
                        )  # Increased buffer

                        # Run the solve_model with timeout and catch any exceptions
                        async with asyncio.timeout(safe_timeout):
                            try:
                                # Call the model manager to solve the model
                                result = await model_mgr.solve_model(
                                    timeout=timeout_val
                                )

                                # Check if the result indicates a timeout
                                if (
                                    result.get("status") == "timeout"
                                    or result.get("timeout") is True
                                ):
                                    logging.getLogger(__name__).info(
                                        "Model solving timed out, but handled gracefully"
                                    )

                                # Always ensure we're returning a valid result to prevent disconnections
                                if not isinstance(result, dict):
                                    # Convert to dictionary with success=True
                                    result = {
                                        "message": "Invalid result format from model manager",
                                        "success": True,
                                        "status": "error",
                                        "error": "Invalid result format",
                                    }

                                # Ensure success is True to avoid client disconnection
                                if not result.get("success", False):
                                    result["success"] = True
                                    result["status"] = "error"

                                # Convert result to string for client response
                                return [
                                    types.TextContent(type="text", text=str(result))
                                ]

                            except Exception as e:
                                # Catch any exceptions from the model manager
                                logging.getLogger(__name__).error(
                                    f"Error in model manager during solve: {e!s}",
                                    exc_info=True,
                                )
                                # Return a valid result even when errors occur
                                error_result = {
                                    "message": f"Error solving model: {e!s}",
                                    "success": True,  # Important: still success=True
                                    "status": "error",
                                    "error": str(e),
                                }
                                return [
                                    types.TextContent(
                                        type="text", text=str(error_result)
                                    )
                                ]

                    except TimeoutError:
                        # This should rarely happen since we use a timeout in model_mgr.solve_model
                        # But as an extra safety net, we handle it here at the server level
                        logging.getLogger(__name__).warning(
                            f"Server timeout occurred after {safe_timeout} seconds"
                        )
                        timeout_result = {
                            "message": f"Model solving timed out after {timeout_val.total_seconds()} seconds",
                            "success": True,
                            "status": "timeout",
                            "timeout": True,
                        }
                        return [
                            types.TextContent(type="text", text=str(timeout_result))
                        ]

                    except Exception as e:
                        # Handle any other errors during timeout handling itself
                        logging.getLogger(__name__).error(
                            f"Error in timeout handling: {e!s}", exc_info=True
                        )
                        error_result = {
                            "message": f"Error in timeout handling: {e!s}",
                            "success": True,  # Always success=True for client connection
                            "status": "error",
                            "error": str(e),
                        }
                        return [types.TextContent(type="text", text=str(error_result))]

                case _:
                    raise ValueError(f"Unknown tool: {name}")
        except Exception as e:
            logging.getLogger(__name__).error("Tool execution failed", exc_info=True)
            error_message = f"Tool execution failed: {e!s}"
            return [types.TextContent(type="text", text=error_message)]

    # Wrap the server run in try-except to handle unexpected errors
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        try:
            logging.getLogger(__name__).info("Starting MCP server")

            # Set up a handler for unexpected exceptions
            def global_exception_handler(loop, context):
                exception = context.get("exception")
                if exception:
                    logging.getLogger(__name__).error(
                        f"Uncaught exception: {exception!s}", exc_info=exception
                    )
                else:
                    logging.getLogger(__name__).error(
                        f"Uncaught exception context: {context}"
                    )

                # Don't terminate the loop - attempt to continue running
                # This helps maintain the client connection even during errors

            # Get the event loop and set the exception handler
            loop = asyncio.get_running_loop()
            loop.set_exception_handler(global_exception_handler)

            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="mcp-solver",
                    server_version=version_str,
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
        except asyncio.CancelledError:
            # This is typically raised during normal shutdown, log but don't treat as an error
            logging.getLogger(__name__).info(
                "MCP server operation cancelled, shutting down gracefully"
            )
            raise  # Re-raise to allow proper shutdown
        except Exception as e:
            # Log the error but try to avoid crashing the server
            logging.getLogger(__name__).error(
                f"Error in MCP server operation: {e!s}", exc_info=True
            )
            # Don't re-raise - this would terminate the server and disconnect clients
            # Instead, attempt to continue or at least shutdown more gracefully
            try:
                # Try to send a final message to the client before exiting
                if write_stream:
                    error_msg = {
                        "message": "Server experienced an error but is attempting to recover",
                        "error": str(e),
                    }
                    await write_stream.write(json.dumps(error_msg).encode() + b"\n")
            except:
                pass  # Ignore any errors in the error handler


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="MCP Solver")
    parser.add_argument("--z3", action="store_true", help="Use Z3 solver")
    parser.add_argument("--pysat", action="store_true", help="Use PySAT solver")
    parser.add_argument(
        "--maxsat", action="store_true", help="Use MaxSAT optimization solver"
    )
    parser.add_argument("--asp", action="store_true", help="Use ASP solver")
    parser.add_argument("--port", type=int, help="Port to listen on (debug)")
    args = parser.parse_args()

    # Set global flags based on arguments
    global Z3_MODE, PYSAT_MODE, MAXSAT_MODE, ASP_MODE
    Z3_MODE = args.z3
    PYSAT_MODE = args.pysat
    MAXSAT_MODE = args.maxsat
    ASP_MODE = args.asp

    # Check for incompatible flags
    if sum([Z3_MODE, PYSAT_MODE, MAXSAT_MODE, ASP_MODE]) > 1:
        print("Error: Cannot use multiple solver mode flags at the same time")
        return 1

    # Log the mode
    if Z3_MODE:
        logging.getLogger(__name__).info("Server running with Z3 solver")
    elif PYSAT_MODE:
        logging.getLogger(__name__).info("Server running with PySAT solver")
    elif MAXSAT_MODE:
        logging.getLogger(__name__).info(
            "Server running with MaxSAT optimization solver"
        )
    elif ASP_MODE:
        logging.getLogger(__name__).info("Server running with ASP solver")
    else:
        logging.getLogger(__name__).info("Server running with MiniZinc solver")

    # Setup logging
    log_level = logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    asyncio.run(serve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
