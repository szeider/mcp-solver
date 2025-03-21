import sys
import asyncio
import logging
import os
from typing import List, Optional, Any, Tuple
from datetime import timedelta
from pathlib import Path
from importlib.metadata import version

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types
from mcp.shared.exceptions import McpError

from .constants import MIN_SOLVE_TIMEOUT, MAX_SOLVE_TIMEOUT, VALIDATION_TIMEOUT, CLEANUP_TIMEOUT, ITEM_CHARS, INSTRUCTIONS_PROMPT

# Global flags for mode selection
Z3_MODE = False
PYSAT_MODE = False

try:
    version_str = version("mcp-solver")
    logging.getLogger(__name__).info(f"Loaded version: {version_str}")
except Exception:
    version_str = "0.0.0"
    logging.getLogger(__name__).warning("Failed to load version from package, using default: 0.0.0")

# Import after setting up logging and flags
from .mzn.model_manager import MiniZincModelManager, ModelError

def format_model_items(items: List[Tuple[int, str]], max_chars: Optional[int] = None) -> str:
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
        from .z3.model_manager import Z3ModelManager
        model_mgr = Z3ModelManager()
        logging.getLogger(__name__).info("Using Z3 model manager")
    elif PYSAT_MODE:
        from .pysat.model_manager import PySATModelManager
        model_mgr = PySATModelManager()
        logging.getLogger(__name__).info("Using PySAT model manager")
    else:
        model_mgr = MiniZincModelManager()
        logging.getLogger(__name__).info("Using MiniZinc model manager")
    
    # Helper function to get mode-specific descriptions
    def get_description(descriptions: dict) -> str:
        """
        Get the appropriate description based on the current mode.
        
        Args:
            descriptions: A dictionary of descriptions keyed by mode ('z3', 'pysat', 'mzn')
                          or a string for a common description across all modes
        
        Returns:
            The appropriate description string for the current mode
        """
        if isinstance(descriptions, str):
            return descriptions
            
        if Z3_MODE and 'z3' in descriptions:
            return descriptions['z3']
        elif PYSAT_MODE and 'pysat' in descriptions:
            return descriptions['pysat']
        elif not Z3_MODE and not PYSAT_MODE and 'mzn' in descriptions:
            return descriptions['mzn']
        elif 'default' in descriptions:
            return descriptions['default']
        else:
            # Return the first available description if no matching mode is found
            return next(iter(descriptions.values()))

    @server.list_prompts()
    async def handle_list_prompts() -> list[types.Prompt]:
        return [
            types.Prompt(
                name="instructions",
                title="MCP Solver Instructions",
                description="Instructions for using the MCP Solver"
            )
        ]
    
    @server.get_prompt()
    async def handle_get_prompt(name: str, arguments: dict[str, str] | None) -> types.GetPromptResult:
        # Choose the appropriate instruction prompt based on mode
        if name == "instructions":
            # Z3 and PySAT modes
            if Z3_MODE:
                prompt_path = INSTRUCTIONS_PROMPT.replace(".md", "_z3.md")
                logging.getLogger(__name__).info("Using Z3 instructions")
            elif PYSAT_MODE:
                prompt_path = INSTRUCTIONS_PROMPT.replace(".md", "_pysat.md")
                logging.getLogger(__name__).info("Using PySAT instructions")
            # MiniZinc mode
            else:
                prompt_path = INSTRUCTIONS_PROMPT.replace(".md", "_mzn.md")
                logging.getLogger(__name__).info("Using MiniZinc instructions")
            
            try:
                with open(prompt_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    
                    # Add debugging logs
                    logging.getLogger(__name__).info(f"Prompt loaded from: {prompt_path}")
                    logging.getLogger(__name__).info(f"Prompt content length: {len(content)}")
                    logging.getLogger(__name__).info(f"Prompt content first 100 chars: {content[:100]}")
            except FileNotFoundError:
                logging.getLogger(__name__).error(f"Prompt file not found: {prompt_path}")
                return types.GetPromptResult(
                    messages=[
                        types.PromptMessage(
                            role="user",
                            content=types.TextContent(
                                type="text",
                                text="Error: Prompt file not found"
                            )
                        )
                    ]
                )
            except Exception as e:
                logging.getLogger(__name__).error(f"Error reading prompt file: {str(e)}")
                return types.GetPromptResult(
                    messages=[
                        types.PromptMessage(
                            role="user",
                            content=types.TextContent(
                                type="text",
                                text=f"Error reading prompt file: {str(e)}"
                            )
                        )
                    ]
                )
            
            # Return with the new format
            return types.GetPromptResult(
                messages=[
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(
                            type="text",
                            text=content
                        )
                    )
                ]
            )
        else:
            logging.getLogger(__name__).error(f"Unknown prompt: {name}")
            return types.GetPromptResult(
                messages=[
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(
                            type="text",
                            text="Error: Unknown prompt"
                        )
                    )
                ]
            )

    def format_array_access(variable_name: str, indices: List[int]) -> str:
        return variable_name if not indices else f"{variable_name}[{','.join(str(i) for i in indices)}]"

    def get_array_value(array: Any, indices: List[int]) -> Any:
        if not indices:
            return array
        if not hasattr(array, "__getitem__"):
            raise ValueError("Variable is not an array")
        try:
            return array[indices[0]-1] if len(indices) == 1 else get_array_value(array[indices[0]-1], indices[1:])
        except IndexError:
            raise ValueError(f"Index {indices[0]} is out of bounds")
        except TypeError:
            raise ValueError("Invalid index type")

    @server.list_tools()
    async def list_tools() -> List[types.Tool]:
        # Tools for all modes
        tools = [
            types.Tool(
                name="clear_model", 
                description=get_description({
                    'mzn': "Remove all items from the minizinc model, effectively resetting it.",
                    'z3': "Remove all items from the Z3 Python model, effectively resetting it.",
                    'pysat': "Remove all items from the PySAT Python model, effectively resetting it."
                }),
                inputSchema={"type": "object", "properties": {}}
            ),
            types.Tool(
                name="add_item", 
                description=get_description({
                    'mzn': "Add new minizinc item to the model at a specific index (indices start at 1).",
                    'z3': "Add new Python code to the Z3 model at a specific index (indices start at 1).",
                    'pysat': "Add new Python code to the PySAT model at a specific index (indices start at 1)."
                }),
                inputSchema={
                    "type": "object", 
                    "properties": {
                        "index": {"type": "integer"},
                        "content": {"type": "string"}
                    }, 
                    "required": ["index", "content"]
                }
            ),
            types.Tool(
                name="replace_item", 
                description=get_description({
                    'mzn': "Replace an existing item in the minizinc model at a specified index with new content.",
                    'z3': "Replace an existing item in the Z3 Python model at a specified index with new content.",
                    'pysat': "Replace an existing item in the PySAT Python model at a specified index with new content."
                }),
                inputSchema={
                    "type": "object", 
                    "properties": {
                        "index": {"type": "integer"},
                        "content": {"type": "string"}
                    }, 
                    "required": ["index", "content"]
                }
            ),
            types.Tool(
                name="delete_item", 
                description=get_description({
                    'mzn': "Delete an item from the minizinc model at the specified index.",
                    'z3': "Delete an item from the Z3 Python model at the specified index.",
                    'pysat': "Delete an item from the PySAT Python model at the specified index."
                }),
                inputSchema={
                    "type": "object", 
                    "properties": {
                        "index": {"type": "integer"}
                    }, 
                    "required": ["index"]
                }
            ),
            types.Tool(
                name="get_model", 
                description=get_description({
                    'mzn': "Fetch the current content of the minizinc model, listing each item with its index.",
                    'z3': "Fetch the current content of the Z3 Python model, listing each item with its index.",
                    'pysat': "Fetch the current content of the PySAT Python model, listing each item with its index."
                }),
                inputSchema={"type": "object", "properties": {}}
            ),
            types.Tool(
                name="solve_model", 
                description=get_description({
                    'mzn': "Solve the current minizinc model with a timeout parameter.",
                    'z3': "Solve the current Z3 Python model with a timeout parameter.",
                    'pysat': "Solve the current PySAT Python model with a timeout parameter."
                }),
                inputSchema={
                    "type": "object", 
                    "properties": {
                        "timeout": {
                            "description": f"Solve timeout in seconds (minimum: {MIN_SOLVE_TIMEOUT.seconds}, maximum: {MAX_SOLVE_TIMEOUT.seconds})",
                            "type": "number"
                        }
                    },
                    "required": ["timeout"]
                }
            ),
        ]
        return tools

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict) -> List[types.TextContent]:
        try:
            match name:
                case "get_model":
                    items = model_mgr.get_model()
                    if not items:
                        return [types.TextContent(type="text", text="Model is empty")]
                    return [types.TextContent(type="text", text="\n".join(f"{i} | {content}" for i, content in items))]
                case "add_item":
                    result = await model_mgr.add_item(arguments["index"], arguments["content"])
                    items = model_mgr.get_model()
                    return [types.TextContent(type="text", 
                        text=f"Item added\nCurrent model:\n{format_model_items(items, ITEM_CHARS)}")]
                case "delete_item":
                    result = await model_mgr.delete_item(arguments["index"])
                    items = model_mgr.get_model()
                    return [types.TextContent(type="text", 
                        text=f"Item deleted\nCurrent model:\n{format_model_items(items, ITEM_CHARS)}")]
                case "replace_item":
                    result = await model_mgr.replace_item(arguments["index"], arguments["content"])
                    items = model_mgr.get_model()
                    return [types.TextContent(type="text", 
                        text=f"Item replaced\nCurrent model:\n{format_model_items(items, ITEM_CHARS)}")]
                case "clear_model":
                    result = await model_mgr.clear_model()
                    return [types.TextContent(type="text", text="Model cleared")]
                case "solve_model":
                    timeout_val = None
                    try:
                        raw_timeout = arguments.get("timeout")
                        # Default to MIN_SOLVE_TIMEOUT if missing or not a number
                        if raw_timeout is None or not isinstance(raw_timeout, (int, float)):
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
                        
                    result = await model_mgr.solve_model(timeout=timeout_val)
                    return [types.TextContent(type="text", text=str(result))]
                case _:
                    raise ValueError(f"Unknown tool: {name}")
        except Exception as e:
            logging.getLogger(__name__).error("Tool execution failed", exc_info=True)
            error_message = f"Tool execution failed: {str(e)}"
            return [types.TextContent(type="text", text=error_message)]

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
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

def main() -> int:
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP Solver")
    parser.add_argument("--z3", action="store_true", help="Use Z3 solver")
    parser.add_argument("--pysat", action="store_true", help="Use PySAT solver")
    parser.add_argument("--port", type=int, help="Port to listen on (debug)")
    args = parser.parse_args()
    
    # Set global flags based on arguments
    global Z3_MODE, PYSAT_MODE
    Z3_MODE = args.z3
    PYSAT_MODE = args.pysat
    
    # Check for incompatible flags
    if Z3_MODE and PYSAT_MODE:
        print("Error: Cannot use both --z3 and --pysat flags at the same time")
        return 1
    
    # Log the mode
    if Z3_MODE:
        logging.getLogger(__name__).info("Server running with Z3 solver")
    elif PYSAT_MODE:
        logging.getLogger(__name__).info("Server running with PySAT solver")
    else:
        logging.getLogger(__name__).info("Server running with MiniZinc solver")
    
    # Setup logging
    log_level = logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    asyncio.run(serve())
    return 0

if __name__ == "__main__":
    sys.exit(main())