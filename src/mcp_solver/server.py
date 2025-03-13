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

from .constants import DEFAULT_SOLVE_TIMEOUT, MEMO_FILE, ITEM_CHARS, INSTRUCTIONS_PROMPT
from .memo import MemoManager

# Global flags for mode selection
LITE_MODE = False
Z3_MODE = False

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
        model_mgr = Z3ModelManager(lite_mode=LITE_MODE)
        logging.getLogger(__name__).info("Using Z3 model manager")
    else:
        model_mgr = MiniZincModelManager(lite_mode=LITE_MODE)
        logging.getLogger(__name__).info("Using MiniZinc model manager")
    
    memo = MemoManager(MEMO_FILE)

    @server.list_prompts()
    async def handle_list_prompts() -> list[types.Prompt]:
        return [
            types.Prompt(
                name="Instructions",
                description="Basic instructions for using the tools. Get this prompt before any interaction with mcp-solver.",
                arguments=[]
            )
        ]
    
    @server.get_prompt()
    async def handle_get_prompt(name: str, arguments: dict[str, str] | None) -> types.GetPromptResult:
        # In lite mode, don't try to format a memo into the prompt
        if LITE_MODE:
            # Use the instruction prompt directly without attempting to format
            prompt = INSTRUCTIONS_PROMPT
        else:
            # Only in full mode, format the memo into the prompt
            current_memo = memo.content or "No knowledge available yet."
            prompt = INSTRUCTIONS_PROMPT.format(memo=current_memo)
            
        return types.GetPromptResult(
            description="MCP Solver Guidelines, read first!",
            messages=[
                types.PromptMessage(
                    role="user", 
                    content=types.TextContent(
                        type="text",
                        text=prompt
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
        # Base tools common to both MiniZinc and Z3 modes
        base_tools = [
            types.Tool(
                name="clear_model", 
                description=f"Remove all items from the {'Z3' if Z3_MODE else 'minizinc'} model, effectively resetting it.",
                inputSchema={"type": "object", "properties": {}}
            ),
            types.Tool(
                name="add_item", 
                description=f"Add new {'Python code' if Z3_MODE else 'minizinc item'} to the model at a specific index (indices start at 1).",
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
                description=f"Replace an existing item in the {'Z3' if Z3_MODE else 'minizinc'} model at a specified index with new content.",
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
                description=f"Delete an item from the {'Z3' if Z3_MODE else 'minizinc'} model at the specified index.",
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
                description=f"Fetch the current content of the {'Z3' if Z3_MODE else 'minizinc'} model, listing each item with its index.",
                inputSchema={"type": "object", "properties": {}}
            ),
            types.Tool(
                name="solve_model", 
                description=f"Solve the current {'Z3' if Z3_MODE else 'minizinc'} model with an optional timeout parameter.",
                inputSchema={
                    "type": "object", 
                    "properties": {
                        "timeout": {
                            "type": ["number", "null"],
                            "description": f"Optional solve timeout in seconds, must be smaller than the default of {DEFAULT_SOLVE_TIMEOUT} seconds"
                        }
                    }
                }
            ),
            types.Tool(
                name="get_solution", 
                description="Retrieve the current solution from the last solve operation.",
                inputSchema={"type": "object", "properties": {}}
            ),
            types.Tool(
                name="get_variable_value", 
                description="Get the value of a specific variable from the solution.",
                inputSchema={
                    "type": "object", 
                    "properties": {
                        "variable_name": {"type": "string"}
                    }, 
                    "required": ["variable_name"]
                }
            ),
            types.Tool(
                name="get_solve_time", 
                description="Retrieve the execution time of the most recent solve operation.",
                inputSchema={"type": "object", "properties": {}}
            )
        ]

        if LITE_MODE:
            # In lite mode, return a reduced set of tools
            return base_tools
        else:
            # Full set of tools for non-lite mode
            full_tools = base_tools + [
                types.Tool(
                    name="get_memo", 
                    description="Retrieve the current knowledge base memo.",
                    inputSchema={"type": "object", "properties": {}}
                ),
                types.Tool(
                    name="edit_memo", 
                    description="Edit the knowledge base memo by adding content within a specified line range with new text.",
                    inputSchema={"type": "object", "properties": {
                        "line_start": {"type": "integer"},
                        "line_end": {"type": ["integer", "null"]},
                        "content": {"type": "string"}
                    }, "required": ["line_start", "content"]}
                )
            ]
            return full_tools

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
                    if arguments and "timeout" in arguments:
                        raw_timeout = arguments.get("timeout")
                        if raw_timeout is not None:
                            timeout_val = timedelta(seconds=float(raw_timeout))
                    result = await model_mgr.solve_model(timeout=timeout_val)
                    return [types.TextContent(type="text", text=str(result))]
                case "get_solution":
                    result = model_mgr.get_solution()
                    return [types.TextContent(type="text", text=str(result))]
                case "get_variable_value":
                    var_name = arguments["variable_name"]
                    result = model_mgr.get_variable_value(var_name)
                    return [types.TextContent(type="text", text=str(result))]
                case "get_solve_time":
                    result = model_mgr.get_solve_time()
                    return [types.TextContent(type="text", text=str(result))]
                case "get_memo":
                    return [types.TextContent(type="text", text=memo.content)]
                case "edit_memo":
                    memo.edit_range(
                        arguments["line_start"],
                        arguments.get("line_end"),
                        arguments["content"]
                    )
                    return [types.TextContent(type="text", text="Memo updated")]
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
    logging.basicConfig(
        filename='mcp_solver.log',
        filemode='a',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting MCP solver with version: {version_str}")

    # Process command-line flags
    global LITE_MODE, Z3_MODE, INSTRUCTIONS_PROMPT
    
    # Check for Z3 mode
    if "--z3" in sys.argv:
        Z3_MODE = True
        logger.info("Z3 mode activated")
        
    # Check for lite mode
    if "--lite" in sys.argv:
        LITE_MODE = True
        logger.info("Lite mode activated")
    
    # Load the appropriate instruction prompt
    if Z3_MODE and LITE_MODE:
        try:
            prompt_path = Path(__file__).resolve().parents[2] / "instructions_prompt_z3_lite.md"
            with open(prompt_path, "r", encoding="utf-8") as f:
                INSTRUCTIONS_PROMPT = f.read()
            logger.info("Using instructions_prompt_z3_lite.md")
        except Exception as e:
            logger.error(f"Failed to load instructions_prompt_z3_lite.md: {e}")
    elif LITE_MODE:
        try:
            prompt_path = Path(__file__).resolve().parents[2] / "instructions_prompt_lite.md"
            with open(prompt_path, "r", encoding="utf-8") as f:
                INSTRUCTIONS_PROMPT = f.read()
            logger.info("Using instructions_prompt_lite.md")
        except Exception as e:
            logger.error(f"Failed to load instructions_prompt_lite.md: {e}")

    try:
        asyncio.run(serve())
        return 0
    except KeyboardInterrupt:
        return 0

if __name__ == "__main__":
    main()