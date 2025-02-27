import sys
import asyncio
import logging
from typing import List, Optional, Any, Tuple
from datetime import timedelta
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types
from mcp.shared.exceptions import McpError

from .constants import DEFAULT_SOLVE_TIMEOUT, MEMO_FILE, ITEM_CHARS, INSTRCUTIONS_PROMPT
from .model_manager import ModelManager, ModelError
from .memo import MemoManager

from importlib.metadata import version
import os

logger = logging.getLogger(__name__)

try:
    version_str = version("mcp-solver")
    logger.info(f"Loaded version: {version_str}")
except Exception:
    version_str = "0.0.0"
    logger.warning("Failed to load version from package, using default: 0.0.0")

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
    model_mgr = ModelManager()
    memo = MemoManager(MEMO_FILE)

    @server.list_prompts()
    async def handle_list_prompts() -> list[types.Prompt]:
        return [
            types.Prompt(
                name="Instructions",
                description="Basic instructions for using the tools, get this prompt before any interaction with mcp-solver",
                arguments=[]
            )
        ]
    
    @server.get_prompt()
    async def handle_get_prompt(name: str, arguments: dict[str, str] | None) -> types.GetPromptResult:
        # Do not clear the model here—let previous model items persist.
        current_memo = memo.content or "No knowledge available yet."
        prompt = INSTRCUTIONS_PROMPT.format(memo=current_memo)
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
        return [
            types.Tool(
                name="add_item", 
                description="Add new minizinc item to the model at a specific index, where indices start at 0; gets back the current model in truncated form. Do not add minizinc output statements. ",
                inputSchema={"type": "object", "properties": {
                    "index": {"type": "integer"},
                    "content": {"type": "string"}
                }, "required": ["index", "content"]}
            ),
            types.Tool(
                name="solve_model", 
                description="Solve the current minizinc model using the Chuffed constraint solver with an optional timeout parameter, returning the result of the computation.",
                inputSchema={"type": "object", "properties": {
                    "timeout": {"type": ["number", "null"],
                                "description": f"Optional solve timeout in seconds, must be smaller than the default of {DEFAULT_SOLVE_TIMEOUT} seconds"}
                }}
            ),
            types.Tool(
                name="get_solution", 
                description="Retrieve the value of a specific variable from the model's solution, optionally accessing array elements using 1-based indices.",
                inputSchema={"type": "object", "properties": {
                    "variable_name": {"type": "string"},
                    "indices": {"type": "array", "items": {"type": "integer"},
                                "description": "Array indices (optional, 1-based)"}
                }, "required": ["variable_name"]}
            ),
            types.Tool(
                name="get_model", 
                description="Fetch the current content of the minizinc model, listing each item with its index. To save bandwith, only the first few charaters of each item is shown.",
                inputSchema={"type": "object", "properties": {}}
            ),
            types.Tool(
                name="clear_model", 
                description="Remove all items from the minizinc model, effectively resetting it.",
                inputSchema={"type": "object", "properties": {}}
            ),
            types.Tool(
                name="delete_item", 
                description="Delete an item from the minizinc model at the specified index, then return the updated model.",
                inputSchema={"type": "object", "properties": {
                    "index": {"type": "integer"}
                }, "required": ["index"]}
            ),
            types.Tool(
                name="replace_item", 
                description="Replace an existing item in the minizinc model at a specified index with new content, returning the updated model.",
                inputSchema={"type": "object", "properties": {
                    "index": {"type": "integer"},
                    "content": {"type": "string"}
                }, "required": ["index", "content"]}
            ),
            types.Tool(
                name="get_solve_time", 
                description="Retrieve the execution time of the most recent solve operation for performance monitoring.",
                inputSchema={"type": "object", "properties": {}}
            ),
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
                    await model_mgr.insert_item(arguments["index"], arguments["content"])
                    items = model_mgr.get_model()
                    return [types.TextContent(type="text", 
                        text=f"Item added\nCurrent model:\n{format_model_items(items, ITEM_CHARS)}")]
                case "delete_item":
                    await model_mgr.delete_item(arguments["index"])
                    items = model_mgr.get_model()
                    return [types.TextContent(type="text", 
                        text=f"Item deleted\nCurrent model:\n{format_model_items(items, ITEM_CHARS)}")]
                case "replace_item":
                    await model_mgr.replace_item(arguments["index"], arguments["content"])
                    items = model_mgr.get_model()
                    return [types.TextContent(type="text", 
                        text=f"Item replaced\nCurrent model:\n{format_model_items(items, ITEM_CHARS)}")]
                case "clear_model":
                    model_mgr.clear_model()
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
                    var_name = arguments["variable_name"]
                    indices = arguments.get("indices", [])
                    val = model_mgr.get_variable_value(var_name)
                    if val is None:
                        return [types.TextContent(type="text", text=f"No value for {var_name}")]
                    try:
                        if indices:
                            val = get_array_value(val, indices)
                            var_display = format_array_access(var_name, indices)
                        else:
                            var_display = var_name
                        return [types.TextContent(type="text", text=f"{var_display} = {val}")]
                    except ValueError as e:
                        return [types.TextContent(type="text", text=f"Error accessing {var_name}: {str(e)}")]
                    
                case "get_solve_time":
                    solve_time_info = model_mgr.get_solve_time()
                    st = solve_time_info.get("solve_time")
                    if st is None:
                        return [types.TextContent(type="text", text="No solve time available")]
                    return [types.TextContent(type="text", text=f"Last solve: {st:.3f}s")]
                
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
            logger.error("Tool execution failed", exc_info=True)
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
    logger.info(f"Starting MCP solver with version: {version_str}")
    try:
        asyncio.run(serve())
        return 0
    except KeyboardInterrupt:
        return 0

if __name__ == "__main__":
    main()
