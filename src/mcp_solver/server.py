import sys
import asyncio
import logging
from typing import List, Optional, Union, Any, Tuple
from datetime import timedelta
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types
from mcp.shared.exceptions import McpError

from .constants import DEFAULT_SOLVE_TIMEOUT, MEMO_FILE, PROMPT_TEMPLATE, ITEM_CHARS
from .model_manager import ModelManager, ModelError
from .memo import MemoManager

from importlib.metadata import version

logger = logging.getLogger(__name__)

try:
    version_str = version("mcp-solver")
    logger.info(f"Loaded version: {version_str}")
except Exception:
    version_str = "0.0.0"
    logger.warning("Failed to load version from package, using default: 0.0.0")

def format_model_items(items: List[Tuple[int, str]], max_chars: Optional[int] = None) -> str:
    """Format model items with optional truncation.
    
    Args:
        items: List of (index, content) tuples
        max_chars: Maximum characters to show per item before truncating
    
    Returns:
        Formatted string with numbered items
    """
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
                name="Guidelines",
                description="Basic instructions for using the tools, get this prompt before any interaction with mcp-solver",
                arguments=[]
            )
        ]
    
    
    @server.get_prompt()
    async def handle_get_prompt(name: str, arguments: dict[str, str] | None) -> types.GetPromptResult:
        # Always perform initialization
        model_mgr.clear_model()
        
        # Get current memo content
        current_memo = memo.content or "No knowledge available yet."
        prompt = PROMPT_TEMPLATE.format(memo=current_memo)
        
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
            types.Tool(name="get_model", description="Get current model content with numbered items",
                inputSchema={"type": "object", "properties": {}}),
            types.Tool(name="add_item", description="Add new item at specific index",
                inputSchema={"type": "object", "properties": {
                    "index": {"type": "integer"},
                    "content": {"type": "string"}
                }, "required": ["index", "content"]}),
            types.Tool(name="delete_item", description="Delete item at index",
                inputSchema={"type": "object", "properties": {
                    "index": {"type": "integer"}
                }, "required": ["index"]}),
            types.Tool(name="replace_item", description="Replace item at index",
                inputSchema={"type": "object", "properties": {
                    "index": {"type": "integer"},
                    "content": {"type": "string"}
                }, "required": ["index", "content"]}),
            types.Tool(name="clear_model", description="Clear all items in the model",
                inputSchema={"type": "object", "properties": {}}),
            types.Tool(name="solve_model", description="Solve the model with the Chuffed constraint solver",
                inputSchema={"type": "object", "properties": {
                    "timeout": {"type": ["number", "null"],
                              "description": f"Optional solve timeout in seconds, must be smaller than the default of {DEFAULT_SOLVE_TIMEOUT} seconds"}
                }}),
            types.Tool(name="get_solution", description="Get specific variable value from solution with optional array indices",
                inputSchema={"type": "object", "properties": {
                    "variable_name": {"type": "string"},
                    "indices": {"type": "array", "items": {"type": "integer"},
                              "description": "Array indices (optional, 1-based)"}
                }, "required": ["variable_name"]}),
            types.Tool(name="get_solve_time", description="Get last solve execution time",
                inputSchema={"type": "object", "properties": {}}),
            types.Tool(name="get_memo", description="Get current knowledge base",
                inputSchema={"type": "object", "properties": {}}),
            types.Tool(name="edit_memo", description="Edit knowledge base",
                inputSchema={"type": "object", "properties": {
                    "line_start": {"type": "integer"},
                    "line_end": {"type": ["integer", "null"]},
                    "content": {"type": "string"}
                }, "required": ["line_start", "content"]})
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
                    # Handle the timeout argument more gracefully
                    timeout_val = None
                    if arguments and "timeout" in arguments:
                        raw_timeout = arguments.get("timeout")
                        if raw_timeout is not None:  # Only convert if not None
                            timeout_val = timedelta(seconds=float(raw_timeout))
                    result = await model_mgr.solve_model(
                        timeout=timeout_val  # Will use DEFAULT_SOLVE_TIMEOUT if None
                    )
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
                    solve_time = model_mgr.get_solve_time()
                    if solve_time is None:
                        return [types.TextContent(type="text", text="No solve time available")]
                    return [types.TextContent(type="text", text=f"Last solve: {solve_time:.3f}s")]

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
                # Instead of converting an error_response dict to JSON, simply return a plain string.
                error_message = f"Tool execution failed: {str(e)}"
                return [types.TextContent(type="text", text=error_message)]


        

        # except Exception as e:
        #     logger.error("Tool execution failed", exc_info=True)
        #     raise McpError(f"Tool execution failed: {e}")
    
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
        stream=sys.stderr,
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
