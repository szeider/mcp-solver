import asyncio
import logging
from typing import List, Optional, Union, Any
from datetime import timedelta
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types
from mcp.shared.exceptions import McpError

from .constants import DEFAULT_SOLVE_TIMEOUT, MEMO_FILE, PROMPT_TEMPLATE
from .model_manager import ModelManager 
from .memo import MemoManager

logger = logging.getLogger(__name__)

async def serve() -> None:
    server = Server("mcp-solver")
    model_mgr = ModelManager()
    memo = MemoManager(MEMO_FILE)

    @server.list_prompts()
    async def handle_list_prompts() -> list[types.Prompt]:
        return [
            types.Prompt(
                name="Guidelines",
                description="A prompt to help user to solve problems with the MCP solver",
                arguments=[]
            )
        ]

    @server.get_prompt()
    async def handle_get_prompt(name: str, arguments: dict[str, str] | None) -> types.GetPromptResult:
        if name != "Guidelines":
            raise ValueError(f"Unknown prompt: {name}")

        current_memo = memo.content or "No knowledge available yet."
        prompt = PROMPT_TEMPLATE.format(memo=current_memo)

        return types.GetPromptResult(
            description="MiniZinc solver template",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=prompt),
                )
            ],
        )

    def format_array_access(variable_name: str, indices: List[int]) -> str:
        """Format array access string with indices"""
        if not indices:
            return variable_name
        return f"{variable_name}[{','.join(str(i) for i in indices)}]"

    def get_array_value(array: Any, indices: List[int]) -> Any:
        """Recursively access array elements using provided indices"""
        if not indices:
            return array
        if not hasattr(array, "__getitem__"):
            raise ValueError("Variable is not an array")
            
        try:
            # Handle single dimension
            if len(indices) == 1:
                return array[indices[0]-1]  # Convert from 1-based to 0-based indexing
                
            # Handle multiple dimensions recursively
            return get_array_value(array[indices[0]-1], indices[1:])
        except IndexError:
            raise ValueError(f"Index {indices[0]} is out of bounds")
        except TypeError:
            raise ValueError("Invalid index type")

    @server.list_tools()
    async def list_tools() -> List[types.Tool]:
        return [
            types.Tool(
                name="get_model",
                description="Get current constraint model content.",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            types.Tool(
                name="edit_model",
                description="""Edit constraint model content.
                Supports finite domain variable and global constraints.
                Solve satisfy or solve optimize.
                No MIP or linear programming.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "line_start": {
                            "type": "integer",
                            "description": "Starting line number (1-based)"
                        },
                        "line_end": {
                            "type": ["integer", "null"],
                            "description": "Ending line number, null for end"
                        },
                        "content": {
                            "type": "string",
                            "description": "New content to insert"
                        }
                    },
                    "required": ["line_start", "content"]
                }
            ),
            types.Tool(
                name="validate_model",
                description="Validate model syntax and semantics.",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            types.Tool(
                name="solve_model",
                description="Solve the model with the Chuffed constraint solver.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "timeout": {
                            "type": ["number", "null"],
                            "description": "Optional solve timeout in seconds"
                        }
                    }
                }
            ),
            types.Tool(
                name="get_variable",
                description="Get specific variable value from solution with optional array indices",
                inputSchema={
                    "type": "object", 
                    "properties": {
                        "variable_name": {
                            "type": "string",
                            "description": "Variable name to retrieve"
                        },
                        "indices": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Array indices (optional, 1-based)"
                        }
                    },
                    "required": ["variable_name"]
                }
            ),
            types.Tool(
                name="get_solve_time",
                description="Get last solve execution time",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            types.Tool(
                name="get_memo",
                description="Get current knowledge base",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            types.Tool(
                name="edit_memo",
                description="Edit knowledge base",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "line_start": {
                            "type": "integer",
                            "description": "Starting line number (1-based)"
                        },
                        "line_end": {
                            "type": ["integer", "null"],
                            "description": "Ending line number, null for end"
                        },
                        "content": {
                            "type": "string",
                            "description": "New content"
                        }
                    },
                    "required": ["line_start", "content"]
                }
            )
        ]

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict) -> List[types.TextContent]:
        try:
            match name:
                case "get_model":
                    return [types.TextContent(type="text", text=model_mgr.model_string)]

                case "edit_model":
                    model_mgr.edit_range(
                        arguments["line_start"],
                        arguments.get("line_end"),
                        arguments["content"]
                    )
                    return [types.TextContent(type="text", text="Model updated")]

                case "validate_model":
                    valid, message = await model_mgr.validate_model()
                    status = "Valid" if valid else "Invalid"
                    return [types.TextContent(type="text", text=f"{status}: {message}")]

                case "solve_model":
                    timeout_secs = arguments.get("timeout")
                    timeout = timedelta(seconds=timeout_secs) if timeout_secs else DEFAULT_SOLVE_TIMEOUT
                    result = await model_mgr.solve_model(timeout=timeout)
                    return [types.TextContent(type="text", text=str(result))]

                case "get_variable":
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
            raise McpError("Tool execution failed", str(e))

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="mcp-solver",
                server_version="0.2.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

def main() -> int:
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(serve())
        return 0
    except KeyboardInterrupt:
        return 0

if __name__ == "__main__":
    main()