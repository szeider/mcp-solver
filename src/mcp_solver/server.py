# src/mcp_solver/server.py

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

from .constants import DEFAULT_SOLVE_TIMEOUT, MEMO_FILE, ITEM_CHARS
from .model_manager import ModelManager, ModelError
from .memo import MemoManager

from importlib.metadata import version
import os

# ----------------------
# Static Prompt Definitions
# ----------------------
PROMPTS = {
    "quick_prompt": {
         "name": "quick_prompt",
         "description": "The default quick prompt with essential guidelines.",
         "template": "Quick Prompt: Please follow the standard operational guidelines."
    },
    "detailed_prompt": {
         "name": "detailed_prompt",
         "description": "A detailed prompt with extended instructions.",
         "template": "Detailed Prompt: Please adhere to the following extended instructions: [Insert comprehensive guidelines here...]"
    }
}

def load_prompt_file(filename: str) -> str:
    filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to load prompt file {filename}: {e}")
        return f"Error loading prompt: {filename}"

QUICK_PROMPT = load_prompt_file("quick_prompt.md")
DETAILED_PROMPT = load_prompt_file("detailed_prompt.md")
# ----------------------

try:
    version_str = version("mcp-solver")
    logging.getLogger(__name__).info(f"Loaded version: {version_str}")
except Exception:
    version_str = "0.0.0"
    logging.getLogger(__name__).warning("Failed to load version from package, using default: 0.0.0")

async def serve() -> None:
    server = Server("mcp-solver")
    
    # Build detailed metadata for each tool endpoint.
    detailed_tools = [
        {
            "name": "get_model",
            "description": "Returns current model content with numbered items.",
            "inputSchema": {"type": "object", "properties": {}},
            "supportedOperations": ["read"]
        },
        {
            "name": "add_item",
            "description": "Adds a new item at a specific index to the model.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "content": {"type": "string"}
                },
                "required": ["index", "content"]
            },
            "supportedOperations": ["create"]
        },
        {
            "name": "delete_item",
            "description": "Deletes an item at a specific index from the model.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"}
                },
                "required": ["index"]
            },
            "supportedOperations": ["delete"]
        },
        {
            "name": "replace_item",
            "description": "Replaces an existing item at a specific index with new content.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "content": {"type": "string"}
                },
                "required": ["index", "content"]
            },
            "supportedOperations": ["update"]
        },
        {
            "name": "clear_model",
            "description": "Clears all items in the model.",
            "inputSchema": {"type": "object", "properties": {}},
            "supportedOperations": ["delete"]
        },
        {
            "name": "solve_model",
            "description": "Solves the model using the Chuffed constraint solver.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "timeout": {"type": ["number", "null"],
                              "description": f"Optional solve timeout in seconds, must be smaller than the default of {DEFAULT_SOLVE_TIMEOUT} seconds"}
                }
            },
            "supportedOperations": ["compute"]
        },
        {
            "name": "get_solution",
            "description": "Retrieves the value of a specified variable from the solution.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "variable_name": {"type": "string"},
                    "indices": {"type": "array", "items": {"type": "integer"}, "description": "Array indices (optional, 1-based)"}
                },
                "required": ["variable_name"]
            },
            "supportedOperations": ["read"]
        },
        {
            "name": "get_solve_time",
            "description": "Returns the last recorded solve execution time.",
            "inputSchema": {"type": "object", "properties": {}},
            "supportedOperations": ["read"]
        },
        {
            "name": "get_memo",
            "description": "Retrieves the current knowledge base (memo).",
            "inputSchema": {"type": "object", "properties": {}},
            "supportedOperations": ["read"]
        },
        {
            "name": "edit_memo",
            "description": "Edits the current knowledge base (memo).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "line_start": {"type": "integer"},
                    "line_end": {"type": ["integer", "null"]},
                    "content": {"type": "string"}
                },
                "required": ["line_start", "content"]
            },
            "supportedOperations": ["update"]
        }
    ]
    
    # Declare capabilities explicitly for prompts and tools.
    capabilities = {
        "prompts": {
            "default": "quick_prompt",   # Preferred prompt for clients.
            "listChanged": False         # Static list: no dynamic changes.
        },
        "tools": detailed_tools
    }
    server.capabilities = capabilities
    logging.info("Declared MCP Server Capabilities: %s", capabilities)
    
    model_mgr = ModelManager()
    memo = MemoManager(MEMO_FILE)

    # --- Prompt Endpoints ---
    @server.list_prompts()
    async def handle_list_prompts() -> list[types.Prompt]:
        # Return our two static prompts (no arguments)
        prompt_list = []
        for prompt in PROMPTS.values():
            prompt_list.append(
                types.Prompt(
                    name=prompt["name"],
                    description=prompt["description"],
                    arguments=[]  # Static prompts have no arguments.
                )
            )
        return prompt_list

    @server.get_prompt()
    async def handle_get_prompt(name: str, arguments: dict[str, str] | None) -> types.GetPromptResult:
        if name not in PROMPTS:
            return types.GetPromptResult(
                description="Error: Unknown prompt",
                messages=[
                    types.PromptMessage(
                        role="assistant",
                        content=types.TextContent(
                            type="text",
                            text=f"Prompt '{name}' not found."
                        )
                    )
                ]
            )
        prompt = PROMPTS[name]
        return types.GetPromptResult(
            description=prompt["description"],
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=prompt["template"]
                    )
                )
            ]
        )
    # --- End of Prompt Endpoints ---

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

    def format_model_items(items: List[Tuple[int, str]], max_chars: Optional[int] = None) -> str:
        if not items:
            return "Model is empty"
        def truncate(text: str) -> str:
            if max_chars is None or len(text) <= max_chars:
                return text
            return text[:max_chars] + "..."
        return "\n".join(f"{i} | {truncate(content)}" for i, content in items)

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
                            from .utils import get_array_value, format_array_access
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
        stream=sys.stderr,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.getLogger(__name__).info(f"Starting MCP solver with version: {version_str}")
    try:
        asyncio.run(serve())
        return 0
    except KeyboardInterrupt:
        return 0

if __name__ == "__main__":
    main()
