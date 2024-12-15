import asyncio
import json
import logging
from enum import Enum
from typing import List

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types
from mcp.shared.exceptions import McpError

from .solver import SolverSession
from .types import ToolNames

logger = logging.getLogger(__name__)

async def serve() -> None:
    server = Server("mcp-solver")
    session = SolverSession()

    @server.list_tools()
    async def list_tools() -> List[types.Tool]:
        return [
            types.Tool(
                name=ToolNames.SUBMIT_MODEL,
                description="""Submit and validate a MiniZinc constraint model.
           
This solver requires models written in standard MiniZinc using:
- Variable declarations with finite domains
- Global constraints (alldifferent, circuit, etc.)  
- Logical constraints (and, or, implication)
- Reification and channeling constraints

The solver uses Chuffed, a lazy clause generation solver, and does NOT support:
- Mixed Integer Programming (MIP) formulations
- Linear programming
- Optimization with linear objective functions
- The 'mip' library or any MIP-specific constraints

Please write your model using standard MiniZinc modeling techniques.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "model": {
                            "type": "string",
                            "description": "Full MiniZinc model code"
                        }
                    },
                    "required": ["model"]
                }
            ),
            types.Tool(
                name=ToolNames.SOLVE_MODEL,
                description="Solve the current constraint model",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            types.Tool(
                name=ToolNames.GET_SOLUTION,
                description="Retrieve the stored solution from the last solve operation",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            types.Tool(
                name=ToolNames.SET_PARAMETER,
                description="Set a parameter value for the current model",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "param_name": {
                            "type": "string",
                            "description": "Name of the parameter to set"
                        },
                        "param_value": {
                            "description": "Value to set for the parameter"
                        }
                    },
                    "required": ["param_name", "param_value"]
                }
            ),
            types.Tool(
                name=ToolNames.GET_VARIABLE,
                description="Get a variable's value from the most recent solution",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "variable_name": {
                            "type": "string",
                            "description": "Name of the variable to retrieve"
                        }
                    },
                    "required": ["variable_name"]
                }
            ),
            types.Tool(
                name=ToolNames.GET_SOLVE_TIME,
                description="Get the running time used to find the most recent solution",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            types.Tool(
                name=ToolNames.GET_SOLVER_STATE,
                description="Check if the solver is currently running and get elapsed time",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> List[types.TextContent]:
        try:
            match name:
                case ToolNames.SUBMIT_MODEL:
                    success, message = session.validate_and_define_model(arguments["model"])
                    status = "Success" if success else "Error"
                    return [types.TextContent(type="text", text=f"{status}: {message}")]

                case ToolNames.SOLVE_MODEL:
                    result = await session.solve_model()
                    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

                case ToolNames.GET_SOLUTION:
                    solution = session.get_current_solution()
                    if solution is None:
                        return [types.TextContent(type="text", text="No solution available yet. Use solve-model first.")]
                    return [types.TextContent(type="text", text=json.dumps(solution, indent=2))]

                case ToolNames.SET_PARAMETER:
                    msg = session.set_parameter(arguments["param_name"], arguments["param_value"])
                    return [types.TextContent(type="text", text=msg)]

                case ToolNames.GET_VARIABLE:
                    val = session.get_variable_value(arguments["variable_name"])
                    if val is None:
                        return [types.TextContent(type="text", text=f"No value found for {arguments['variable_name']}. Make sure you've solved the model first.")]
                    return [types.TextContent(type="text", text=f"{arguments['variable_name']} = {val}")]

                case ToolNames.GET_SOLVE_TIME:
                    solve_time = session.get_solve_time()
                    if solve_time is None:
                        return [types.TextContent(type="text", text="No solve time available. Use solve-model first.")]
                    return [types.TextContent(type="text", text=f"Solve time: {solve_time:.3f} seconds")]

                case ToolNames.GET_SOLVER_STATE:
                    state = session.get_solver_state()
                    return [types.TextContent(type="text", text=json.dumps(state, indent=2))]

                case _:
                    raise ValueError(f"Unknown tool: {name}")

        except Exception as e:
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
    logger.info("MCP Constraint Solver Server starting...")
    try:
        asyncio.run(serve())
        return 0
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
        return 0

if __name__ == "__main__":
    main()
