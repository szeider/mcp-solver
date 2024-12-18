import asyncio
import logging
from typing import List
from datetime import timedelta
from minizinc import Model, Instance, Solver, Result
from minizinc.error import MiniZincError
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types
from mcp.shared.exceptions import McpError

from .constants import (
    DEFAULT_SOLVE_TIMEOUT,
    MAX_SOLVE_TIMEOUT,
    FLATTEN_TIMEOUT,
    MODEL_INSTRUCTIONS,
    PROMPT_TEMPLATE
)
from .solver import SolverSession

logger = logging.getLogger(__name__)

async def serve() -> None:
    server = Server("mcp-solver")
    session = SolverSession()

    @server.list_prompts()
    async def handle_list_prompts() -> list[types.Prompt]:
        return [
            types.Prompt(
                name="constraint-solver",
                description="Guide for formulating and solving constraint programming problems",
                arguments=[
                    types.PromptArgument(
                        name="topic",
                        description="Problem domain or specific constraint problem to solve",
                        required=True,
                    )
                ],
            )
        ]

    @server.get_prompt()
    async def handle_get_prompt(name: str, arguments: dict[str, str] | None) -> types.GetPromptResult:
        """
        Returns the final prompt to the LLM, including a system message containing
        general syntax instructions or best practices for MiniZinc.
        """
        if name != "constraint-solver":
            raise ValueError(f"Unknown prompt: {name}")

        if not arguments or "topic" not in arguments:
            raise ValueError("Missing required argument: topic")

        topic = arguments["topic"]
        prompt = PROMPT_TEMPLATE.format(topic=topic)

        # The system-level instructions. These are injected automatically
        # once at the start of the conversation so the LLM knows to obey them.
        system_instructions = """
        You are a constraint modeling assistant. Always produce valid MiniZinc code.
        - For 2D arrays, use nested brackets or array2d().
        - Avoid mixing var declarations with parameter definitions.
        - Ensure all parameters are set before solving, but keep decision variables as 'var int:'.
        - Syntax errors must be prevented whenever possible.
        """

        system_message = types.PromptMessage(
            role="system",
            content=types.TextContent(
                type="text",
                text=system_instructions.strip()
            ),
        )

        user_message = types.PromptMessage(
            role="user",
            content=types.TextContent(
                type="text",
                text=prompt.strip()
            ),
        )

        return types.GetPromptResult(
            description=f"Constraint solving guide for {topic}",
            # The system message appears first, user message second
            messages=[system_message, user_message],
        )

    @server.list_tools()
    async def list_tools() -> List[types.Tool]:
        return [
            types.Tool(
                name="submit_model",
                description=MODEL_INSTRUCTIONS,
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
                name="solve_model",
                description="Solve the current constraint model",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            types.Tool(
                name="get_solution",
                description="Retrieve the stored solution from the last solve operation",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            types.Tool(
                name="set_parameter",
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
                name="get_variable",
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
                name="get_solve_time",
                description="Get the running time used to find the most recent solution",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            types.Tool(
                name="get_solver_state",
                description="Check if the solver is currently running and get elapsed time",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            )
        ]

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict) -> List[types.TextContent]:
        try:
            match name:
                case "submit_model":
                    success, message = await session.validate_and_define_model(arguments["model"])
                    status = "Success" if success else "Error"
                    return [types.TextContent(type="text", text=f"{status}: {message}")]

                case "solve_model":
                    result = await session.solve_model()
                    return [types.TextContent(type="text", text=str(result))]

                case "get_solution":
                    solution = session.get_current_solution()
                    if solution is None:
                        return [types.TextContent(type="text", text="No solution available yet. Use solve-model first.")]
                    return [types.TextContent(type="text", text=str(solution))]

                case "set_parameter":
                    session.set_parameter(arguments["param_name"], arguments["param_value"])
                    return [types.TextContent(type="text", text=f"Parameter {arguments['param_name']} set successfully")]

                case "get_variable":
                    val = session.get_variable_value(arguments["variable_name"])
                    if val is None:
                        return [types.TextContent(type="text", text=f"No value found for {arguments['variable_name']}")]
                    return [types.TextContent(type="text", text=f"{arguments['variable_name']} = {val}")]

                case "get_solve_time":
                    solve_time = session.get_solve_time()
                    if solve_time is None:
                        return [types.TextContent(type="text", text="No solve time available - no solution has been computed yet")]
                    return [types.TextContent(type="text", text=f"Last solve operation took {solve_time:.3f} seconds")]

                case "get_solver_state":
                    state = session.get_solver_state()
                    return [types.TextContent(type="text", text=str(state))]

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
