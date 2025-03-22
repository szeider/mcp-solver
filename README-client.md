# Test Client

This client is a one-shot agent for testing the MCP solver. It functions as an intermediary between an LLM and MCP server, facilitating the translation of natural language problem statements into formal constraint programming solutions.

## Available Tools

The test-client provides a set of tools to the LLM that allow it to interact with the MCP solver:
- CRUD operations for solver models
- Solving models with timeout

## Installation

The client is an optional dependency in the mcp-solver package. To install it:

```bash
uv install "mcp-solver[client]"
```

## Testing Client Setup

To verify your client installation and environment is correctly configured:

```bash
uv run test-setup-client
```

This checks:
- Required configuration files
- LLM client dependencies 
- API key for the default LLM model (Anthropic Claude)
- Basic client functionality

## Custom ReAct Agent

The MCP Solver includes a custom implementation of a ReAct (Reasoning + Acting) agent built from scratch using LangGraph. This implementation offers fine-grained control over the agent's behavior and demonstrates how to create a custom agent that can work with MCP tools.

### Features

- **Custom State Management**: The agent maintains a state with a history of messages.
- **Flexible Model Interaction**: Calls language models with the current state and system prompt.
- **Tool Execution**: Processes tool calls in both object and dictionary formats.
- **Smart Routing**: Determines the next node in the workflow based on message types.
- **Complete Graph**: Compiles a workflow graph with model and tools nodes.

### Usage

The custom agent can be created and used as follows:

```python
from mcp_solver.client.react_agent import create_custom_react_agent

# Create the agent with your LLM and tools
agent = create_custom_react_agent(
    llm=your_llm,
    tools=your_tools,
    system_prompt="You are a helpful assistant for solving constraint problems."
)

# Run the agent with a human message
final_state = agent.invoke({"messages": [HumanMessage(content="your question")]})
```

### Status and Known Issues

Currently, the client supports two agent implementations:

- **Built-in Agent**: The default LangChain ReAct agent implementation, which is stable and recommended for most users.
- **Custom Agent**: Our custom LangGraph implementation, which has been improved and is now functioning properly.

**Current Status:**
- The built-in agent works reliably with all MCP tools
- The custom agent has been fixed and now works correctly with the MCP tools
- Both implementations are suitable for solving constraint problems

**Previous Issues (Now Resolved):**
- Event loop conflicts when running in an asynchronous context
- Compatibility issues with some structured tools that don't support synchronous invocation
- Problems executing certain MCP tools that require specific invocation patterns

These issues have been addressed in the latest version.

### Toggling Agent Implementation

To switch between agent implementations, modify the `USE_CUSTOM_AGENT` constant in `src/mcp_solver/client/client.py`:

```python
# Flag to toggle between the built-in React agent and our custom implementation
USE_CUSTOM_AGENT = True  # Set to False to use the built-in agent
```

By default, the custom agent is now enabled (`USE_CUSTOM_AGENT = True`) as it provides full functionality with MCP tools.

### Testing

The repository includes several test scripts that demonstrate the custom agent:

- `test_custom_agent.py`: Tests the core functionality of the agent.
- `test_intermediate_steps.py`: Shows how to observe the intermediate states during agent execution.
- `test_mcp_tools.py`: Demonstrates integrating MCP tools with the custom agent.

Run tests with:

```bash
uv run python -m src.mcp_solver.client.test_intermediate_steps
```

## Command Line Usage

### Streamlined Commands

The client provides specialized commands for each solver backend that automatically select the appropriate prompt and server configuration:

#### MiniZinc (Constraint Programming)
```bash
uv run test-client --query queens_query.md
```

#### PySAT (Boolean Satisfiability)
```bash
uv run test-client-pysat --query graph_coloring.md
```

#### Z3 (SMT Solver)
```bash
uv run test-client-z3 --query send_more_money.md
```

### How It Works

The specialized commands simplify usage by:
- Automatically finding and using the appropriate prompt file
- Setting up the correct solver backend server
- Handling flag compatibility with test scripts
- Providing a consistent interface across solver types

### Custom Configuration

For more control, you can override any of the automatic settings:

```bash
uv run test-client-[backend] --query <path_to_query_file> [options]
```

Where `[backend]` is optional or one of `pysat` or `z3`.

### Arguments

| Argument | Description |
|----------|-------------|
| `--query` | Path to a markdown file containing the problem to solve |
| `--prompt` | (Optional) Path to a markdown file containing custom prompt instructions |
| `--model` | (Optional) Model to use (default: AT:claude-3-7-sonnet-20250219) |
| `--server` | (Optional) Custom command to run the MCP server |


## Basic Workflow

1. Create a markdown file with your problem statement (query file)
2. Choose the appropriate specialized command based on the solver you need
3. Run the command with your query file
4. The client will:
   - Connect to the selected LLM
   - Find and use the appropriate prompt for the selected solver
   - Process the LLM's responses and tool calls
   - Report the final results

## Examples

### Solving N-Queens with MiniZinc

```bash
uv run test-client --query problems/mzn/queens_query.md
```

### Solving Graph Coloring with PySAT

```bash
uv run test-client-pysat --query problems/pysat/graph_coloring.md
```

### Solving Cryptarithmetic with Z3

```bash
uv run test-client-z3 --query problems/z3/send_more_money.md
```

## API Keys

To access LLM providers, you'll need the corresponding API keys:

- For Anthropic (our default LLM): Set the `ANTHROPIC_API_KEY` environment variable

You can set these in your environment or include them in a `.env` file in the project root.

## Running Tests

The test scripts automatically use the specialized commands:

### MiniZinc Tests

```bash
uv run run_mzn_tests.py [problem_name]
```

### PySAT Tests

```bash
uv run run_pysat_tests.py [problem_name]
```

### Z3 Tests

```bash
uv run run_z3_tests.py [problem_name]
```
