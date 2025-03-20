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
