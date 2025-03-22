------

# MCP Solver

[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io/) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) [![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)

A Model Context Protocol (MCP) server that exposes SAT and constraint solving capabilities to Large Language Models.

------

## Overview

The *MCP Solver* integrates SAT, SMT and Constraint Solving with LLMs through the Model Context Protocol, enabling AI models to interactively create, edit, and solve:

- Constraint models in [MiniZinc](https://www.minizinc.org/)
- SAT models in [PySAT](https://pysathq.github.io/)
- SMT formulas in [Z3 Python](https://ericpony.github.io/z3py-tutorial/guide-examples.htm)

For a detailed description of the *MCP Solver's* system architecture and theoretical foundations, see the accompanying research paper: Stefan Szeider, ["MCP-Solver: Integrating Language Models with Constraint Programming Systems"](https://arxiv.org/abs/2501.00539), arXiv:2501.00539, 2024.

## Available Tools

In the following, *item* refers to some part of the (MinZinc/Pysat/Z3) code, and *model* to the encoding. 

| Tool Name      | Description                                   |
| -------------- | --------------------------------------------- |
| `clear_model`  | Remove all items from the model               |
| `add_item`     | Add new item at a specific index              |
| `delete_item`  | Delete item at index                          |
| `replace_item` | Replace item at index                         |
| `get_model`    | Get current model content with numbered items |
| `solve_model`  | Solve the model (with timeout parameter)      |

------

## System Requirements

- Python and project manager [uv](https://docs.astral.sh/uv/) 
- Python 3.11+
- Mode-specific requirements: MiniZinc, PySAT, Python Z3 (required packages are installed via pip)
- Operating systems: macOS, Windows, Linux (with appropriate adaptations)

------

## Available Modes / Solving Backends

The MCP Solver provides three distinct operational modes, each integrating with a different constraint solving backend. Each mode requires specific dependencies and offers unique capabilities for addressing different classes of problems.

### MiniZinc Mode

MiniZinc mode provides integration with the MiniZinc constraint modeling language with the following features:

- Rich constraint expression with global constraints
- Integration with the Chuffed constraint solver
- Optimization capabilities
- Access to solution values via `get_solution`

**Dependencies**: Base installation is sufficient (included in core dependencies)

**Configuration**: To run in MiniZinc mode, use:

```
mcp-solver-mzn
```

### PySAT Mode

PySAT mode allows interaction with the Python SAT solving toolkit with the following features:

- Propositional constraint modeling using CNF (Conjunctive Normal Form)
- Access to various SAT solvers (Glucose3, Glucose4, Lingeling, etc.)
- Cardinality constraints (at_most_k, at_least_k, exactly_k)
- Support for boolean constraint solving

**Dependencies**: Requires the `python-sat` package (`uv pip install -e ".[pysat]"`)

**Configuration**: To run in PySAT mode, use:

```
mcp-solver-pysat
```

### Z3 Mode

Z3 mode provides access to Z3 SMT (Satisfiability Modulo Theories) solving capabilities with the following features:

- Rich type system: booleans, integers, reals, bitvectors, arrays
- Constraint solving with quantifiers
- Optimization capabilities
- Template library for common modeling patterns

**Dependencies**: Requires the `z3-solver` package (`uv pip install -e ".[z3]"`)

**Configuration**: To run in Z3 mode, use:

```
mcp-solver-z3
```

## Installation

1. Install an MCP-compatible client e.g., [Claude Desktop app](https://claude.ai/download) (for testing, you can also use our [internal test client](https://claude.ai/chat/990a5d65-2a47-469f-aceb-f9948d37408c#test-client))

2. Install the MCP Solver with appropriate dependencies based on the mode(s) you intend to use:

   ```bash
   git clone https://github.com/szeider/mcp-solver.git
   cd mcp-solver
   uv venv
   source .venv/bin/activate 
   
   # For MiniZinc mode
   uv pip install -e ".[mzn]"
   
   # For Z3 mode
   uv pip install -e ".[z3]"
   
   # For PySAT mode
   uv pip install -e ".[pysat]"
   
   # For all modes
   uv pip install -e ".[all]"
   
   # For development and testing
   uv pip install -e ".[dev]"
   ```

3. The setup can be tested for each mode:

   ```bash
   # For MiniZinc mode
   uv run test-setup-mzn
   
   # For PySAT mode
   uv run test-setup-pysat
   
   # For Z3 mode
   uv run test-setup-z3
   ```

4. Create the configuration file:

   For macOS/Linux:

```
   ~/Library/Application/Support/Claude/claude_desktop_config.json
```

For Windows:

```
   %APPDATA%\Claude\claude_desktop_config.json
```

Add the following content for MiniZinc mode (similar for PySAT or Z3):

```json
{
  "mcpServers": {
    "MCP Solver": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/mcp-solver/",
        "run",
        "mcp-solver-mzn"
      ]
    }
  }
}
```

For Windows:

```json
{
  "mcpServers": {
    "MCP Solver": {
      "command": "cmd.exe",
      "args": [
        "/C",
        "cd C:\\absolute\\path\\to\\mcp-solver && uv run mcp-solver-mzn"
      ]
    }
  }
}
```

1. For each mode, the appropriate instructions prompt is loaded from:
   - MiniZinc mode: `instructions_prompt_mzn.md`
   - PySAT mode: `instructions_prompt_pysat.md`
   - Z3 mode: `instructions_prompt_z3.md`

## Test MCP Client 

The MCP Solver includes a test MCP client for development, experimentation, and diagnostic purposes, based on the *ReAct* agent framework. This client functions as an intermediary between an LLM and the MCP server, facilitating the translation of natural language problem statements into formal constraint programming solutions. 

### Installation

```bash
# Install client dependencies
uv pip install -e ".[client]"

# Verify client installation and configuration
uv run test-setup-client
```

The client requires an **API key** from an LLM provider. For Anthropic (the default LLM is Claude Sonnet 3.7), set the `ANTHROPIC_API_KEY` environment variable. This can be set in your environment or included in a `.env` file in the project root.



### Usage

The client provides specialized commands for each solver backend:

```bash
# MiniZinc mode
uv run test-client --query <query_file>.md

# PySAT mode
uv run test-client-pysat --query <query_file>.md

# Z3 mode
uv run test-client-z3 --query <query_file>.md
```

------

## Examples

### Example 1: Casting Problem (MiniZinc)

This is based on an example by George Johnson (NYT, July 13, 1999).

**User**

> Suppose that a theatrical director feels obligated to cast either his ingenue, Actress Alvarez, or his nephew, Actor Cohen, in a production. But Miss Alvarez won't be in a play with Mr. Cohen (her former lover), and she demands that the cast include her new flame, Actor Davenport. The producer, with her own favors to repay, insists that Actor Branislavsky have a part. But Mr. Branislavsky won't be in any play with Miss Alvarez or Mr. Davenport. Can the director cast the play?

**System:** **`clear_model`**

> ```
> Model cleared
> ```

**System:** **`add_item`** (multiple)

> ```minizinc
> var bool: alvarez;
> var bool: cohen;
> var bool: branislavsky;
> var bool: davenport;
> constraint alvarez \/ cohen;                          % Must cast either Alvarez or Cohen
> constraint not (alvarez /\ cohen);                    % Alvarez won't work with Cohen
> constraint alvarez -> davenport;                      % If Alvarez is cast, Davenport must be cast
> constraint branislavsky;                              % Branislavsky must be cast
> constraint not (branislavsky /\ alvarez);             % Branislavsky won't work with Alvarez
> constraint not (branislavsky /\ davenport);           % Branislavsky won't work with Davenport
> solve satisfy;
> ```

**System:** **`solve_model`**

> ```
> {'status': 'SAT', 'solution': Solution(alvarez=False, cohen=True, branislavsky=True, davenport=False, _checker=''), 'solve_time': 0.0}
> ```

The solution (cast Cohen and Branislavsky, but not Alvarez or Davenport) satisfies all constraints.

### Example 2: N-Queens Problem (MiniZinc)

**User**

> Check whether you can place n Queens on an nxn chessboard. Try n=10,20,30,40 and compare the solving times

**System:** **`clear_model`**

> ```
> Model cleared
> ```

**System:** **`add_item`** (multiple)

> ```minizinc
> int: n = 10;
> array[1..n] of var 1..n: queens;
> constraint alldifferent(queens);
> constraint alldifferent([queens[i] + i | i in 1..n]);
> constraint alldifferent([queens[i] - i | i in 1..n]);
> solve satisfy;
> ```

**System:** **`solve_model`** (with n=10, 20, 30, and 40 by changing the value of n)

Solve times comparison: n=10: 0.001s n=20: 0.005s n=30: 0.040s n=40: 0.043s

Solutions exist for all tested board sizes. Time complexity increases non-linearly but remains manageable up to n=40.

### Example 3: Traveling Salesperson Problem (MiniZinc)

**User**

> A saleswoman based in Vienna needs to plan her upcoming tour through Austria, visiting each province capital once. Help find the shortest route. Distances in km: 1=Vienna, 2=St. Pölten, 3=Eisenstadt, 4=Linz, 5=Graz, 6=Klagenfurt, 7=Salzburg, 8=Innsbruck, 9=Bregenz
>
> [Distance matrix table provided in user message]

**System:** **`add_item`** (multiple)

> ```minizinc
> include "globals.mzn";
> int: n = 9;
> array[1..n,1..n] of int: dist = [|0,65,60,184,195,319,299,478,631|65,0,125,119,130,254,234,413,566|60,125,0,184,157,281,261,440,593|184,119,184,0,208,252,136,315,468|195,130,157,208,0,136,280,459,629|319,254,281,252,136,0,217,391,566|299,234,261,136,280,217,0,188,343|478,413,440,315,459,391,188,0,157|631,566,593,468,629,566,343,157,0|];
> array[1..n] of var 1..n: succ;
> var int: total_dist = sum(i in 1..n)(dist[i,succ[i]]);
> constraint circuit(succ);
> solve minimize total_dist;
> ```

**System:** **`solve_model`**

> ```
> {'status': 'SAT', 'solution': Solution(objective=1564, succ=[2, 4, 1, 7, 3, 5, 9, 6, 8], _checker=''), 'solve_time': 0.002}
> ```

Optimal route: Vienna → St. Pölten → Linz → Salzburg → Bregenz → Innsbruck → Klagenfurt → Graz → Eisenstadt → Vienna with a total distance of 1,564 km.

## Feedback

You can provide feedback to the author via this [form](https://form.jotform.com/szeider/mcp-solver-feedback-form).

## Disclaimer

This MCP Solver is in its prototype stage and should be used with caution. Users are encouraged to experiment, but any use in critical environments is at their own risk.

------

## License

This project is licensed under the MIT License - see the [LICENSE](https://claude.ai/chat/LICENSE) file for details.

------