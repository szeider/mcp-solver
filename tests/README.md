# MCP Solver Test Suite

This directory contains tests for the MCP Solver project.

## Quick Tests

For a quick end-to-end test after code changes, run any of these commands:

```bash
# MiniZinc test 
uv run run-test mzn 

# PySAT test 
uv run run-test pysat 

# Z3 test - Cryptarithmetic puzzle
uv run run-test z3 
```

## Test Structure

```
├── tests/
│   ├── __init__.py                # Package marker
│   ├── test_config.py             # Configuration values for tests
│   ├── run_test.py                # Unified test runner for all solvers
│   ├── problems/                  # All problem definitions in one place
│   │   ├── mzn/                   # MiniZinc problem definitions
│   │   │   ├── nqueens.md         # N-Queens problem
│   │   │   └── sudoku.md          # Sudoku problem
│   │   ├── pysat/                 # PySAT problem definitions
│   │   │   ├── graph_coloring.md  # Graph coloring problem
│   │   │   ├── scheduling.md      # Scheduling problem
│   │   │   ├── furniture-arrangement.md  # Furniture arrangement problem
│   │   │   ├── mine-sweeper-hard.md      # Mine sweeper problem
│   │   │   └── sudoku-16x16.md    # 16x16 Sudoku problem
│   │   └── z3/                    # Z3 problem definitions
│   │       ├── bounded_sum.md     # Bounded sum problem
│   │       └── cryptarithmetic.md # Cryptarithmetic problem
│   └── results/                   # Directory for test results (optional)
```

## Running Tests

### Running Tests for a Specific Solver

```bash
cd /path/to/mcp-solver
uv run run-test mzn    # Run MiniZinc test.md by default if present, otherwise all MiniZinc tests
uv run run-test pysat  # Run PySAT test.md by default if present, otherwise all PySAT tests
uv run run-test z3     # Run Z3 test.md by default if present, otherwise all Z3 tests
```

### Running a Specific Problem

```bash
cd /path/to/mcp-solver
uv run run-test mzn --problem tests/problems/mzn/nqueens.md
uv run run-test pysat --problem tests/problems/pysat/graph_coloring.md
uv run run-test z3 --problem tests/problems/z3/cryptarithmetic.md
```

Note: If no problem is specified, the system will look for a `test.md` file in the respective solver's problems directory and run that as a default test.

### Available Problems

#### MiniZinc Problems:
- `carpet_cutting.md` - Carpet cutting optimization problem
- `test.md` - Default test problem
- `tsp.md` - Traveling Salesperson Problem
- `university_scheduling.md` - University course scheduling problem
- `university_scheduling_unsat.md` - Unsatisfiable variant of scheduling problem
- `zebra.md` - Einstein's Zebra puzzle (Five Houses Puzzle)

#### PySAT Problems:
- `equitable_coloring_hajos.md` - Equitable graph coloring problem
- `furniture_arrangement.md` - Furniture arrangement problem
- `petersen_12_coloring_unsat.md` - Unsatisfiable Petersen graph coloring
- `queens_and_knights_6x6.md` - Combined queens and knights placement puzzle
- `sudoku_16x16.md` - 16x16 Sudoku problem
- `test.md` - Default test problem

#### Z3 Problems:
- `array_property_verifier.md` - Array property verification problem
- `bounded_sum_unsat.md` - Unsatisfiable bounded sum problem
- `cryptarithmetic.md` - Cryptarithmetic puzzle (SEND+MORE=MONEY)
- `processor_verification.md` - Processor behavior verification
- `sos_induction.md` - Sum-of-squares induction problem
- `test.md` - Default test problem

### Test Options

- `--verbose` or `-v`: Enable verbose output
- `--timeout` or `-t`: Set timeout in seconds (default: 300)
- `--save` or `-s`: Save test results to the results directory
- `--result`: Save detailed JSON results to the specified directory
- `--mc`: Specify direct model code (e.g., "AT:claude-3-7-sonnet-20250219")

Examples:
```bash
# Run with all default options
uv run run-test mzn --problem tests/problems/mzn/nqueens.md --verbose --timeout 120 --save --result ./json_results

# Run with specific LLM model (using direct model code)
uv run run-test mzn --problem tests/problems/mzn/nqueens.md --mc "AT:claude-3-7-sonnet-20250219"
```

## Troubleshooting Common Issues

### Error connecting to MCP server

If you see "Error connecting to MCP server", check that:
1. The server command is correctly set
2. The appropriate solver package is installed
3. Environment variables are properly set if needed

### Missing prompt files

If you see a warning about missing prompt files, check that the instruction prompt files exist:
- MiniZinc: `instructions_prompt_mzn.md`
- PySAT: `instructions_prompt_pysat.md`
- Z3: `instructions_prompt_z3.md`

### PySAT Environment

The PySAT execution environment includes:

1. Standard Python modules: `math`, `random`, `collections`, `itertools`, `re`, `json`
2. PySAT modules: `pysat.formula`, `pysat.solvers`, `pysat.card`
3. Constraint helpers: `at_most_one`, `exactly_one`, `implies`, etc.
4. Cardinality templates: `at_most_k`, `at_least_k`, `exactly_k`

If you add new helper functions, make sure to include them in:
- `restricted_globals` dictionary in `environment.py`
- The processed code template in `execute_pysat_code`

## Adding New Tests

### Adding a New Problem

1. Create a Markdown file in the appropriate problem directory under `tests/problems/`:
   - MiniZinc: `tests/problems/mzn/`
   - PySAT: `tests/problems/pysat/`
   - Z3: `tests/problems/z3/`
2. Run the test with the `uv run run-test` command 