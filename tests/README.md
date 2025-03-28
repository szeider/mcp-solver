# MCP Solver Test Suite

This directory contains tests for the MCP Solver project.

## Quick Tests

For a quick end-to-end test after code changes, run any of these commands:

```bash
# MiniZinc test - N-Queens problem
uv run test-client-mzn --problem nqueens

# PySAT test - Graph Coloring problem
uv run test-client-pysat --problem graph_coloring

# Z3 test - Cryptarithmetic puzzle
uv run test-client-z3 --problem cryptarithmetic
```

## Test Structure

```
├── tests/
│   ├── __init__.py                # Package marker
│   ├── conftest.py                # Shared test fixtures and configuration
│   ├── test_config.py             # Configuration values for tests
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
│   ├── run_all_tests.py           # Main runner for all tests
│   ├── run_test_mzn.py            # Runner for MiniZinc tests
│   ├── run_test_pysat.py          # Runner for PySAT tests
│   ├── run_test_z3.py             # Runner for Z3 tests
│   └── results/                   # Directory for test results (optional)
```

## Running Tests

### Running All Tests

```bash
cd /path/to/mcp-solver
uv run python tests/run_all_tests.py
```

### Running Tests for a Specific Solver

```bash
cd /path/to/mcp-solver
uv run python tests/run_all_tests.py --mzn   # Run only MiniZinc tests
uv run python tests/run_all_tests.py --pysat # Run only PySAT tests
uv run python tests/run_all_tests.py --z3    # Run only Z3 tests
```

### Running a Specific Problem

```bash
cd /path/to/mcp-solver
uv run python tests/run_test_mzn.py --problem nqueens
uv run python tests/run_test_pysat.py --problem graph_coloring
uv run python tests/run_test_z3.py --problem cryptarithmetic
```

You can also use the client directly:

```bash
uv run test-client-mzn --problem nqueens
uv run test-client-pysat --problem graph_coloring
uv run test-client-z3 --problem cryptarithmetic
```

### Available Problems

#### MiniZinc Problems:
- `nqueens` - The N-Queens problem
- `sudoku` - Sudoku solver

#### PySAT Problems:
- `graph_coloring` - Graph coloring problem
- `scheduling` - Scheduling problem
- `furniture-arrangement` - Furniture arrangement problem
- `mine-sweeper-hard` - Mine sweeper problem
- `sudoku-16x16` - 16x16 Sudoku problem

#### Z3 Problems:
- `bounded_sum` - Bounded sum problem
- `cryptarithmetic` - Cryptarithmetic problem (SEND+MORE=MONEY)

### Test Options

- `--verbose` or `-v`: Enable verbose output
- `--timeout` or `-t`: Set timeout in seconds (default: 300)
- `--save` or `-s`: Save test results to the results directory

Example:
```bash
uv run python tests/run_test_mzn.py --problem nqueens --verbose --timeout 120 --save
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
2. Run the test with the corresponding test runner or client command 