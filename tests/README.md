# MCP Solver Test Suite

This directory contains tests for the MCP Solver project.

## Test Structure

```
├── tests/
│   ├── __init__.py            # Package marker
│   ├── conftest.py            # Shared test fixtures and configuration
│   ├── test_config.py         # Configuration values for tests
│   ├── problems/              # All problem definitions in one place
│   │   ├── mzn/               # MiniZinc problem definitions
│   │   │   └── nqueens.md     # N-Queens problem
│   │   ├── pysat/             # PySAT problem definitions
│   │   │   └── graph_coloring.md  # Graph coloring problem
│   │   └── z3/                # Z3 problem definitions
│   │       └── cryptarithmetic.md # Cryptarithmetic problem
│   ├── run_all_tests.py       # Main runner for all tests
│   ├── run_test_mzn.py        # Runner for MiniZinc tests
│   ├── run_test_pysat.py      # Runner for PySAT tests
│   ├── run_test_z3.py         # Runner for Z3 tests
│   ├── test_mzn.py            # Unit tests for MiniZinc
│   ├── test_pysat.py          # Unit tests for PySAT
│   ├── test_z3.py             # Unit tests for Z3
│   └── test_integration.py    # Cross-solver tests
```

## Running Tests

### Running All Tests

```bash
cd tests
uv run python run_all_tests.py
```

### Running Tests for a Specific Solver

```bash
cd tests
uv run python run_all_tests.py --mzn   # Run only MiniZinc tests
uv run python run_all_tests.py --pysat # Run only PySAT tests
uv run python run_all_tests.py --z3    # Run only Z3 tests
```

### Running a Specific Problem

```bash
cd tests
uv run python run_test_mzn.py --problem nqueens
uv run python run_test_pysat.py --problem graph_coloring
uv run python run_test_z3.py --problem cryptarithmetic
```

### Running Individual Tests

You can run specific test files directly with Python:

```bash
# In the project root directory
uv run python -m tests.test_environment_imports
uv run python -m tests.test_pysat_constraints
uv run python -m tests.test_pysat_solution_export
```

Or from the tests directory:

```bash
cd tests
uv run python test_environment_imports.py
uv run python test_pysat_constraints.py
```

### Custom Test Scripts

For quick testing during development, you can create and run simple test scripts in the project root:

```bash
# Custom test scripts
uv run python pysat_import_test2.py  # Test PySAT import functionality
```

### Options

- `--verbose` or `-v`: Enable verbose output
- `--timeout` or `-t`: Set timeout in seconds (default: 300)

## Troubleshooting Common Issues

### Import Errors

If you encounter `ModuleNotFoundError` when running tests with pytest, try running the tests as Python modules instead:

```bash
# Instead of:
pytest tests/test_environment_imports.py

# Use:
python -m tests.test_environment_imports
```

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

1. Create a Markdown file in the appropriate problem directory
2. Run the test with the corresponding test runner

### Adding a New Unit Test

1. Add test functions to the appropriate test file
2. Test functions should start with `test_` 