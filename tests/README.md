# MCP Solver Tests

This directory contains test scripts for verifying various aspects of the MCP Solver project.

## Main Test Files

- **test_all_modes.py**: Comprehensive test script that verifies all three solver modes can start correctly:
  - MiniZinc mode (default)
  - Z3 mode (--z3 --lite)
  - PySAT mode (--pysat --lite)

- **test_pysat_basic.py**: Tests basic PySAT integration and functionality, including:
  - CNF formula creation
  - SAT solving
  - Solution extraction
  - Memory management
  - Error handling

## Functional Tests

- **test_templates.py**: Tests template functions for constraint solving models
- **test_function_templates.py**: Validates function templates for user-defined functions
- **test_map_coloring.py**: Tests the map coloring problem implementation
- **test_mcp_map_coloring.py**: Tests the MCP-specific map coloring implementation
- **test_us_states_coloring.py**: Advanced map coloring problem for US states
- **test_subset_template.py**: Tests subset problem templates

## Running Tests

Run individual tests with:
```bash
uv run python tests/test_all_modes.py
```

Run all tests with:
```bash
for test in tests/test_*.py; do uv run python $test; done
```

## Adding New Tests

When adding new tests, please follow these guidelines:
1. Name files with the `test_` prefix
2. Include documentation on the test's purpose
3. Use proper error handling and exit codes
4. Include the file in this README under the appropriate section 