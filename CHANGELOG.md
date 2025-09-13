# Changelog

### [3.4.0] - 2025-09-13

- **New Feature:** Complete Answer Set Programming (ASP) mode implementation with Clingo solver (contributed by Luis Angel Rodriguez Reiners)
  - Full ASP syntax support including facts, rules, constraints, and choice rules
  - Support for optimization statements (#maximize, #minimize) and weak constraints
  - Enhanced error handling with detailed validation messages
  - Six comprehensive test problems demonstrating ASP capabilities

### [3.3.0] - 2025-06-10

- **New Feature:** Added MaxSAT as a 4th mode for weighted optimization problems using RC2 solver
- **Improvement:** Implemented exact token counting with callback handlers for better usage tracking
- **Update:** Replaced Black with Ruff for code formatting and linting
- **Improvement:** Enhanced MaxSAT and Z3 instructions with clearer guidance and examples
- **Fix:** Resolved error handling issues in PySAT/MaxSAT to preserve UNSAT status correctly
- **Update:** Added MaxSAT test infrastructure and aligned test problems with experiment6
- **Improvement:** Added cardinality constraint templates and helper functions for MaxSAT

### [3.2.0] - 2025-05-11

- **Improvement:** Simplified model specification by removing internal model code shortcuts (MC1, MC2, etc.)
- **Improvement:** Updated client to use direct model codes (AT:claude-3-7-sonnet-20250219, etc.) for better clarity
- **Update:** Enhanced documentation for model code formats in README.md and INSTALL.md
- **Update:** Added support for LM Studio local models

---

### [3.1.0] - 2025-04-03

- **Improvement:** Enhanced test client with additional features for better usability and testing workflows.
- **Improvement:** Improved solution export for Z3 mode, providing better data representation and accessibility.
- **Update:** Enhanced installation documentation with more detailed setup notes and requirements

### [3.0.0] - 2025-03-28

- **Major Change:** Added PySAT mode and Z3 mode, expanding the supported constraint programming paradigms.
- **Major Change:** Added a standalone test client for easier testing and demonstration.
- **Major Change:** Lite mode is now the default mode. The additional tools have been removed from the default configuration.
- **Update:** The server now advertises only a reduced set of tools by default (clear_model, add_item, replace_item, delete_item, and solve_model).

### [2.3.0] - 2025-02-28

- **New Feature:** Introduced Lite Mode for the MCP Solver. When run with the `--lite` flag, the server advertises only a reduced set of tools (clear_model, add_item, replace_item, delete_item, and solve_model).
- **New Feature:** In Lite Mode, the `solve_model` tool returns only the status (and the solution if SAT) without additional metadata.
- **New Feature:** Mode-specific instruction prompts are used: `instructions_prompt_mzn.md` for MiniZinc, `instructions_prompt_pysat.md` for PySAT, and `instructions_prompt_z3.md` for Z3.

### [2.2.0] - 2025-02-15

- **New Feature:** Integrated static prompt endpoints (`prompts/list` and `prompts/get`) to advertise MCP prompt templates ("quick_prompt" and "detailed_prompt") without requiring any arguments.
- **New Feature:** Advertised detailed tool capabilities by adding descriptive metadata for each tool in the server's capabilities declaration.
- **Improvement:** Enhanced error reporting for tool endpoints with improved logging and standardized error responses.
- **Update:** Refactored server initialization to explicitly log the declared capabilities for greater transparency and easier debugging.

### [2.1.0] - 2025-02-09

- **Update:** Change minimum Python requirement to 3.11+ (to support `asyncio.timeout`).
- **Update:** Bump dependency on `mcp` to version 1.2.0 or later.
- **Improvement:** Update tool handler messages so that "delete_item" and "replace_item" commands correctly report the operation performed.
- **Update:** Miscellaneous documentation and cleanup.

### [2.0.0] - 2024-12-29

- Major change: Use item-based editing.

### [1.0.0] - 2024-12-21

- Major change: Use line-based model editing.
- Makes parameter handling obsolete.
- Added dynamic knowledge base handling.

### [0.2.1] - 2024-12-16

- Changed parameter handling.

### [0.2.0] - 2024-12-15

- Initial release.