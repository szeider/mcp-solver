# Changelog

### [4.4.0] - 2026-11-14

- **Major Change:** The helper library `mcp_solver.helpers` (modules for
  pysat, maxsat and z3) is removed; it had known defects, among them
  auxiliary-variable collisions that could give a wrong UNSAT. Programs
  submitted under earlier versions that import `mcp_solver.helpers` fail with
  `ModuleNotFoundError`. Migration: name variables through
  `pysat.formula.IDPool` and pass `vpool=` to `CardEnc`/`PBEnc`; for z3, use
  plain z3 idioms.
- **Behavior Change:** Solve kernels and the CLI's re-execution install only
  the backend's solver libraries; they no longer install this distribution
  (formerly the checkout in dev mode, else `agentic-mcp-solver==<version>`
  from PyPI). Dev mode (`--dev`, `MCP_SOLVER_DEV`) now only reads the
  templates from a checkout, and an unpublished version without dev mode is
  no longer an error (formerly exit 2 in the CLI and a tool error in
  `select_backend`).
- **Improvement:** The pysat, maxsat and z3 templates no longer use the
  helpers (`IDPool` with `CardEnc`/`PBEnc` and `vpool=`, plain z3 idioms).
- **Fix:** The clingo template is repaired (dead v3 content removed, snippets
  and claims checked against clingo 5.8.3) and under 50,000 characters, so
  Claude Code passes the whole `select_backend` result to the host instead of
  a 2 KB preview; a unit test checks the size for every backend.
- **Fix:** The snippets and library claims of all templates were checked by
  execution and repaired where wrong.
- **Fix:** The CLI's re-execution of the submitted program has a time limit
  of 360 seconds (the kernel's 300-second execution maximum plus time for uv);
  a program that runs longer is killed (on macOS and Linux with its child
  processes; on Windows only the direct child) and the CLI exits with the new
  code 124 and a message on stderr (formerly it waited without limit). A
  SIGTERM during re-execution kills the program too and exits with 143.
- **Fix:** `mcp-solver-bench` kills the whole process group of a timed-out
  solve on macOS and Linux (uv, python, engine), not only the CLI process;
  the result row is unchanged.
- **Fix:** `mcp-solver-serve`: a `python_exec` that follows a timed-out one
  first probes the kernel. If code that ignores interrupts still holds the
  kernel, the new code is not run and the reply says so and points to
  `python_reset`. Previously the new call was reported as timed out although
  it never ran.

### [4.3.0] - 2026-11-14

- **Behavior Change:** Requires the MCP Python SDK 2.x (`mcp>=2,<3`; 1.x is no
  longer supported), agentic-python-coder 3.5.0 and mcp-minion 0.3.0.
- **Behavior Change:** `mcp-solver` caps a run at 2,000,000 total (input +
  output) tokens by default, checked after each step; `--token-limit N`
  changes the cap (`0` = no limit). A run stopped by the cap without an
  accepted submission exits with code 3; the stats JSON reports
  `token_limit_reached` and `token_limit`. `mcp-solver-bench --token-limit`
  passes the value to each solve.
- **Behavior Change:** New exit code 4: the agent stopped early with an error
  (for example an LLM API failure) after an accepted submission. That
  submission is saved and re-executed, and exit 4 replaces 0 when it runs
  cleanly, since it may not be the agent's final, verified program. Such an
  error before any accepted submission, a failed kernel setup, or an
  exception during the solve exits with 1 and a one-line message instead of
  a traceback.
- **Behavior Change:** The engine, its kernels and the CLI's re-execution of
  the submitted program (in `mcp-solver` and `mcp-solver-serve`) get a
  filtered environment: the mcp library's default whitelist (`PATH`, `HOME`,
  ...), locale, temp-dir, proxy, CA-bundle and native-build variables, and
  variables starting with `UV_`, `CODER_`, `MCP_SOLVER_`, `XDG_` or `LC_`
  unless the name contains `KEY`, `TOKEN`, `SECRET`, `PASSWORD` or
  `CREDENTIAL` (`UV_ENV_FILE` is dropped too). Other variables no longer
  reach the kernel. This keeps API keys out of model-written code; it is
  hygiene, not a sandbox.
- **Behavior Change:** `submit_code` on the toolkit server returns an MCP
  error result (`isError`) carrying the engine's reason for a rejected
  program, instead of a normal result with `{"ok": false, ...}`.
- **New Feature:** `--model` accepts a model config file
  (`--model my_model.json`) in the engine's format, and aliases resolve
  through the engine's `load_model_config`, so a local `<alias>.json`
  overrides a bundled one. A model config may set `base_url` and
  `api_key_env` (another OpenAI-compatible endpoint and the variable that
  holds its key). Its `reasoning`, `provider`, `top_k` and `request_timeout`
  settings are now sent; before they were dropped.
- **Behavior Change:** `mcp-solver-runfolder --model` resolves aliases and
  model files like the CLI. A full OpenRouter ID gets no sampling settings
  (it used to get `temperature 0` and `max_tokens 8192`), a model with
  `base_url` or `api_key_env` is rejected, and the generated config sets
  `agent.max_tool_result_chars` to at least twice the backend template's
  length (minimum 64,000).
- **Improvement:** `mcp-solver-bench` runs each solve with its own run
  directory as working directory (path arguments made absolute), saves the
  solve's stderr to `cli_stderr.txt` there, and copies its last 20 lines
  into the result row (`stderr_tail`) of a failed or invalid run.
- **Improvement:** The CLI and the toolkit server check that the engine
  offers all five tools they use (`python_exec`, `python_reset`,
  `python_status`, `python_interrupt`, `submit_code`) and name the missing
  ones together with the required engine version.
- **Fix:** The pysat and maxsat kernels install `python-sat[pb]` instead of
  `python-sat` and `pypblib`. The `[pb]` extra installs pindakaas, without
  which every `pysat.pb.PBEnc` encoding call raises.
- **Fix:** In dev mode, newly started kernels pick up edits to the helper
  sources (the uv build cache is keyed on `src/**/*.py`, not only on
  `pyproject.toml`).

### [4.2.0] - 2026-11-13

- **Major Change:** Published on PyPI as `agentic-mcp-solver`, since the name
  `mcp-solver` is held by an unrelated upload (PEP 541 request pending). The
  import package and the commands keep their names. Solve kernels fetch the
  helper library as `agentic-mcp-solver==<version>`.
- **New Feature:** Supports the MCP Python SDK 2.x alongside 1.x
  (`mcp>=1.25,<3`); requires agentic-python-coder 3.4.4 and mcp-minion 0.2.2.
  Resource errors (for example an unknown solver) now raise `ResourceError`,
  so their text reaches the client under mcp 2.x.
- **Behavior Change:** The didp backend requires didppy 0.11.1 or later
  (`>=0.11.1,<0.12`).

### [4.1.0] - 2026-07-22

- **New Feature:** DIDP backend (domain-independent dynamic programming via
  didppy): state-based modeling with dual-bound and dominance safety rules in
  the template, plus four benchmark problems (knapsack, talent scheduling,
  weighted tardiness, TSPTW) with in-validator exact oracles. The bundled
  benchmark now spans 30 problems across six backends.

### [4.0.0] - 2026-07-14

- **Major Change:** Complete re-architecture. The solving LLM now writes, runs,
  and verifies a real solver program in a persistent IPython kernel (the kernel
  server of agentic-python-coder) instead of editing a model item by item
  through protocol tools. The final program is submitted through a
  syntax-checked gate (`submit_code`), persisted, and re-executed to produce
  the solution JSON.
- **Major Change:** The MCP server (`mcp-solver-serve`) is a solver toolkit for
  the host LLM — it runs no LLM and needs no API key: `select_backend` sets up
  a kernel with the solver library, helpers, and modeling instructions; kernel
  tools and `submit_code` do the rest.
- **Major Change:** MiniZinc mode retired; CPMpy added. Backends are now
  PySAT, MaxSAT, Z3, CPMpy, and Clingo. The v3 MCP server lives on unchanged
  on the `v3` branch.
- **New Feature:** `mcp-solver` CLI and `mcp-solver-bench` benchmark harness,
  driven by the bundled **mcp-minion** ReAct agent (also published separately
  on PyPI) via OpenRouter.
- **New Feature:** Dependency-free helper library (`mcp_solver.helpers`)
  injected into solve-time kernels via `uv run --with`.
- **Improvement:** Every shipped test problem carries an output-format contract
  and a semantic ground-truth validator; validated end-to-end on the bundled
  26-problem benchmark and externally on CP-Bench and ASP-Bench.
- **Note:** Not yet installable from PyPI: the name `mcp-solver` is held by an
  unrelated third-party upload of stale v2.0.0 code, and PyPI blocks
  confusably similar names; a PEP 541 name-transfer request is pending. The
  companion agent **mcp-minion** is on PyPI.

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