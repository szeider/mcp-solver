# MCP Solver

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) [![Python Version](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)

LLM-driven constraint solving (SAT, MaxSAT, SMT, CP, ASP). An LLM agent
writes, runs, verifies, and saves a real solver program for your problem, then
prints the solution.

> **v4 is a complete re-architecture.** Earlier versions of MCP Solver were an
> MCP *server* that exposed model-editing tools (`add_item`, `solve_model`, …)
> to a chat client. v4 instead runs **mcp-minion**, a minimal ReAct agent (a
> workspace package in this repo, [`minion/`](minion/)), against the IPython
> kernel MCP server of
> [agentic-python-coder](https://github.com/szeider/agentic-python-coder): the
> agent writes an actual Python solver program in a persistent IPython kernel,
> executes it, checks the result, and submits it via the server's
> `submit_code` tool. The CLI persists the submission and re-executes it for
> the final answer. The MCP protocol surface returns as **`mcp-solver-serve`**,
> a thin server exposing a single `solve` tool that delegates to the same
> agent (see [MCP server](#mcp-server) below).
>
> **Using v3?** The previous MCP-server implementation (MiniZinc, PySAT,
> MaxSAT, Z3, ASP, and the ReAct test client) lives on unchanged on the
> **`heritage`** branch.

## What it is

You give MCP Solver a problem in natural language and pick a backend. It hands
the problem, together with a backend-specific *project template*, to the
mcp-minion agent. The agent encodes the problem, runs it through the real
solver inside an ephemeral `uv` kernel, sanity-checks the output, and submits
the finished program. The CLI writes it to your working directory (e.g.
`n_queens_code.py`), re-executes it in a fresh kernel, and prints the solution
as JSON on stdout.

For background, see Stefan Szeider,
["Bridging Language Models and Symbolic Solvers via the Model Context
Protocol"](https://doi.org/10.4230/LIPIcs.SAT.2025.30), SAT 2025 (the original
MCP-Solver paper), and the architecture paper behind the v4 engine,
[arXiv:2508.07468](https://arxiv.org/abs/2508.07468) (agentic-python-coder).

## Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/) (a hard runtime requirement — solver
  libraries are supplied at solve time via `uv run --with`)
- An [OpenRouter](https://openrouter.ai) API key

## Quick start

> **v4 is not yet on PyPI.** PyPI still serves the old v3 server, so
> `uv pip install "mcp-solver[agent]"` installs the wrong software. Until v4 is
> published, the only working install is a clone plus an editable install, shown
> below.

```bash
# Clone and install the product layer (CLI + templates + agent + kernel server)
git clone https://github.com/szeider/mcp-solver.git
cd mcp-solver
uv pip install -e ".[agent]"

# Provide your OpenRouter key
mkdir -p ~/.config/coder
echo 'OPENROUTER_API_KEY="sk-or-v1-..."' > ~/.config/coder/.env
```

Once v4 is published, `uv pip install "mcp-solver[agent]"` will install it directly.

Solve a problem from inline text:

```bash
mcp-solver pysat "Place N non-attacking queens on an 8x8 board and give one solution."
```

Or from a markdown file (canonical example, shipped in the repo):

```bash
mcp-solver pysat --problem tests/problems/pysat/n_queens.md
```

From an editable clone like the one above, the CLI runs in **dev mode**: it
takes both the helper library and the solver templates from the source
checkout (auto-detected), so no extra flags are needed and a single provenance
line is printed to stderr. Pass `--dev PATH` (or set `MCP_SOLVER_DEV`) to point
at a checkout explicitly, or `--no-dev` to force the published (PyPI-pinned)
helpers instead.

The solution JSON goes to stdout; the submitted solver program (e.g.
`n_queens_code.py`) and a complete run log (`run_*.json`) land in the working
directory.

See [INSTALL.md](INSTALL.md) for full installation and troubleshooting.

## Backends

Each backend is selected as the first CLI argument. The right solver library is
injected into the solve-time kernel automatically.

| Backend  | Domain                          | Solver library      |
| -------- | ------------------------------- | ------------------- |
| `pysat`  | Boolean satisfiability (SAT)    | python-sat          |
| `maxsat` | Weighted optimization           | python-sat (RC2)    |
| `z3`     | SMT solving / verification      | z3-solver           |
| `cpmpy`  | Constraint programming (CP)     | cpmpy               |
| `clingo` | Answer Set Programming (ASP)    | clingo              |

MiniZinc mode from v3 is retired; constraint programming is now covered by
CPMpy.

## CLI

```
mcp-solver <solver> "task text" [options]
mcp-solver <solver> --problem FILE.md [options]
```

| Option           | Description                                            |
| ---------------- | ------------------------------------------------------ |
| `--problem FILE` | Read the task from a markdown file instead of text     |
| `--model NAME`   | Agent model: an alias like `gpt56terra` or a full OpenRouter ID like `openai/gpt-5.6-terra` (default: `gpt56terra`) |
| `--workdir DIR`  | Working directory for the run (default: current dir)   |
| `--step-limit N` | Maximum agent steps before stopping (default: 30)      |
| `-q`, `--quiet`  | Suppress per-step progress output                      |
| `--dev [PATH]`   | Dev mode: take helpers and templates from a local checkout (bare `--dev` auto-detects; auto-on inside a checkout; also `MCP_SOLVER_DEV`) |
| `--no-dev`       | Disable dev mode; use the published (PyPI-pinned) helpers |
| `--stats-json FILE` | Write the run statistics to `FILE` as JSON          |

Model aliases resolve via the model files bundled with agentic-python-coder.
The `OPENROUTER_API_KEY` is read from the environment, `~/.config/coder/.env`,
or `~/.mcp-minion`.

## MCP server

`mcp-solver-serve` exposes the same solving pipeline over the Model Context
Protocol (stdio), so MCP hosts like Claude Desktop, Claude Code, or Cursor can
delegate constraint problems to it:

- **One tool:** `solve(solver, problem)` — returns the solution JSON plus
  `resource_link`s to the generated solver program and the full run log.
- **One resource:** `mcp-solver://guide` — a short backend-selection guide the
  host model can consult before calling.
- Per-step progress notifications during the 30–120 s a solve typically takes.

Claude Desktop configuration (once v4 is on PyPI):

```json
{
  "mcpServers": {
    "mcp-solver": {
      "command": "uvx",
      "args": ["--from", "mcp-solver[agent]", "mcp-solver-serve"]
    }
  }
}
```

From a checkout (the standard setup today, and the development path always),
use `"command": "uv"`, `"args": ["run", "--project", "/path/to/mcp-solver",
"mcp-solver-serve"]` instead — the server then runs entirely from the local
version, no PyPI involved (see
[INSTALL.md](INSTALL.md#using-the-local-version-no-pypi-involved)). The server
reads the same `OPENROUTER_API_KEY` locations as the CLI.

## How it works

The base `mcp-solver` package is a dependency-free **solver helper library**
(`mcp_solver.helpers.{pysat,maxsat,z3}`). It is never pip-installed into your
environment; instead it is injected into each solve-time kernel via
`uv run --with mcp-solver==<version>`, alongside the backend's solver library.
This keeps the host environment clean and each solve reproducible.

Each backend ships a **project template** — a markdown prompt (package data)
that tells the agent the encoding conventions, output format, and helper API for
that solver. When you run `mcp-solver <solver>`, the CLI loads that template as
the system prompt of an **mcp-minion** agent (a minimal ReAct loop over MCP,
developed in this repo under [`minion/`](minion/)) and connects it over stdio
to the `ipython_mcp` kernel server from
[agentic-python-coder](https://github.com/szeider/agentic-python-coder). The
agent drives the write → execute → verify loop in a persistent IPython kernel
and must end the solve by calling the server's `submit_code` tool with the
final self-contained program (syntax-checked server-side, never written to
disk by the kernel). The CLI extracts the last successful submission from the
run, writes it to `<basename>_code.py`, and re-executes it in a fresh `uv`
kernel to produce the answer.

## Benchmark harness

`mcp-solver-bench` runs the bundled test problems in `tests/problems/<solver>/`
end-to-end and validates each result against a per-problem
`*_ground_truth.py` validator (which reads the solution JSON on stdin and
returns `{"valid": ..., "message": ...}`).

```bash
# Benchmark every backend
mcp-solver-bench

# Specific backends, several runs each, in parallel
mcp-solver-bench pysat z3 --runs 3 --jobs 4
```

Each run is bounded by `--step-limit` (default: 30 agent steps), and results are
appended to `results.jsonl` in the output directory. Like the main CLI, the
harness runs in dev mode from an editable checkout; pass `--dev [PATH]` to
force a specific checkout (otherwise the CLI decides).

Current status: 26/26 test problems solve correctly with `gpt-5.6-terra`.

## Citations

- Stefan Szeider, "Bridging Language Models and Symbolic Solvers via the Model
  Context Protocol", SAT 2025.
  [DOI 10.4230/LIPIcs.SAT.2025.30](https://doi.org/10.4230/LIPIcs.SAT.2025.30)
- Stefan Szeider, [arXiv:2508.07468](https://arxiv.org/abs/2508.07468) — the
  agentic-python-coder architecture behind the v4 engine.

## License

MIT — see [LICENSE](LICENSE).
