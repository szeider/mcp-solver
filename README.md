# MCP Solver

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) [![Python Version](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)

An MCP server for constraint solving (SAT, MaxSAT, SMT, CP, ASP). It turns
the connected LLM host into a solver-writing agent: the host gets a Python
kernel preloaded with a real solver library, modeling instructions for the
chosen backend, and a submission gate. The host encodes the problem, runs
and verifies it against the real solver, and submits the final program —
the outcome is the solution plus the verified solver program that produced
it.

> **Version 4 is a complete re-architecture.** Both the MCP interface and the
> solving engine changed; the design from the SAT 2025 paper (v3) lives on
> unchanged on the
> [`heritage`](https://github.com/szeider/mcp-solver/tree/heritage) branch.
> See [From v3 to v4](#from-v3-to-v4-a-new-architecture) below.

## The MCP server

`mcp-solver-serve` runs over stdio and works with any MCP host: Claude
Desktop, Claude Code, Cursor, or your own client. The host LLM does the
solving itself — the server is a solver toolkit; it runs no LLM and needs
no API key:

- **`select_backend(solver)`** sets up a persistent IPython kernel with the
  backend's solver library and helper functions, and returns the modeling
  instructions for that backend. Calling it again recycles the kernel for
  the next problem.
- **Kernel tools** (`python_exec`, `python_reset`, `python_status`,
  `python_interrupt`) let the host write, run, and verify a real solver
  program; bare `python_exec` calls are routed to the solving kernel
  automatically.
- **`submit_code(code)`** is the finish line: the final self-contained
  program is syntax-checked and, on success, stored and linked back as an
  MCP resource (`mcp-solver://submissions/{id}`), with the verdict as
  structured content.
- **Resources:** `mcp-solver://guide` (backend selection and workflow) and
  `mcp-solver://template/{solver}` (the full modeling instructions,
  browsable without selecting).
- **Statistics** (optional): set `MCP_SOLVER_STATS=/path/to/stats.jsonl` in
  the server's `env` to log one JSON line per solving episode — tool-call
  counts, execution failures, submissions, wall time. Host tokens are
  invisible to the server by protocol design; tool usage is the comparable
  metric across hosts. (`submit_code` results also carry a compact stats
  snapshot as structured content; the CLI path reports tokens and actual
  OpenRouter cost via `--stats-json`.)

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

From a checkout (the standard setup today, and the development path always):

```json
{
  "mcpServers": {
    "mcp-solver": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/mcp-solver", "mcp-solver-serve"]
    }
  }
}
```

The server needs no API key — the model doing the solving belongs to the
host. An [OpenRouter](https://openrouter.ai) key is required only for the
[command-line path](#command-line-use) below, which brings its own agent.

## Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/) (a hard runtime requirement: solver
  libraries are supplied at solve time via `uv run --with`)
- An [OpenRouter](https://openrouter.ai) API key — for the CLI and benchmark
  harness only; the MCP server itself runs without one

## Installation

> **v4 is not yet on PyPI.** PyPI still serves the old v3 server, so
> `uv pip install "mcp-solver[agent]"` installs the wrong software. Until v4
> is published, the only working install is a clone plus an editable install:

```bash
git clone https://github.com/szeider/mcp-solver.git
cd mcp-solver
uv pip install -e ".[agent]"

mkdir -p ~/.config/coder
echo 'OPENROUTER_API_KEY="sk-or-v1-..."' > ~/.config/coder/.env
```

See [INSTALL.md](INSTALL.md) for details and troubleshooting.

## Command-line use

The `mcp-solver` CLI runs the same solving pipeline without an MCP host,
which is handy for scripts and benchmarks:

```bash
mcp-solver pysat "Place 8 non-attacking queens on an 8x8 board and give one solution."
mcp-solver pysat --problem tests/problems/pysat/n_queens.md
```

The solution JSON goes to stdout; the submitted solver program (e.g.
`n_queens_code.py`) and a complete run log (`run_*.json`) land in the working
directory.

| Option           | Description                                            |
| ---------------- | ------------------------------------------------------ |
| `--problem FILE` | Read the task from a markdown file instead of text     |
| `--model NAME`   | Agent model: an alias like `gpt56terra` or a full OpenRouter ID (default: `gpt56terra`) |
| `--workdir DIR`  | Working directory for the run (default: current dir)   |
| `--step-limit N` | Maximum agent steps before stopping (default: 30)      |
| `-q`, `--quiet`  | Suppress per-step progress output                      |
| `--dev [PATH]`   | Dev mode: take helpers and templates from a local checkout (bare `--dev` auto-detects; auto-on inside a checkout; also `MCP_SOLVER_DEV`) |
| `--no-dev`       | Disable dev mode; use the published (PyPI-pinned) helpers |
| `--stats-json FILE` | Write the run statistics to `FILE` as JSON          |

From an editable clone, the CLI runs in **dev mode**: helpers and templates
come from the source checkout (auto-detected), with one provenance line on
stderr.

### mcp-minion

The CLI runs **mcp-minion** underneath: a minimal ReAct agent over MCP,
developed in this repo as the workspace package [`minion/`](minion/). It is
a lightweight, batchable substitute for a full MCP host such as Claude
Desktop — point its standalone CLI at `mcp-solver-serve` (or any MCP server)
to script what a chat host would do interactively (see
[minion/README.md](minion/README.md)). The default agent model is
`gpt56terra`, the bundled alias for OpenRouter's `openai/gpt-5.6-terra`.

## Backends

Each backend is selected as the first CLI argument (or via the
`select_backend` tool). The right solver library is injected into the
solve-time kernel automatically.

| Backend  | Domain                          | Solver library      |
| -------- | ------------------------------- | ------------------- |
| `pysat`  | Boolean satisfiability (SAT)    | python-sat          |
| `maxsat` | Weighted optimization           | python-sat (RC2)    |
| `z3`     | SMT solving / verification      | z3-solver           |
| `cpmpy`  | Constraint programming (CP)     | cpmpy               |
| `clingo` | Answer Set Programming (ASP)    | clingo              |

MiniZinc mode from v3 is retired; constraint programming is now covered by
CPMpy.

## How it works

The base `mcp-solver` package is a dependency-free **solver helper library**
(`mcp_solver.helpers.{pysat,maxsat,z3}`; the CPMpy and Clingo backends need
no helper module). It is never pip-installed into your
environment; instead it is injected into each solve-time kernel via
`uv run --with mcp-solver==<version>`, alongside the backend's solver library.
This keeps the host environment clean and each solve reproducible.

Each backend ships a **project template**: a markdown prompt (package data)
with the encoding conventions, output format, and helper API for that
solver. The template goes to whoever solves: an MCP host receives it from
`select_backend`; the CLI hands it to its mcp-minion agent as the system
prompt. Either way, the solving LLM drives the write → execute → verify
loop in a persistent IPython kernel — the `ipython_mcp` server from
[agentic-python-coder](https://github.com/szeider/agentic-python-coder) —
and must end the solve by calling `submit_code` with the final
self-contained program (syntax-checked server-side, never written to disk
by the kernel). On the MCP path the accepted submission is linked back to
the host as a resource; on the CLI path the client extracts the last
successful submission, writes it to `<basename>_code.py`, and re-executes
it in a fresh `uv` kernel to produce the answer.

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

Each run is bounded by `--step-limit` (default: 30 agent steps), and results
are appended to `results.jsonl` in the output directory.

Current status: in our runs with `gpt-5.6-terra`, all 26 test problems solve
correctly. The MCP server path is validated with Claude Opus as host: 229/229
instances across [CP-Bench](https://doi.org/10.5281/zenodo.18034815) (101)
and [ASP-Bench](https://doi.org/10.5281/zenodo.18062939) (128) solve
correctly under strict semantic validation, as do the bundled SAT, MaxSAT,
and SMT problems.

## From v3 to v4: a new architecture

The SAT 2025 paper, ["Bridging Language Models and Symbolic Solvers via the
Model Context Protocol"](https://doi.org/10.4230/LIPIcs.SAT.2025.30),
documents **v3**: an MCP server exposing model-editing tools (`add_item`,
`replace_item`, `solve_model`, …) through which a chat LLM builds up a solver
model item by item, with solving happening inside the server.

**v4 follows the IPython-kernel approach** laid out in the
[CP-Agent](https://doi.org/10.1145/3786181.3788711) (LLM4Code '26,
[arXiv:2508.07468](https://arxiv.org/abs/2508.07468)) and
[ASP-Bench](https://doi.org/10.1145/3786168.3788402) (NSE '26,
[arXiv:2602.01171](https://arxiv.org/abs/2602.01171)) papers: instead of a
host LLM editing a model through protocol tools, a dedicated coding agent
works in a persistent IPython kernel, where it iteratively writes an actual
Python solver program, executes it against the real solver, verifies the
result independently, and only then submits the program as the answer.

|                    | v3 (SAT 2025 paper)                          | v4 (CP-Agent / ASP-Bench approach)                 |
| ------------------ | -------------------------------------------- | -------------------------------------------------- |
| Who does the work  | The host chat LLM, via MCP editing tools     | The host LLM (or **mcp-minion** for batch runs), as a solver programmer |
| Unit of work       | Model items (`add_item`, `solve_model`, …)   | A complete, runnable Python solver program         |
| Execution          | Inside the MCP server                        | In a persistent IPython kernel (`ipython_mcp`)     |
| Verification       | Manual, by the host LLM                      | Built into the loop: run → check → `submit_code`   |
| Artifact           | Transient model state                        | A verified, re-executable program you keep         |
| Host interface     | Many fine-grained model-editing tools        | A solving kernel: `select_backend`, `python_exec`, `submit_code` |
| Backends           | MiniZinc, PySAT, MaxSAT, Z3, ASP             | PySAT, MaxSAT, Z3, CPMpy, Clingo                   |

## Citations

- Stefan Szeider, "Bridging Language Models and Symbolic Solvers via the Model
  Context Protocol", SAT 2025. Documents the v3 architecture (heritage branch).
  [DOI 10.4230/LIPIcs.SAT.2025.30](https://doi.org/10.4230/LIPIcs.SAT.2025.30)
- Stefan Szeider, "CP-Agent: Agentic Constraint Programming", LLM4Code '26
  (ACM). The IPython-kernel approach behind v4.
  [DOI 10.1145/3786181.3788711](https://doi.org/10.1145/3786181.3788711),
  preprint [arXiv:2508.07468](https://arxiv.org/abs/2508.07468)
- Stefan Szeider, "ASP-Bench: From Natural Language to Logic Programs",
  NSE '26 (IEEE/ACM). The verification-gated benchmark methodology used by
  v4's test problems.
  [DOI 10.1145/3786168.3788402](https://doi.org/10.1145/3786168.3788402),
  preprint [arXiv:2602.01171](https://arxiv.org/abs/2602.01171)

## License

MIT; see [LICENSE](LICENSE).
