# MCP Solver Installation

MCP Solver v4 drives the [agentic-python-coder](https://github.com/szeider/agentic-python-coder)
engine to write and run real solver programs. This guide covers prerequisites,
the install variants, verification, and common problems.

> Looking for the v3 MCP server (MiniZinc/PySAT/MaxSAT/Z3/ASP)? It lives on the
> **`heritage`** branch, which keeps its own setup instructions.

---

## Prerequisites

### Python 3.13

MCP Solver v4 requires Python 3.13.

```bash
# macOS (Homebrew)
brew install python@3.13

# Windows
winget install --id Python.Python.3.13

# Linux: install via your distribution or pyenv
python3.13 --version
```

### uv

[uv](https://docs.astral.sh/uv/) is a hard runtime requirement, not just a
convenience: solver libraries and the MCP Solver helper library are supplied at
solve time via `uv run --with`, never pip-installed into your environment.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify:

```bash
uv --version
```

### OpenRouter API key

The coding agent calls models through [OpenRouter](https://openrouter.ai). Store
your key where the engine looks for it:

```bash
mkdir -p ~/.config/coder
echo 'OPENROUTER_API_KEY="sk-or-v1-..."' > ~/.config/coder/.env
```

`OPENROUTER_API_KEY` in the environment works too. This is the same key and
location the engine uses; MCP Solver does not require a separate key.

---

## Install

### Product layer (normal use)

Installs the CLI, the backend templates, and the coding-agent engine:

```bash
uv pip install "mcp-solver[agent]"
```

The `[agent]` extra is what pulls in `agentic-python-coder`. The bare
`mcp-solver` package is only the dependency-free helper library and cannot solve
anything on its own.

### Development / testing

To run the benchmark harness and the helper unit tests, install the `[test]`
extra (which adds the solver libraries and pytest), or `[all]` for everything:

```bash
git clone https://github.com/szeider/mcp-solver.git
cd mcp-solver
uv pip install -e ".[agent,test]"   # or ".[all]"
```

The individual extras are:

| Extra     | Contents                                                        |
| --------- | -------------------------------------------------------------- |
| `[agent]` | Product layer: CLI, templates, agentic-python-coder engine      |
| `[test]`  | Solver libraries (python-sat, z3-solver, cpmpy, clingo) + pytest |
| `[dev]`   | ruff                                                            |
| `[all]`   | `[agent]` + `[test]` + `[dev]`                                  |

---

## Verify

Check the CLI is available:

```bash
mcp-solver --help
```

Run your first solve (canonical example shipped in the repo):

```bash
mcp-solver pysat --problem tests/problems/pysat/n_queens.md
```

You should see engine progress on stderr, a compact stats line, and the
solution JSON on stdout. The saved solver program (`n_queens_code.py`) appears
in the working directory.

For a full end-to-end check across all backends, run the benchmark harness:

```bash
mcp-solver-bench pysat --runs 1
```

---

## Troubleshooting

**`mcp-solver: the coding-agent engine is not installed.`**
You installed the bare package without the product layer. Install the `[agent]`
extra: `uv pip install "mcp-solver[agent]"`.

**`uv: command not found` (or the solve fails to start a kernel).**
uv is missing from your PATH. Install it (see Prerequisites) and make sure its
install directory (usually `~/.local/bin`) is on your `PATH`. uv is mandatory —
solver libraries are injected through `uv run --with`.

**Authentication / missing-key errors from the engine.**
The OpenRouter key is not being found. Confirm `~/.config/coder/.env` contains
`OPENROUTER_API_KEY="..."`, or export `OPENROUTER_API_KEY` in your shell. The
same key drives both MCP Solver and the underlying engine.

**Wrong Python version.**
v4 needs Python 3.13. Check with `python3.13 --version`; uv can also pin a
version with `uv python install 3.13`.

For engine-specific options (model list, library API, project templates), see
the [agentic-python-coder README](https://github.com/szeider/agentic-python-coder).
