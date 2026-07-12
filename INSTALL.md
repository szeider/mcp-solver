# MCP Solver Installation

MCP Solver v4 runs an [mcp-minion](minion/) agent against the IPython kernel
server of [agentic-python-coder](https://github.com/szeider/agentic-python-coder)
to write and run real solver programs. This guide covers prerequisites, the
install variants, verification, and common problems.

> **v4 is not yet on PyPI.** PyPI still serves the old v3 server, so
> `uv pip install "mcp-solver[agent]"` installs the wrong package. Until v4 is
> published, the only working install — for both normal use and development — is
> a clone plus an editable install (see [Install](#install) below).

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

The agent calls models through [OpenRouter](https://openrouter.ai). Store your
key in one of the places the CLI looks:

```bash
mkdir -p ~/.config/coder
echo 'OPENROUTER_API_KEY="sk-or-v1-..."' > ~/.config/coder/.env
```

`OPENROUTER_API_KEY` in the environment works too, as does `~/.mcp-minion`
(the standalone mcp-minion CLI's location). One key serves everything.

---

## Install

### Clone + editable (required today)

Until v4 reaches PyPI, normal use and development share the same path: clone the
repo and install it editable. The `[agent]` extra pulls in `mcp-minion` (the
agent loop, a workspace package under `minion/`) and `agentic-python-coder`
(the IPython kernel MCP server and model aliases); the bare `mcp-solver`
package is only the dependency-free helper library and cannot solve anything
on its own.

```bash
git clone https://github.com/szeider/mcp-solver.git
cd mcp-solver
uv pip install -e ".[agent]"          # normal use
uv pip install -e ".[agent,test]"     # + benchmark harness and unit tests
# uv pip install -e ".[all]"          # everything, including dev tools
```

An editable install runs in **dev mode**: the CLI auto-detects the source
checkout and takes both the helper library and the solver templates straight
from it (printing one provenance line to stderr), so nothing extra is needed to
run a solve. Use `--dev PATH` (or `MCP_SOLVER_DEV`) to point at a checkout
explicitly, or `--no-dev` to force the published (PyPI-pinned) helpers.

The individual extras are:

| Extra     | Contents                                                        |
| --------- | -------------------------------------------------------------- |
| `[agent]` | Product layer: CLI, templates, mcp-minion agent, kernel server  |
| `[test]`  | Solver libraries (python-sat, z3-solver, cpmpy, clingo) + pytest |
| `[dev]`   | ruff                                                            |
| `[all]`   | `[agent]` + `[test]` + `[dev]`                                  |

### Once v4 is published

When v4 is on PyPI, `uv pip install "mcp-solver[agent]"` will install the product
layer directly, with no clone required.

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

You should see per-step progress on stderr, a compact stats line, and the
solution JSON on stdout. The submitted solver program (`n_queens_code.py`) and
a run log (`run_*.json`) appear in the working directory.

For a full end-to-end check across all backends, run the benchmark harness:

```bash
mcp-solver-bench pysat --runs 1
```

---

## Troubleshooting

**`mcp-solver: the agent layer is not installed`**
You installed the bare package without the product layer. Install the `[agent]`
extra. Until v4 is on PyPI, that means the clone + editable install:
`uv pip install -e ".[agent]"` from a checkout (see [Install](#install)).

**`the ipython_mcp server does not provide submit_code`**
Your installed `agentic-python-coder` is older than 3.4.0. Upgrade it (from a
checkout: `uv pip install -e /path/to/agentic-python-coder`, or from PyPI once
3.4.0 is published).

**`uv: command not found` (or the solve fails to start a kernel).**
uv is missing from your PATH. Install it (see Prerequisites) and make sure its
install directory (usually `~/.local/bin`) is on your `PATH`. uv is mandatory —
solver libraries are injected through `uv run --with`.

**`mcp-solver: no OpenRouter API key found.`**
The key is not being located. Either export `OPENROUTER_API_KEY` in your shell,
or put it in `~/.config/coder/.env` as `OPENROUTER_API_KEY="..."`. The `.env`
file is read even when the environment variable is unset, so the file alone is
enough. The same key drives both MCP Solver and the underlying engine.

**`mcp-solver: problem file not found: <path>`**
The path passed to `--problem` does not exist. Check the spelling and that you
are running from the repo root (the shipped problems live under
`tests/problems/<solver>/`).

**Wrong Python version.**
v4 needs Python 3.13. Check with `python3.13 --version`; uv can also pin a
version with `uv python install 3.13`.

For kernel-server details and the bundled model aliases, see the
[agentic-python-coder README](https://github.com/szeider/agentic-python-coder);
for the agent loop itself, see [minion/README.md](minion/README.md).
