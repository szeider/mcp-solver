# MCP Solver Installation

MCP Solver v4 lets an LLM write and run real solver programs in a
persistent IPython kernel (the kernel server of
[agentic-python-coder](https://github.com/szeider/agentic-python-coder)).
The solving LLM is either your MCP host (Claude Desktop/Code, Cursor —
via `mcp-solver-serve`) or the bundled [mcp-minion](minion/) agent (via the
`mcp-solver` CLI). This guide covers prerequisites, the install variants,
verification, and common problems.

> **v4 is not yet installable from PyPI.** The PyPI name `mcp-solver` is held
> by a third-party upload of stale v2.0.0 code, and PyPI blocks confusably
> similar names; a [PEP 541](https://peps.python.org/pep-0541/) name-transfer
> request is pending. Until it resolves, the only working install — for both
> normal use and development — is a clone plus an editable install (see
> [Install](#install) below).

> **Platform note.** v4 has so far been tested on macOS only. The Windows and
> Linux instructions below carry over from v3 and have not yet been
> re-verified under v4; an update will follow.

> Looking for the v3 MCP server (MiniZinc/PySAT/MaxSAT/Z3/ASP)? It lives on the
> **`v3`** branch, which keeps its own setup instructions.

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

### OpenRouter API key (CLI path only)

The MCP server needs no API key — the host brings its own model. A key is
required only for the `mcp-solver` CLI and the benchmark harness, whose
bundled mcp-minion agent calls models through
[OpenRouter](https://openrouter.ai). Store the key in one of the places the
CLI looks:

```bash
mkdir -p ~/.config/coder
echo 'OPENROUTER_API_KEY="sk-or-v1-..."' > ~/.config/coder/.env
```

`OPENROUTER_API_KEY` in the environment works too, as does `~/.mcp-minion`
(the standalone mcp-minion CLI's location). One key serves everything.

---

## Install

### Clone + editable (required today)

The `[agent]` extra pulls in `mcp-minion` (the agent loop) and
`agentic-python-coder` (the IPython kernel MCP server and model aliases); the
bare `mcp-solver` package is only the dependency-free helper library and
cannot solve anything on its own.

```bash
git clone https://github.com/szeider/mcp-solver.git
cd mcp-solver
uv pip install -e ".[agent]"          # product layer
uv pip install -e ".[agent,test]"     # + benchmark harness and unit tests
# uv pip install -e ".[all]"          # everything, including dev tools
```

An editable install runs in **dev mode**: the CLI auto-detects the source
checkout and takes both the helper library and the solver templates straight
from it (printing one provenance line to stderr), so nothing extra is needed to
run a solve. Use `--dev PATH` (or `MCP_SOLVER_DEV`) to point at a checkout
explicitly, or `--no-dev` to force the published (PyPI-pinned) helpers.

### From PyPI (once the name transfer completes)

When the PEP 541 request resolves, `uv pip install "mcp-solver[agent]"` will
install the product layer directly, with no clone required, and the MCP
server will run without any install at all:
`uvx --from "mcp-solver[agent]" mcp-solver-serve`.

The individual extras are:

| Extra     | Contents                                                        |
| --------- | -------------------------------------------------------------- |
| `[agent]` | Product layer: CLI, templates, mcp-minion agent, kernel server  |
| `[test]`  | Solver libraries (python-sat, z3-solver, cpmpy, clingo) + pytest |
| `[dev]`   | ruff                                                            |
| `[all]`   | `[agent]` + `[test]` + `[dev]`                                  |

### Using the local version (no PyPI involved)

The checkout is fully self-sufficient — development and testing never wait on
a PyPI release. Everything resolves locally: `mcp-minion` comes from the
`minion/` workspace member, and dev mode injects the helper library and
templates straight from the checkout (auto-detected; also available
explicitly as `--dev PATH` or `MCP_SOLVER_DEV=/path`).

Run the CLI or the benchmark harness from anywhere by pointing `uv` at the
checkout:

```bash
uv run --project /path/to/mcp-solver mcp-solver z3 "task ..."
uv run --project /path/to/mcp-solver mcp-solver-bench pysat
```

Use the local MCP server in an MCP host (Claude Desktop, Claude Code, Cursor)
with this configuration:

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

This is the supported development path alongside the PyPI release: `--dev`
always wins over the PyPI pin.

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
extra: `uv pip install -e ".[agent]"` from a checkout (see
[Install](#install); the PyPI path opens up once the name transfer completes).

**`the ipython_mcp server does not provide submit_code`** (CLI) or
**`the ipython_mcp engine is unavailable or lacks submit_code`** (server)
Your installed `agentic-python-coder` is older than 3.4.1, or missing from
the environment the server runs in. Upgrade it:
`uv pip install "agentic-python-coder>=3.4.1"` (or `-e` from a checkout).

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
