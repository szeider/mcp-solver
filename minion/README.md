# mcp-minion

A minimal ReAct agent that drives one or more [MCP](https://modelcontextprotocol.io)
servers through an OpenAI-compatible API (OpenRouter by default). It reads a
task from a *run folder*, loops "think → call tool → observe" until the model
produces a final answer, and writes a complete JSON log of the run.

`mcp-minion` is developed in the [mcp-solver](https://github.com/szeider/mcp-solver)
monorepo, where it is used to run solver templates over code-execution MCP
servers, but it has no dependency on the solver and works with any MCP server.

## Install

From the monorepo (workspace member):

```bash
uv sync --package mcp-minion
```

Or standalone:

```bash
pip install -e minion
```

Set an OpenRouter API key, either in `~/.mcp-minion` or in a run folder's `.env`:

```
OPENROUTER_API_KEY=sk-...
```

## Run-folder config

A run folder holds everything for a single run:

| File | Required | Purpose |
| --- | --- | --- |
| `config.json` | yes | Model, agent, MCP server, and kernel settings |
| `task.md` | yes | The specific task for this run |
| `project.md` | no | Shared instructions prepended to the task |
| `files.system` target | no | System-prompt body (see below) |
| `run_*.json` | auto | Per-run JSON log (written automatically) |

`config.json` (nested format):

```json
{
  "model": {
    "name": "google/gemini-2.5-flash",
    "temperature": 0,
    "max_tokens": 2048
  },
  "agent": { "max_steps": 10 },
  "files": { "system": "system.md" },
  "packages": ["z3-solver"],
  "mcpServers": {
    "ipython": { "command": "ipython_mcp", "args": [] }
  }
}
```

Key options:

- **`files.system`** — path (relative to the run folder) to a markdown file
  whose contents become the system prompt. If the file contains the literal
  placeholder `{tool_sections}`, the generated per-server tool list is spliced
  in at that spot (via a literal replace, so other `{`/`}` in the file — e.g.
  ASP templates — are left untouched); otherwise the tool list is appended
  after the file contents. When `files.system` is absent, a built-in default
  system prompt is used.
- **`packages`** — packages to preinstall in a code-execution kernel. On the
  first call to a `python_exec` tool, if no `python_reset` has happened yet,
  `mcp-minion` automatically calls `python_reset` with these packages and
  reuses any returned `kernel_id` for subsequent `python_exec` calls.

## Examples

### Bundled test server

The package ships a tiny MCP server exposing `echo` and `add`:

```json
{
  "model": { "name": "google/gemini-2.5-flash", "temperature": 0 },
  "agent": { "max_steps": 5 },
  "mcpServers": {
    "test": {
      "command": "python",
      "args": ["-m", "mcp_minion.test_mcp_server"]
    }
  }
}
```

```bash
mcp-minion path/to/run_folder -v
```

### IPython code execution

With an [`ipython_mcp`](https://pypi.org/project/ipython-mcp/) server available:

```json
{
  "model": { "name": "google/gemini-2.5-flash", "temperature": 0 },
  "agent": { "max_steps": 5 },
  "mcpServers": {
    "ipython": { "command": "ipython_mcp", "args": [] }
  }
}
```

`task.md`:

```
Use the python_exec tool to compute sum(range(1, 11)) and report the result.
```

## Artifacts

`mcp_minion.artifacts.extract_last_submission(log_or_steps, tool_name="submit_code")`
returns the `code` argument of the last *successful* call to a submission tool,
so callers can persist an agent's final code alongside the run log.

## Tests

```bash
uv run --with-editable ./minion --with pytest --with pytest-asyncio \
    python -m pytest minion/tests -q
```

The end-to-end tests under `minion/e2e_tests/` make real API calls and are
skipped unless `OPENROUTER_API_KEY` is set.
