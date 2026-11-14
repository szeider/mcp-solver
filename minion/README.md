# mcp-minion

A minimal ReAct agent that drives one or more [MCP](https://modelcontextprotocol.io)
servers through an OpenAI-compatible API (OpenRouter by default). It reads a
task from a *run folder*, loops "think → call tool → observe" until the model
produces a final answer, and writes a complete JSON log of the run.

`mcp-minion` is developed in the [mcp-solver](https://github.com/szeider/mcp-solver)
monorepo, where it runs solver templates over code-execution MCP servers, but
it has no dependency on the solver and works with any MCP server. It requires
Python 3.13 or 3.14 and the MCP Python SDK 2.x.

## Install

```bash
uv tool install mcp-minion          # the mcp-minion command
uv add mcp-minion                   # as a library in a uv project
uv sync --package mcp-minion        # inside the monorepo (workspace member)
```

The OpenRouter API key is looked up in this order: the `OPENROUTER_API_KEY`
environment variable, the run folder's `.env`, then `~/.mcp-minion`
(`OPENROUTER_API_KEY=sk-...`). The files are read, not loaded into the process
environment; `mcp_minion.find_api_key()` does the same lookup for library users.

## Run folder

| File | Purpose |
| --- | --- |
| `config.json` | Model, agent, MCP server and kernel settings (required) |
| `task.md` | The task for this run (required) |
| `project.md` | Shared instructions prepended to the task (optional) |
| `files.system` target | System-prompt body (optional, see below) |
| `run_*.json` | Per-run JSON log, including the system prompt sent (written automatically) |
| `submission.py` | Code of the last accepted `submit_code` call, if any |
| `server.log` | stderr of the MCP servers and their children |

`config.json` uses the nested format; the pre-0.3 flat format
(`modelstring`, top-level `max_steps`, ...) is rejected with an error:

```json
{
  "model": {
    "name": "google/gemini-3.1-flash-lite-preview",
    "temperature": 0,
    "max_tokens": 2048,
    "reasoning": { "effort": "low" },
    "request_timeout": 120
  },
  "agent": { "max_steps": 10, "max_total_tokens": 500000 },
  "files": { "system": "system.md" },
  "packages": ["z3-solver"],
  "mcpServers": {
    "ipython": { "command": "ipython_mcp", "args": [], "env": {}, "cwd": "." }
  }
}
```

- **`model.*`** — `name` is the OpenRouter model id. `reasoning`, `provider`,
  `top_k` and the entries of an optional `extra_body` object go into the
  request body (`AgentConfig.extra_body`; OpenRouter usage accounting stays
  on). `request_timeout` is the per-request timeout in seconds. Every other
  key (`temperature`, `max_tokens`, ...) is passed to the API call as is.
- **`agent.max_total_tokens`** — cap on the cumulative input+output tokens of
  a run, checked after each step; the run then stops with
  `[Agent stopped: per-run token cap of N exceeded (cumulative M)]` and the
  log records `token_cap_reached`. Absent or `0` means no cap.
- **`agent.max_tool_result_chars`** — tool results longer than this are cut
  for the model (see Tool results); default 64000, `0` disables.
- **`files.system`** — markdown file (relative to the run folder) used as the
  system prompt. A literal `{tool_sections}` placeholder is replaced by the
  generated tool list (other braces are left alone); without it the list is
  appended. Absent: a built-in default prompt.
- **`mcpServers`** — servers to start, each with `command` and optional
  `args`, `env` (merged by the mcp library over its default environment
  whitelist) and `cwd`. A server that fails to start aborts the run. Each
  server's `instructions` from its initialize result appear in the system
  prompt under that server's heading. Without any server, the CLI offers four
  arithmetic demo tools (`add`, `subtract`, `multiply`, `divide`).
- **`packages`** — for the kernel server
  [`ipython_mcp`](https://pypi.org/project/agentic-python-coder/) (PyPI
  package `agentic-python-coder`): before the first LLM call, `mcp-minion`
  calls `python_reset` with these packages; if that fails the run ends with an
  error. The returned `kernel_id` is added to every `python_exec`,
  `python_reset`, `python_status` and `python_interrupt` call that lacks one,
  so a `python_reset` by the model restarts the same kernel with its packages.
  If the model's `python_reset` passes its own `packages` list, the configured
  packages are sent first, followed by the model's entries they do not already
  cover (same distribution name, ignoring extras, versions and case).

## Tool results

`MCPManager.call_tool` returns a `ToolOutcome(text, is_error, links)`. The
model sees `text` unchanged (prefixed with `ERROR: ` for errors), cut to
`agent.max_tool_result_chars` in config.json (`AgentConfig.max_tool_result_chars`,
default 64000; head and tail kept, `0` disables); the step records and the run log keep the full text as
`"result"` plus `"is_error"`. Images, audio and blobs appear as one line such
as `[image mime=image/png bytes=1234]`. A `resource_link` appears as a
`[resource_link] URI (name)` line; the built-in `read_resource` tool fetches
it, or any resource a server announced, from its origin server only.
Reasoning that the model returns (`reasoning_details`, or a plain
`reasoning` string) is passed back unmodified on the next request and kept in
the step records and log.

`mcp_minion.extract_last_submission(log_or_steps, tool_name="submit_code")`
returns the `code` argument of the last accepted submission: one without
`is_error` whose text is not a JSON object with `"ok": false`. Run logs of
mcp-minion 0.2 and earlier are read too.

## Example

The package ships a small MCP test server (`echo`, `add`, a resource and
instructions):

```json
{
  "model": { "name": "google/gemini-3.1-flash-lite-preview", "temperature": 0 },
  "agent": { "max_steps": 5 },
  "mcpServers": {
    "test": { "command": "python", "args": ["-m", "mcp_minion.test_mcp_server"] }
  }
}
```

```bash
mcp-minion path/to/run_folder -v
```

## Tests

```bash
cd minion && uv run pytest -q
```

The end-to-end tests under `minion/e2e_tests/` make real API calls and are
skipped unless `OPENROUTER_API_KEY` is set.
