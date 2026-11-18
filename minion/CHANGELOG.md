# Changelog

Changes to the `mcp-minion` package. The monorepo's root `CHANGELOG.md`
tracks `mcp-solver` releases, which version independently of this package.

### [0.3.1] - unreleased

- **Behavior Change:** The kernel convenience (tracking the `kernel_id` of
  `python_reset` results and adding it to kernel tool calls without one,
  merging the configured packages into a model reset) applies only when
  `packages` are configured, that is when minion set up the kernel itself
  (the mcp-solver CLI path). Without configured packages, as against
  `mcp-solver-serve`, which routes calls without `kernel_id` to its solving
  kernel itself, minion passes kernel tool calls unchanged. Formerly a
  model's reset of another kernel (or a `python_reset()` before
  `select_backend`, which reaches the engine's default kernel) made minion
  send every later call without `kernel_id` to that kernel.

### [0.3.0] - 2026-11-14

- **Behavior Change:** `MCPManager.call_tool` returns a `ToolOutcome`
  (`text`, `is_error`, `links`) instead of a JSON string `{"result": ...}` or
  `{"error": ...}`. An MCP `isError` result and client-side failures (unknown
  tool, non-object arguments, server not connected, timeout, transport
  error) come back with `is_error` set; nothing is raised for them.
  `Tool.execute` returns a `ToolOutcome` too (a string return value is the
  text, other values are JSON-encoded). The model sees the text, prefixed
  with `ERROR: ` for an error outcome.
- **Behavior Change:** Step records (`AgentResult.steps` and the run log)
  hold each tool call as `{"name", "arguments", "result", "is_error"}`, where
  `result` is the plain tool text, no longer a JSON envelope. Steps carry
  `reasoning_details` or `reasoning` when the model returns them, and a step
  cut off at `max_tokens` carries `truncated: true`. The run log also
  records `system_prompt` and `kernel_setup`. `extract_last_submission`
  reads the new records and run logs of 0.2.x.
- **Behavior Change:** An LLM API failure no longer raises out of
  `run_async`. The run ends with `AgentResult.error` set (one line with the
  exception type), the steps and token totals so far, and the answer
  `[Agent stopped: LLM call failed: ...]`; the CLI then exits with 1. Error
  responses the SDK parses without raising (HTTP 200 with an error body,
  `finish_reason: "error"`, malformed or non-JSON bodies) are retried 3
  times, after 5, 10 and 15 seconds, before the run ends this way.
- **Behavior Change:** The flat `config.json` format (`modelstring`,
  top-level `max_steps`, `temperature`, ...) is no longer read; a flat
  config or any unknown top-level key is an error. New keys in the nested
  format: `model.reasoning`, `model.provider`, `model.top_k` and
  `model.extra_body` (sent in the request body), `model.request_timeout`,
  `agent.max_tool_result_chars`, and `cwd` for an MCP server.
- **Behavior Change:** Requires the MCP Python SDK 2.x (`mcp>=2,<3`); 1.x
  is no longer supported.
- **Behavior Change:** A server's `env` in `mcpServers` is merged over the
  mcp library's default environment whitelist (`PATH`, `HOME`, ...) instead
  of over the full parent environment, so other variables (API keys, for
  instance) reach the server only when listed. A server without `env` gets
  the whitelist, as before.
- **Behavior Change:** Tool results longer than
  `agent.max_tool_result_chars` (default 64,000; `0` disables the cut) reach
  the model as head and tail around a marker line; step records and the run
  log keep the full text.
- **Behavior Change:** With `packages` configured and a server offering
  `python_reset`, the kernel is set up with those packages at the start of
  the run, before the first LLM call (previously before the first
  `python_exec`), and a failed setup ends the run with `AgentResult.error`.
  The tracked `kernel_id` is added to every kernel tool call that lacks one
  (`python_exec`, `python_reset`, `python_status`, `python_interrupt`), so a
  `python_reset` from the model restarts the same kernel; a model reset
  with its own `packages` list keeps the configured packages.
- **Behavior Change:** The `reasoning_details` (or `reasoning`) of each
  assistant message are replayed unmodified in the conversation history, as
  OpenRouter requires for reasoning models.
- **Behavior Change:** The agent calls the API through `AsyncOpenAI`, with
  one client created per run and closed when the run ends, so one `Agent`
  can run several times and LLM calls no longer block the event loop. A
  client set on `agent.client` is used for every run and not closed; it
  must provide an async `chat.completions.create`. The
  `Agent.SYSTEM_PROMPT_BASE` class attribute is gone; use the module-level
  `SYSTEM_PROMPT_BASE`.
- **Behavior Change:** The CLI offers its arithmetic demo tools only when
  no MCP server is configured.
- **New Feature:** Server instructions from the MCP `initialize` result
  appear in the system prompt. Image, audio and blob content renders as a
  one-line note (type, MIME type, size) instead of base64 or an object repr.
- **New Feature:** `mcp_minion.find_api_key()` looks the key up in the
  environment, a run folder's `.env`, extra dotenv files and
  `~/.mcp-minion`, without loading the files into `os.environ`; the CLI uses
  it.
- **Fix:** A rejected `submit_code` call (`{"ok": false, ...}`) is no longer
  taken as the last submission. A reply cut off at `max_tokens` without
  tool calls is no longer accepted as the final answer; the model is asked
  to continue. Missing or empty tool arguments count as `{}`, arguments
  that are not a JSON object give an error result, and the history echoes
  valid JSON arguments back to the API.

### [0.2.2] - 2026-11-13

- **New Feature:** Supports the MCP Python SDK 2.x alongside 1.x
  (`mcp>=1.25.0,<3`): paginated listing through `PaginatedRequestParams`,
  protocol fields read under both the 2.x snake_case and the 1.x camelCase
  names, and `read_resource` called with a plain string URI.
- **Behavior Change:** Requires Python 3.13 or 3.14 (`>=3.13,<3.15`).

### [0.2.1] - 2026-08-04

- **Fix:** Pin `mcp>=1.25.0,<2`. The `mcp` 2.0.0 SDK changed the client API
  (`ClientSession.list_tools` no longer takes `cursor`), so fresh installs of
  0.1.0 and 0.2.0 failed to connect to any MCP server. The workspace lockfile
  masked this by resolving `mcp` 1.28.1.

### [0.2.0] - 2026-08-04

- **New Feature:** Per-run token cap, `agent.max_total_tokens` in
  `config.json` (absent or `0` = unlimited). The cumulative input+output
  tokens are checked after each completed step; once they exceed the cap the
  loop stops, the answer becomes `[Agent stopped: per-run token cap of N
  exceeded (cumulative M)]`, and the run log records `token_cap_reached`
  (on the last step and in the result). `max_steps` alone does not bound
  spending, since one step can be arbitrarily expensive.
- **Behavior Change:** A configured MCP server that fails to start now aborts
  the run with a `RuntimeError` naming the server and the underlying error,
  after closing any servers already started. Previously the failure was a
  warning on stderr and the agent continued with a reduced tool set, which
  produced plausible-looking but unusable runs.
