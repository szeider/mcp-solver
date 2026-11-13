# Changelog

Changes to the `mcp-minion` package. The monorepo's root `CHANGELOG.md`
tracks `mcp-solver` releases, which version independently of this package.

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
