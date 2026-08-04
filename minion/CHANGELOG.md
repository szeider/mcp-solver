# Changelog

Changes to the `mcp-minion` package. The monorepo's root `CHANGELOG.md`
tracks `mcp-solver` releases, which version independently of this package.

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
</content>
</invoke>
