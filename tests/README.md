# MCP Solver Tests

Two kinds of tests live here: fast **unit tests** (pytest) and the
end-to-end **benchmark harness** (`mcp-solver-bench`).

## Unit tests

The network-free unit tests in `tests/unit/` cover the CLI, the MCP server,
the engine launch, the benchmark harness, and the run folders. Install
`.[all]` and run:

```bash
uv run pytest tests/unit -q
```

## Benchmark harness

`mcp-solver-bench` runs the problems under `tests/problems/<solver>/`
end-to-end: for each one it invokes the `mcp-solver` CLI, then pipes the
solution JSON on **stdin** into the problem's `*_ground_truth.py` validator,
which prints a `{"valid": ..., "message": ...}` verdict. One result row per run
is appended to `results.jsonl` in the output directory.

```bash
# Every backend
mcp-solver-bench

# Specific backends, several runs each, in parallel
mcp-solver-bench pysat z3 --runs 3 --jobs 4
```

Each solve is bounded by `--step-limit` (default: 30 agent steps). From an
editable checkout the harness runs in dev mode (templates come from the
checkout); pass `--dev [PATH]` to force a specific checkout, or `--no-dev` on
the CLI to use the packaged templates.

## Problem layout

Each backend has its own folder under `tests/problems/<solver>/`
(`pysat`, `maxsat`, `z3`, `cpmpy`, `clingo`, `didp`). A problem is a pair of files:

```
tests/problems/pysat/
  n_queens.md                 # the task, in natural language
  n_queens_ground_truth.py    # validator: reads solution JSON on stdin,
                              #   prints {"valid": bool, "message": str}
```

A `test.md` file (if present) is treated as a smoke problem and skipped by the
harness.
