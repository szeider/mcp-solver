"""Materialize mcp-minion run folders for benchmark problems.

A run folder (config.json + task.md) points mcp-minion at mcp-solver-serve
with instructions to solve one benchmark problem using one specific backend:

    mcp-solver-runfolder didp tsptw ~/demos/tsptw
    uv run --project <checkout> mcp-minion -v ~/demos/tsptw

The problem text comes from ``tests/problems/<solver>/<problem>.md`` in the
source checkout (problems are not shipped as package data, so this tool
requires running from a checkout). The same helpers back the live e2e tests
in ``minion/e2e_tests/test_problems_e2e.py``, so a generated folder and an
e2e test run are the same episode.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mcp_solver.agent.cli import _local_checkout
from mcp_solver.templates import SOLVERS

DEFAULT_MODEL = "openai/gpt-5.6-terra"

# Prepended to every problem statement: the agent must use the requested
# backend and finish with a submission. The backend-substitution clause
# exists because agents otherwise silently fall back to another backend
# when a select_backend call fails.
PREAMBLE = """\
Solve this problem using the mcp-solver tools: call select_backend with the
{solver} backend and follow its modeling instructions. Use the
{solver} backend ONLY - do not substitute another backend. Finish with
submit_code, then give the solution JSON as your final answer.

---

"""

# Substrings (case-insensitive, any match) whose presence in a submitted
# program indicates the backend's solver library was genuinely used. The
# e2e tests use this to detect silent backend substitution.
SOLVER_MARKERS: dict[str, list[str]] = {
    "pysat": ["pysat"],
    "maxsat": ["wcnf", "rc2", "maxsat"],
    "z3": ["z3"],
    "cpmpy": ["cpmpy"],
    "clingo": ["clingo"],
    "didp": ["didppy"],
}


def build_config(repo_root: Path, model: str = DEFAULT_MODEL) -> dict:
    """The minion config.json contents for solving via mcp-solver-serve."""
    return {
        "model": {"name": model, "temperature": 0, "max_tokens": 8192},
        "agent": {"max_steps": 30},
        "mcpServers": {
            "mcp-solver": {
                "command": "uv",
                "args": ["run", "--project", str(repo_root), "mcp-solver-serve"],
            }
        },
    }


def write_run_folder(
    dest: Path,
    solver: str,
    problem_text: str,
    repo_root: Path,
    model: str = DEFAULT_MODEL,
) -> Path:
    """Write config.json + task.md into *dest* (created if needed).

    Only these two files are written; run logs and submissions already in
    the folder are left untouched.
    """
    dest.mkdir(parents=True, exist_ok=True)
    config = build_config(repo_root, model)
    (dest / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    (dest / "task.md").write_text(
        PREAMBLE.format(solver=solver) + problem_text, encoding="utf-8"
    )
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mcp-solver-runfolder",
        description="Write an mcp-minion run folder for a benchmark problem.",
    )
    parser.add_argument("solver", choices=SOLVERS, help="backend to solve with")
    parser.add_argument(
        "problem",
        help="problem name (a .md stem under tests/problems/<solver>/),"
        " or 'list' to show what is available",
    )
    parser.add_argument(
        "dest",
        nargs="?",
        default=None,
        help="run folder to create (required unless problem is 'list')",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="agent model, full OpenRouter ID (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    root = _local_checkout()
    if root is None:
        parser.error(
            "requires a source checkout of mcp-solver"
            " (benchmark problems are not shipped as package data)"
        )
    problems_dir = Path(root) / "tests" / "problems" / args.solver
    available = sorted(p.stem for p in problems_dir.glob("*.md") if p.name != "test.md")

    if args.problem == "list":
        for name in available:
            print(name)
        return 0

    if args.dest is None:
        parser.error("dest is required (or use 'list' to show problems)")
    problem_md = problems_dir / f"{args.problem}.md"
    if not problem_md.is_file():
        parser.error(
            f"no problem {args.problem!r} for {args.solver};"
            f" available: {', '.join(available) or '(none)'}"
        )

    folder = write_run_folder(
        Path(args.dest).expanduser(),
        args.solver,
        problem_md.read_text(encoding="utf-8"),
        Path(root),
        model=args.model,
    )
    print(f"run folder ready: {folder}")
    print(f"run it: uv run --project {root} mcp-minion -v {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
