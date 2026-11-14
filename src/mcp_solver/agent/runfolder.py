"""Materialize mcp-minion run folders for benchmark problems.

A run folder (config.json + task.md) points mcp-minion at mcp-solver-serve
with instructions to solve one benchmark problem using one specific backend:

    mcp-solver-runfolder didp tsptw ~/demos/tsptw
    uv run --project <checkout> --extra agent mcp-minion -v ~/demos/tsptw

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

from mcp_solver.agent.cli import resolve_model
from mcp_solver.agent.engine import local_checkout
from mcp_solver.templates import SOLVERS, get_template

DEFAULT_MODEL = "openai/gpt-5.6-terra"

# Floor for the agent's tool-result cap in generated run folders.
MIN_TOOL_RESULT_CHARS = 64000

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


def model_block(model: str) -> dict:
    """The config.json ``model`` object for *model*, resolved like the CLI.

    *model* is an engine alias, a model .json file or a full OpenRouter ID
    (see ``cli.resolve_model``; a full ID carries no settings). Sampling
    parameters go in as is; ``reasoning``/``provider``/``top_k`` and
    ``request_timeout`` use the keys of minion's run-folder format.
    Raises ValueError for an unknown alias, and for a config with
    ``base_url`` or ``api_key_env``, which run folders cannot express
    (minion would pass them to the API call as request parameters).
    """
    settings = resolve_model(model)
    if settings.base_url or settings.api_key_env:
        raise ValueError(
            f"model {model!r} sets base_url/api_key_env, which mcp-minion"
            " run folders do not support; use an OpenRouter model"
        )
    block: dict = {"name": settings.model, **settings.api_params}
    block.update(settings.extra_body)
    if settings.request_timeout is not None:
        block["request_timeout"] = settings.request_timeout
    return block


def tool_result_cap(solver: str) -> int:
    """The ``agent.max_tool_result_chars`` for a run folder solving with *solver*.

    On the server path select_backend returns the whole backend template as
    one tool result (clingo is the longest), so the cap is twice
    the template length rounded up to the next thousand, and at least
    MIN_TOOL_RESULT_CHARS, independent of minion's default.
    """
    doubled = 2 * len(get_template(solver))
    return max(MIN_TOOL_RESULT_CHARS, -(-doubled // 1000) * 1000)


def build_config(
    repo_root: Path, model: str = DEFAULT_MODEL, solver: str | None = None
) -> dict:
    """The minion config.json contents for solving via mcp-solver-serve.

    With *solver*, the tool-result cap is sized for that backend's template
    (see tool_result_cap); without it, MIN_TOOL_RESULT_CHARS.
    """
    cap = tool_result_cap(solver) if solver else MIN_TOOL_RESULT_CHARS
    return {
        "model": model_block(model),
        "agent": {"max_steps": 30, "max_tool_result_chars": cap},
        "mcpServers": {
            "mcp-solver": {
                "command": "uv",
                "args": [
                    "run",
                    "--project",
                    str(repo_root),
                    "--extra",
                    "agent",
                    "mcp-solver-serve",
                ],
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
    config = build_config(repo_root, model, solver)
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
        help="agent model: an engine alias, a model .json file, or a full"
        " OpenRouter ID (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    root = local_checkout()
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

    try:
        folder = write_run_folder(
            Path(args.dest).expanduser(),
            args.solver,
            problem_md.read_text(encoding="utf-8"),
            Path(root),
            model=args.model,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(f"run folder ready: {folder}")
    print(f"run it: uv run --project {root} --extra agent mcp-minion -v {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
