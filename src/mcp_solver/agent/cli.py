"""mcp-solver command-line interface (v4).

Usage:
    mcp-solver <solver> [task ...] [options]

Drives the agentic-python-coder engine to write a solver program for a
constraint problem, then runs that program and prints its solution (JSON)
to stdout. The engine and the produced program run in ephemeral ``uv``
kernels, so solver libraries are supplied at solve time via ``--with``.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import mcp_solver
from mcp_solver.templates import SOLVERS, get_template

# Solver libraries injected into the solve-time kernel, per backend.
SOLVER_PACKAGES: dict[str, list[str]] = {
    "pysat": ["python-sat"],
    "maxsat": ["python-sat"],
    "z3": ["z3-solver"],
    "cpmpy": ["cpmpy"],
    "clingo": ["clingo"],
}


def _local_checkout() -> str | None:
    """Return the repo root when mcp_solver runs from a source checkout.

    An editable install resolves ``mcp_solver`` to ``<root>/src/mcp_solver``;
    a regular install resolves it inside site-packages, where the checkout
    markers are absent.
    """
    root = Path(mcp_solver.__file__).resolve().parents[2]
    if (root / "pyproject.toml").is_file() and (root / "src" / "mcp_solver").is_dir():
        return str(root)
    return None


def _is_unpublishable(version: str) -> bool:
    """Return True when *version* looks like a pre-release/dev build.

    Published releases are plain numeric (e.g. ``4.0.0``). Any alphabetic
    marker (``a``/``b``/``rc``/``dev`` per PEP 440) means the version is not
    on PyPI, so a PyPI pin to it would fail.
    """
    return bool(re.search(r"[a-zA-Z]", version))


def resolve_dev_path(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> str | None:
    """Return the effective dev-mode path (a source checkout root) or None.

    Dev mode means "everything Python-related comes from the local repo":
    the injected helper library and the solver templates. Resolution order:

    1. ``--no-dev`` → None (published behavior), overrides everything.
    2. ``--dev PATH`` → that path.
    3. ``--dev`` (no path) → auto-detected checkout, or a parser error.
    4. ``$MCP_SOLVER_DEV`` set to ``1``/``true``/``auto`` → auto-detect
       (error if none); any other non-empty value → treat as a path.
    5. Nothing set → auto-default to a detected checkout (may be None).
    """
    if args.no_dev:
        return None
    if args.dev is not None:
        if args.dev == "auto":
            checkout = _local_checkout()
            if checkout is None:
                parser.error("no source checkout detected; pass --dev PATH")
            return checkout
        return args.dev
    env = os.environ.get("MCP_SOLVER_DEV")
    if env:
        if env.lower() in ("1", "true", "auto"):
            checkout = _local_checkout()
            if checkout is None:
                parser.error("no source checkout detected; pass --dev PATH")
            return checkout
        return env
    return _local_checkout()


def helpers_package(dev_path: str | None = None) -> str:
    """Return the ``--with`` spec for the mcp-solver helper library.

    In dev mode (*dev_path* given) the helpers come from that local
    checkout; otherwise from a PyPI pin to the installed version.
    """
    if dev_path:
        return dev_path
    return f"mcp-solver=={mcp_solver.__version__}"


def build_with_packages(solver: str, dev_path: str | None = None) -> list[str]:
    """Assemble the ``--with`` package set for a solver run.

    The solver library comes first, then the mcp-solver helpers. Only the
    pysat/maxsat/z3 templates reference the helpers, but they are injected
    for every backend for uniformity.
    """
    return [*SOLVER_PACKAGES[solver], helpers_package(dev_path)]


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="mcp-solver",
        description="Solve a constraint problem with an LLM coding agent.",
    )
    parser.add_argument(
        "solver",
        choices=SOLVERS,
        help="constraint solver backend",
    )
    parser.add_argument(
        "task",
        nargs="*",
        help="task text (words are joined with spaces)",
    )
    parser.add_argument(
        "--problem",
        metavar="FILE",
        help="read the task from a markdown file instead of positional text",
    )
    parser.add_argument(
        "--model",
        default="gpt56terra",
        help="agent model name (default: %(default)s)",
    )
    parser.add_argument(
        "--workdir",
        metavar="DIR",
        default=None,
        help="working directory for the run (default: current directory)",
    )
    parser.add_argument(
        "--step-limit",
        type=int,
        default=None,
        metavar="N",
        help="maximum agent steps before stopping",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="suppress engine progress output",
    )
    dev_group = parser.add_mutually_exclusive_group()
    dev_group.add_argument(
        "--dev",
        nargs="?",
        const="auto",
        default=None,
        metavar="PATH",
        help="dev mode: take helpers and templates from a local checkout"
        " (bare --dev auto-detects a source checkout; auto-on inside one)",
    )
    dev_group.add_argument(
        "--no-dev",
        action="store_true",
        help="disable dev mode; use the published (PyPI-pinned) helpers",
    )
    parser.add_argument(
        "--stats-json",
        metavar="FILE",
        default=None,
        help="write the engine run statistics to FILE as JSON",
    )
    return parser


def resolve_task(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> tuple[str, str]:
    """Return (task_text, task_basename) enforcing task/problem exclusivity.

    Exactly one of positional task text or ``--problem`` must be given.
    """
    task_text = " ".join(args.task).strip()
    if args.problem and task_text:
        parser.error("provide task text OR --problem, not both")
    if not args.problem and not task_text:
        parser.error("provide task text or --problem <file.md>")
    if args.problem:
        problem_path = Path(args.problem)
        try:
            return problem_path.read_text(encoding="utf-8"), problem_path.stem
        except FileNotFoundError:
            parser.error(f"problem file not found: {args.problem}")
    return task_text, "task"


def find_program(workdir: Path, task_basename: str) -> Path | None:
    """Return the saved solver program in *workdir*, or None if absent."""
    for candidate in (workdir / f"{task_basename}_code.py", workdir / "solution.py"):
        if candidate.is_file():
            return candidate
    return None


def print_stats(stats: dict) -> None:
    """Print a compact one-line summary of the solve run to stderr."""
    tokens = stats.get("token_consumption", {}).get("total_tokens", 0)
    exec_calls = stats.get("tool_usage", {}).get("python_exec", 0)
    elapsed = stats.get("execution_time_seconds", 0)
    print(
        f"mcp-solver: {tokens:,} tokens, {exec_calls} exec calls, {elapsed:.1f}s",
        file=sys.stderr,
    )


def run_program(program: Path, with_packages: list[str]) -> int:
    """Execute the saved program in a uv kernel; return its exit code.

    The program's stdout (the solution JSON) is inherited so it becomes
    this command's stdout.
    """
    cmd = ["uv", "run", "--no-project", "--python", "3.13"]
    for pkg in with_packages:
        cmd += ["--with", pkg]
    cmd += ["python", str(program)]
    return subprocess.run(cmd).returncode


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    task, task_basename = resolve_task(args, parser)
    dev_path = resolve_dev_path(args, parser)

    if dev_path is None and _is_unpublishable(mcp_solver.__version__):
        print(
            f"mcp-solver: installed version {mcp_solver.__version__} is not a"
            " published release and dev mode is off.\n"
            "Run from a source checkout, or pass --dev PATH (or --dev in a"
            " checkout) to take helpers and templates from the local repo.",
            file=sys.stderr,
        )
        return 2

    with_packages = build_with_packages(args.solver, dev_path)
    if dev_path is not None:
        print(
            f"mcp-solver: dev mode — helpers and templates from {dev_path}",
            file=sys.stderr,
        )

    try:
        import agentic_python_coder
    except ImportError:
        print(
            "mcp-solver: the coding-agent engine is not installed.\n"
            "Install the product layer with: uv pip install 'mcp-solver[agent]'\n"
            "(until v4 is on PyPI: clone the repo and run"
            " uv pip install -e '.[agent]')",
            file=sys.stderr,
        )
        return 1

    workdir = Path(args.workdir or os.getcwd()).resolve()
    project_prompt = get_template(args.solver, root=dev_path)

    try:
        _messages, stats, _log_path = agentic_python_coder.solve_task(
            task=task,
            working_directory=str(workdir),
            model=args.model,
            project_prompt=project_prompt,
            with_packages=with_packages,
            quiet=args.quiet,
            save_log=True,
            task_basename=task_basename,
            step_limit=args.step_limit,
        )
    except ValueError as exc:
        if "OPENROUTER_API_KEY" not in str(exc):
            raise
        print(
            "mcp-solver: no OpenRouter API key found.\n"
            "Set OPENROUTER_API_KEY or put it in ~/.config/coder/.env"
            " (see INSTALL.md).",
            file=sys.stderr,
        )
        return 1

    print_stats(stats)
    if args.stats_json:
        Path(args.stats_json).write_text(json.dumps(stats, default=str))

    program = find_program(workdir, task_basename)
    if program is None:
        print("mcp-solver: the run produced no saved program.", file=sys.stderr)
        return 3

    print(f"mcp-solver: running {program}", file=sys.stderr)
    return run_program(program, with_packages)


if __name__ == "__main__":
    sys.exit(main())
