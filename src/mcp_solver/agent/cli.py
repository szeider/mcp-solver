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


def helpers_package(local_package: str | None = None) -> str:
    """Return the ``--with`` spec for the mcp-solver helper library.

    By default this pins to the installed version. A local checkout path
    (from ``--local-package`` or ``$MCP_SOLVER_LOCAL_PACKAGE``) overrides
    it, which is needed until 4.0 is published and while pre-release local
    versions would not resolve from PyPI.
    """
    local = local_package or os.environ.get("MCP_SOLVER_LOCAL_PACKAGE")
    if local:
        return local
    return f"mcp-solver=={mcp_solver.__version__}"


def build_with_packages(solver: str, local_package: str | None = None) -> list[str]:
    """Assemble the ``--with`` package set for a solver run.

    The solver library comes first, then the mcp-solver helpers. Only the
    pysat/maxsat/z3 templates reference the helpers, but they are injected
    for every backend for uniformity.
    """
    return [*SOLVER_PACKAGES[solver], helpers_package(local_package)]


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
    parser.add_argument(
        "--local-package",
        metavar="PATH",
        default=None,
        help="use a local mcp-solver checkout instead of the pinned version",
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
        return problem_path.read_text(encoding="utf-8"), problem_path.stem
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
    with_packages = build_with_packages(args.solver, args.local_package)

    try:
        import agentic_python_coder
    except ImportError:
        print(
            "mcp-solver: the coding-agent engine is not installed.\n"
            "Install the product layer with: uv pip install 'mcp-solver[agent]'",
            file=sys.stderr,
        )
        return 1

    workdir = Path(args.workdir or os.getcwd()).resolve()
    project_prompt = get_template(args.solver)

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
