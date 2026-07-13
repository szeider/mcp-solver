"""mcp-solver command-line interface (v4).

Usage:
    mcp-solver <solver> [task ...] [options]

Runs an mcp-minion agent that writes a solver program for a constraint
problem, working against the agentic-python-coder ``ipython_mcp`` kernel
server over stdio. The agent must submit its final program via the server's
``submit_code`` tool; the CLI persists that submission to
``<basename>_code.py``, re-executes it in a fresh ``uv`` kernel, and prints
its solution (JSON) to stdout.
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import mcp_solver
from mcp_solver.templates import SOLVERS, get_template

# Re-execute the submitted program under the same interpreter version that
# ran the agent's kernel, so in-kernel verification carries over.
PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"

# Solver libraries injected into the solve-time kernel, per backend.
# pypblib backs pysat.pb.PBEnc (pseudo-Boolean encodings — budgets,
# capacities); without it PBEnc raises at import time.
SOLVER_PACKAGES: dict[str, list[str]] = {
    "pysat": ["python-sat", "pypblib"],
    "maxsat": ["python-sat", "pypblib"],
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


def _model_params(config: dict[str, Any]) -> dict[str, Any]:
    """Translate a model-alias config dict into OpenRouter API parameters.

    Mirrors the engine's rules: models flagged ``no_sampling_params`` accept
    only ``max_tokens``; otherwise the standard sampling knobs pass through.
    (``top_k``/``provider``/``reasoning`` need ``extra_body``, which the
    minion agent reserves, so they are not forwarded.)
    """
    params: dict[str, Any] = {}
    if config.get("no_sampling_params"):
        if "max_tokens" in config:
            params["max_tokens"] = config["max_tokens"]
        return params
    for key in (
        "temperature",
        "max_tokens",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
    ):
        if key in config:
            params[key] = config[key]
    return params


def resolve_model(name: str) -> tuple[str, dict[str, Any]]:
    """Resolve a model name to ``(openrouter_id, api_params)``.

    A name containing ``/`` is already an OpenRouter model ID and passes
    through unchanged. Anything else is an alias looked up in the engine's
    bundled ``models/<name>.json`` (e.g. ``gpt56terra``).

    Raises ValueError for an unknown alias.
    """
    if "/" in name:
        return name, {}
    from importlib.resources import files

    try:
        text = (files("agentic_python_coder") / "models" / f"{name}.json").read_text(
            encoding="utf-8"
        )
    except (ModuleNotFoundError, FileNotFoundError) as exc:
        raise ValueError(
            f"unknown model alias {name!r}; pass a full OpenRouter ID"
            " (e.g. openai/gpt-5.6-terra) or install the engine for aliases"
        ) from exc
    config = json.loads(text)
    return config["path"], _model_params(config)


def find_api_key() -> str | None:
    """Return the OpenRouter API key, or None.

    Checked in order: the environment, then the dotenv files used by the
    engine (``~/.config/coder/.env``) and by mcp-minion (``~/.mcp-minion``).
    """
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    from dotenv import dotenv_values

    for dotfile in (
        Path.home() / ".config" / "coder" / ".env",
        Path.home() / ".mcp-minion",
    ):
        if dotfile.is_file():
            key = dotenv_values(dotfile).get("OPENROUTER_API_KEY")
            if key:
                return key
    return None


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
        default=30,
        metavar="N",
        help="maximum agent steps before stopping (default: %(default)s)",
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


def build_stats(result: Any, elapsed: float, model_id: str) -> dict:
    """Summarise an AgentResult as the stats dict consumed by the harness.

    Keeps the key shape of the old engine stats (``token_consumption``,
    ``tool_usage``, ``execution_time_seconds``, ``step_limit_reached``) so
    downstream consumers need no changes.
    """
    tool_usage: dict[str, int] = {}
    for step in result.steps:
        for call in step.get("tool_calls") or []:
            name = call.get("name", "?")
            tool_usage[name] = tool_usage.get(name, 0) + 1
    return {
        "token_consumption": {
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "total_tokens": result.input_tokens + result.output_tokens,
            "cost_usd": round(getattr(result, "cost", 0.0), 6),
        },
        "tool_usage": tool_usage,
        "execution_time_seconds": elapsed,
        "steps": len(result.steps),
        "tool_calls_made": result.tool_calls_made,
        "step_limit_reached": result.max_steps_reached,
        "model": model_id,
    }


def _progress(step_info: dict) -> None:
    """Print a one-line progress note for a completed agent step."""
    calls = step_info.get("tool_calls") or []
    detail = ", ".join(c.get("name", "?") for c in calls) if calls else "final answer"
    print(f"mcp-solver: step {step_info['step']}: {detail}", file=sys.stderr)


async def solve(
    task: str,
    api_key: str,
    config: Any,
    logger: Any,
    quiet: bool,
) -> Any:
    """Run one solve: minion agent over the engine's ipython_mcp server."""
    from mcp_minion import Agent, MCPManager

    servers = {
        "ipython": {
            "command": sys.executable,
            "args": ["-m", "agentic_python_coder.mcp_server"],
        }
    }
    manager = MCPManager(servers)
    await manager.__aenter__()
    try:
        if not manager.has_tool("submit_code"):
            raise RuntimeError(
                "the ipython_mcp server does not provide submit_code;"
                " upgrade agentic-python-coder to >=3.4.0"
            )
        agent = Agent(
            api_key=api_key,
            tools=None,
            config=config,
            logger=logger,
            mcp_manager=manager,
            on_step=None if quiet else _progress,
        )
        return await agent.run_async(task)
    finally:
        await manager.__aexit__(None, None, None)


def print_stats(stats: dict) -> None:
    """Print a compact one-line summary of the solve run to stderr."""
    tokens = stats.get("token_consumption", {}).get("total_tokens", 0)
    cost = stats.get("token_consumption", {}).get("cost_usd", 0)
    exec_calls = stats.get("tool_usage", {}).get("python_exec", 0)
    elapsed = stats.get("execution_time_seconds", 0)
    cost_part = f", ${cost:.4f}" if cost else ""
    print(
        f"mcp-solver: {tokens:,} tokens{cost_part}, {exec_calls} exec calls,"
        f" {elapsed:.1f}s",
        file=sys.stderr,
    )


def run_program(program: Path, with_packages: list[str]) -> int:
    """Execute the saved program in a uv kernel; return its exit code.

    The program's stdout (the solution JSON) is inherited so it becomes
    this command's stdout.
    """
    cmd = ["uv", "run", "--no-project", "--python", PYTHON_VERSION]
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
        import agentic_python_coder  # noqa: F401  (kernel server + model aliases)
        import mcp_minion  # noqa: F401
    except ImportError:
        print(
            "mcp-solver: the agent layer is not installed"
            " (needs agentic-python-coder and mcp-minion).\n"
            "Install it with: uv pip install 'mcp-solver[agent]'\n"
            "(until v4 is on PyPI: clone the repo and run"
            " uv pip install -e '.[agent]')",
            file=sys.stderr,
        )
        return 1

    try:
        model_id, api_params = resolve_model(args.model)
    except ValueError as exc:
        parser.error(str(exc))

    api_key = find_api_key()
    if api_key is None:
        print(
            "mcp-solver: no OpenRouter API key found.\n"
            "Set OPENROUTER_API_KEY or put it in ~/.config/coder/.env"
            " (see INSTALL.md).",
            file=sys.stderr,
        )
        return 1

    workdir = Path(args.workdir or os.getcwd()).resolve()
    project_prompt = get_template(args.solver, root=dev_path)

    from mcp_minion import AgentConfig, RunLogger, extract_last_submission

    config = AgentConfig(
        model=model_id,
        max_steps=args.step_limit,
        api_params=api_params,
        system_prompt=project_prompt,
        packages=with_packages,
    )
    logger = RunLogger(log_dir=workdir)

    start = time.monotonic()
    try:
        result = asyncio.run(solve(task, api_key, config, logger, args.quiet))
    except RuntimeError as exc:
        print(f"mcp-solver: {exc}", file=sys.stderr)
        return 1
    elapsed = time.monotonic() - start

    stats = build_stats(result, elapsed, model_id)
    print_stats(stats)
    if args.stats_json:
        try:
            Path(args.stats_json).write_text(json.dumps(stats, default=str))
        except OSError as exc:
            print(
                f"mcp-solver: could not write stats to {args.stats_json}: {exc}",
                file=sys.stderr,
            )

    code = extract_last_submission(result.steps)
    if code is None:
        print(
            "mcp-solver: the run submitted no final program"
            " (submit_code was never called successfully).",
            file=sys.stderr,
        )
        return 3
    program = workdir / f"{task_basename}_code.py"
    program.write_text(code, encoding="utf-8")

    print(f"mcp-solver: running {program}", file=sys.stderr)
    return run_program(program, with_packages)


if __name__ == "__main__":
    sys.exit(main())
