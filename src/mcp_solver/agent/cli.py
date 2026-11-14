"""mcp-solver command-line interface (v4).

Usage:
    mcp-solver <solver> [task ...] [options]

Runs an mcp-minion agent that writes a solver program for a constraint
problem, working against the agentic-python-coder ``ipython_mcp`` kernel
server over stdio. The agent must submit its final program via the server's
``submit_code`` tool; the CLI persists that submission to
``<basename>_code.py``, re-executes it in a fresh ``uv`` kernel, and prints
its solution (JSON) to stdout.

Exit status:
    0    the submitted program ran and exited 0
    1    setup or agent failure (agent layer missing, no API key, engine
         error, kernel setup failure, an exception from the solve, or an
         agent error before any submission)
    2    usage error
    3    the agent finished without a successful ``submit_code`` (also when
         the step limit or the token limit ended the run first)
    4    the agent stopped early with an error (e.g. an LLM API failure)
         after an accepted submission; that submission was saved and ran
         with exit 0, but it may not be the agent's final, verified program
    124  the re-executed program did not finish within ``RERUN_TIMEOUT``
         seconds and was killed with its descendants
    127  ``uv`` not found when re-executing the program
    143  SIGTERM arrived during re-execution; the program was killed
    any other code is the submitted program's own nonzero exit code
"""

import argparse
import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mcp_solver
from mcp_solver.agent import engine
from mcp_solver.templates import SOLVERS, get_template

# Re-execute the submitted program under the same interpreter version that
# ran the agent's kernel, so in-kernel verification carries over.
PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"

# Exit code for a run the agent ended early with an error after an accepted
# submission (see the module docstring for all exit codes).
EXIT_AGENT_ERROR_WITH_SUBMISSION = 4

# Wall-clock limit (seconds) for re-executing the submitted program: the
# kernel's maximum python_exec timeout (300 s), under which the agent ran and
# verified the program, plus 60 s for uv to set up the cached environment.
RERUN_TIMEOUT = 360
# Exit code when the re-execution hits RERUN_TIMEOUT (as timeout(1)).
EXIT_RERUN_TIMEOUT = 124

# On POSIX, children that run solver code start in their own session (and
# process group), so a timeout kills them with all their descendants (uv,
# python, kernels); elsewhere only the direct child is killed.
NEW_SESSION = os.name == "posix"

# Default cap on the total (input + output) tokens of one run.
DEFAULT_TOKEN_LIMIT = 2_000_000

# API key variable unless a model config names another (api_key_env).
DEFAULT_KEY_VAR = "OPENROUTER_API_KEY"

# Solver libraries injected into the solve-time kernel and the re-execution,
# per backend; nothing else is installed there.
# The [pb] extra installs pindakaas, which backs pysat.pb.PBEnc
# (pseudo-Boolean encodings: budgets, capacities); without it every PBEnc
# encoding call raises "Package 'pindakaas' is unavailable".
SOLVER_PACKAGES: dict[str, list[str]] = {
    "pysat": ["python-sat[pb]"],
    "maxsat": ["python-sat[pb]"],
    "z3": ["z3-solver"],
    "cpmpy": ["cpmpy"],
    "clingo": ["clingo"],
    # didppy is bounded: the didp template documents the 0.10.x API, which
    # was verified empirically; lift the cap only after re-testing.
    "didp": ["didppy>=0.11.1,<0.12"],
}


def resolve_dev_path(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> str | None:
    """Return the effective dev-mode path (a source checkout root) or None.

    Applies ``engine.resolve_dev_path`` to ``--dev``/``--no-dev`` (see there
    for the resolution order); a dev-mode request without a checkout is a
    parser error.
    """
    try:
        return engine.resolve_dev_path(args.dev, args.no_dev)
    except engine.DevModeError as exc:
        parser.error(str(exc))


def build_with_packages(solver: str) -> list[str]:
    """Return the ``--with`` package set for a solver run.

    Only the backend's solver libraries; the kernel does not install this
    distribution.
    """
    return list(SOLVER_PACKAGES[solver])


# Sampling parameters forwarded from a model config (unless the model is
# flagged no_sampling_params, which forwards only max_tokens).
_SAMPLING_KEYS = (
    "temperature",
    "max_tokens",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
)


@dataclass
class ModelSettings:
    """A resolved model: the OpenRouter ID and the request settings.

    ``api_params`` go into the completions call as is, ``extra_body`` into
    the request body (OpenRouter ``reasoning``/``provider``/``top_k``),
    ``request_timeout`` (seconds) is the per-request timeout. ``base_url``
    (None: OpenRouter) is an OpenAI-compatible endpoint and ``api_key_env``
    (None: OPENROUTER_API_KEY) names the variable holding its key, as in
    the engine's model configs.
    """

    model: str
    api_params: dict[str, Any] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    request_timeout: float | None = None
    base_url: str | None = None
    api_key_env: str | None = None

    @property
    def key_var(self) -> str:
        """The environment variable name to look up the API key under."""
        return self.api_key_env or DEFAULT_KEY_VAR


def model_settings(config: dict[str, Any]) -> ModelSettings:
    """Map an engine model config (``load_model_config``) to ModelSettings.

    Follows the engine's own translation: ``no_sampling_params`` models get
    only ``max_tokens`` (and no ``top_k``); ``provider`` and ``reasoning``
    apply to every model; ``base_url`` and ``api_key_env`` select another
    OpenAI-compatible endpoint and its key variable.
    """
    no_sampling = bool(config.get("no_sampling_params"))
    keys = ("max_tokens",) if no_sampling else _SAMPLING_KEYS
    api_params = {k: config[k] for k in keys if k in config}
    extra_body: dict[str, Any] = {}
    if not no_sampling and "top_k" in config:
        extra_body["top_k"] = config["top_k"]
    for key in ("provider", "reasoning"):
        if key in config:
            extra_body[key] = config[key]
    return ModelSettings(
        model=config["path"],
        api_params=api_params,
        extra_body=extra_body,
        request_timeout=config.get("request_timeout"),
        base_url=config.get("base_url") or None,
        api_key_env=config.get("api_key_env") or None,
    )


def resolve_model(name: str) -> ModelSettings:
    """Resolve a model name to ModelSettings.

    A name ending in ``.json`` is a model config file; any other name
    containing ``/`` is an OpenRouter model ID and passes through without
    settings; anything else is an alias resolved by the engine's
    ``load_model_config`` (``./<name>.json``, then the bundled
    ``models/<name>.json``, e.g. ``gpt56terra``).

    Raises ValueError for an unknown alias or an invalid config file.
    """
    if "/" in name and not name.endswith(".json"):
        return ModelSettings(model=name)
    try:
        from agentic_python_coder import load_model_config
    except ImportError as exc:
        raise ValueError(
            f"unknown model alias {name!r}; pass a full OpenRouter ID"
            " (e.g. openai/gpt-5.6-terra) or install the engine for aliases"
        ) from exc
    try:
        config = load_model_config(name)
    except FileNotFoundError as exc:
        raise ValueError(
            f"unknown model alias {name!r}; pass a full OpenRouter ID"
            f" (e.g. openai/gpt-5.6-terra) or a model .json file ({exc})"
        ) from exc
    return model_settings(config)


def find_api_key(env_var: str = DEFAULT_KEY_VAR) -> str | None:
    """Return the API key held in *env_var*, or None.

    ``mcp_minion.find_api_key`` with the engine's dotenv file: checked in
    order are the environment, ``~/.config/coder/.env`` (engine) and
    ``~/.mcp-minion``. The files are read, not loaded into the environment.
    """
    from mcp_minion import find_api_key as minion_find_api_key

    return minion_find_api_key(
        env_var=env_var,
        extra_dotenv=(Path.home() / ".config" / "coder" / ".env",),
    )


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
        help="agent model: an engine alias, a model .json file, or a full"
        " OpenRouter ID (default: %(default)s)",
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
        "--token-limit",
        type=int,
        default=DEFAULT_TOKEN_LIMIT,
        metavar="N",
        help="maximum total (input + output) tokens of the run, checked after"
        " each step; 0 means no limit (default: %(default)s)",
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
        help="dev mode: take templates from a local checkout"
        " (bare --dev auto-detects a source checkout; auto-on inside one)",
    )
    dev_group.add_argument(
        "--no-dev",
        action="store_true",
        help="disable dev mode; use the packaged templates",
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
        except (OSError, UnicodeDecodeError) as exc:
            parser.error(f"cannot read problem file {args.problem}: {exc}")
    return task_text, "task"


def build_stats(
    result: Any, elapsed: float, model_id: str, token_limit: int = 0
) -> dict:
    """Summarise an AgentResult as the stats dict consumed by the harness.

    Keeps the key shape of the old engine stats (``token_consumption``,
    ``tool_usage``, ``execution_time_seconds``, ``step_limit_reached``) so
    downstream consumers need no changes; ``token_limit_reached`` and
    ``token_limit`` (0 = none) report the per-run token cap.
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
        "token_limit_reached": bool(getattr(result, "token_cap_reached", False)),
        "token_limit": token_limit,
        "model": model_id,
    }


def _progress(step_info: dict) -> None:
    """Print a one-line progress note for a completed agent step."""
    calls = step_info.get("tool_calls") or []
    detail = (
        ", ".join(
            c.get("name", "?") + (" (error)" if c.get("is_error") else "")
            for c in calls
        )
        if calls
        else "final answer"
    )
    print(f"mcp-solver: step {step_info['step']}: {detail}", file=sys.stderr)


async def solve(
    task: str,
    api_key: str,
    config: Any,
    logger: Any,
    quiet: bool,
) -> Any:
    """Run one solve: minion agent over the engine's ipython_mcp server.

    Raises engine.EngineError (a RuntimeError) when the engine lacks a
    needed tool.
    """
    from mcp_minion import Agent

    manager = await engine.start_engine()
    try:
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
    if stats.get("token_limit_reached"):
        print(
            f"mcp-solver: the run stopped at the token limit"
            f" ({stats.get('token_limit', 0):,} tokens).",
            file=sys.stderr,
        )
    elif stats.get("step_limit_reached"):
        print("mcp-solver: the run stopped at the step limit.", file=sys.stderr)


def run_program(program: Path, with_packages: list[str]) -> int:
    """Execute the saved program in a uv kernel; return its exit code.

    After ``RERUN_TIMEOUT`` seconds the program and its descendants are
    killed and ``EXIT_RERUN_TIMEOUT`` is returned.

    The program runs under ``engine.kernel_env()``, the environment the
    agent's kernel had (API keys withheld). Its stdout (the solution JSON)
    is inherited so it becomes this command's stdout.
    """
    cmd = ["uv", "run", "--no-project", "--python", PYTHON_VERSION]
    for pkg in with_packages:
        cmd += ["--with", pkg]
    cmd += ["python", str(program)]
    # The program runs in its own session, so a SIGTERM to this process (or
    # to its process group, as the benchmark harness sends) does not reach
    # it: turn SIGTERM into SystemExit, whose cleanup below kills the program.
    handle_term = NEW_SESSION and threading.current_thread() is threading.main_thread()
    if handle_term:
        previous = signal.signal(signal.SIGTERM, _exit_on_sigterm)
    proc = None
    try:
        proc = subprocess.Popen(
            cmd, env=engine.kernel_env(), start_new_session=NEW_SESSION
        )
        return proc.wait(timeout=RERUN_TIMEOUT)
    except FileNotFoundError:
        print(
            "mcp-solver: 'uv' not found on PATH; it is required to run the"
            " submitted program (https://docs.astral.sh/uv/).",
            file=sys.stderr,
        )
        return 127
    except subprocess.TimeoutExpired:
        kill_process_group(proc)
        print(
            f"mcp-solver: the submitted program did not finish within"
            f" {RERUN_TIMEOUT} s and was killed; {program.name} is kept.",
            file=sys.stderr,
        )
        return EXIT_RERUN_TIMEOUT
    except BaseException:
        # In its own session the program no longer receives the terminal's
        # Ctrl-C, so take it down on any interruption of this process.
        if proc is not None:
            kill_process_group(proc)
        raise
    finally:
        if handle_term:
            # None: the previous handler was not installed from Python.
            signal.signal(signal.SIGTERM, previous or signal.SIG_DFL)


def _exit_on_sigterm(signum: int, frame: object) -> None:
    raise SystemExit(128 + signum)


def signal_process_group(proc: subprocess.Popen, sig: int) -> None:
    """Send *sig* to *proc*'s process group (POSIX); elsewhere kill *proc*.

    *proc* must have been started with ``start_new_session=NEW_SESSION``.
    """
    # OSError: the group or process is already gone (ESRCH; EPERM on macOS
    # when only zombies are left).
    with contextlib.suppress(OSError):
        if NEW_SESSION:
            os.killpg(proc.pid, sig)
        else:
            proc.kill()


def kill_process_group(proc: subprocess.Popen, grace: float = 0) -> None:
    """Kill *proc* and, on POSIX, every process in its group; reap *proc*.

    With *grace* > 0 the group first gets SIGTERM, so that *proc* can clean
    up (the CLI kills its re-executed program, which runs in a session of
    its own), and *proc* gets up to *grace* seconds to exit; then the group
    gets SIGKILL. *proc* must have been started with
    ``start_new_session=NEW_SESSION``.
    """
    if NEW_SESSION and grace > 0:
        signal_process_group(proc, signal.SIGTERM)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=grace)
    signal_process_group(proc, signal.SIGKILL if NEW_SESSION else 0)
    proc.wait()


def _describe_exception(exc: BaseException) -> str:
    """Return a one-line ``Type: message`` description of *exc*.

    Single-member exception groups (anyio task groups in the MCP client wrap
    connection errors this way) are unwrapped to the underlying error.
    """
    while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
        exc = exc.exceptions[0]
    message = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    task, task_basename = resolve_task(args, parser)
    if args.token_limit < 0:
        parser.error("--token-limit must be >= 0 (0 means no limit)")
    dev_path = resolve_dev_path(args, parser)

    with_packages = build_with_packages(args.solver)
    if dev_path is not None:
        print(
            f"mcp-solver: dev mode — templates from {dev_path}",
            file=sys.stderr,
        )

    try:
        import agentic_python_coder  # noqa: F401  (kernel server + model aliases)
        import mcp_minion  # noqa: F401
    except ImportError:
        print(
            "mcp-solver: the agent layer is not installed"
            " (needs agentic-python-coder and mcp-minion).\n"
            f"Install it with: uv pip install '{mcp_solver.DIST_NAME}[agent]'"
            " (from a checkout: uv pip install -e '.[agent]')",
            file=sys.stderr,
        )
        return 1

    try:
        model = resolve_model(args.model)
    except ValueError as exc:
        parser.error(str(exc))

    api_key = find_api_key(model.key_var)
    if api_key is None:
        if model.api_key_env:
            message = (
                f"mcp-solver: no API key found in {model.api_key_env}"
                " (named by api_key_env in the model config).\n"
                f"Set {model.api_key_env} or put it in ~/.config/coder/.env."
            )
        else:
            message = (
                "mcp-solver: no OpenRouter API key found.\n"
                "Set OPENROUTER_API_KEY or put it in ~/.config/coder/.env"
                " (see INSTALL.md)."
            )
        print(message, file=sys.stderr)
        return 1

    workdir = Path(args.workdir or os.getcwd()).resolve()
    project_prompt = get_template(args.solver, root=dev_path)

    from mcp_minion import AgentConfig, RunLogger, extract_last_submission

    endpoint = {"base_url": model.base_url} if model.base_url else {}
    config = AgentConfig(
        **endpoint,
        model=model.model,
        max_steps=args.step_limit,
        api_params=model.api_params,
        extra_body=model.extra_body or None,
        request_timeout=model.request_timeout,
        max_total_tokens=args.token_limit,
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
    except Exception as exc:
        print(
            f"mcp-solver: the solve failed: {_describe_exception(exc)}",
            file=sys.stderr,
        )
        return 1
    elapsed = time.monotonic() - start

    stats = build_stats(result, elapsed, model.model, args.token_limit)
    print_stats(stats)
    if args.stats_json:
        try:
            Path(args.stats_json).write_text(json.dumps(stats, default=str))
        except OSError as exc:
            print(
                f"mcp-solver: could not write stats to {args.stats_json}: {exc}",
                file=sys.stderr,
            )

    # Set by mcp-minion when the agent stopped early: an LLM API failure, or
    # a failed kernel setup (python_reset with the solver packages) before
    # the first step. A step or token limit is not an error.
    agent_error = result.error
    code = extract_last_submission(result.steps)
    if code is None:
        if agent_error:
            print(
                f"mcp-solver: the agent stopped with an error before any"
                f" accepted submission: {agent_error}",
                file=sys.stderr,
            )
            return 1
        print(
            "mcp-solver: the run submitted no final program"
            " (submit_code was never called successfully).",
            file=sys.stderr,
        )
        return 3
    program = workdir / f"{task_basename}_code.py"
    program.write_text(code, encoding="utf-8")

    if agent_error:
        print(
            f"mcp-solver: the agent stopped early with an error: {agent_error}\n"
            f"mcp-solver: {program.name} holds the last accepted submission,"
            " which may not be the agent's final program.",
            file=sys.stderr,
        )
    print(f"mcp-solver: running {program}", file=sys.stderr)
    rc = run_program(program, with_packages)
    if agent_error and rc == 0:
        return EXIT_AGENT_ERROR_WITH_SUBMISSION
    return rc


if __name__ == "__main__":
    sys.exit(main())
