"""Benchmark harness: run test problems through the solving CLI and validate.

Each problem is solved in its own subprocess (the engine does not support
concurrent solves in one process), its JSON solution is piped into the
problem's ``*_ground_truth.py`` validator, and one result row per run is
appended to ``results.jsonl`` in the output directory. Each subprocess runs
with its own run directory as working directory, so the kernels and
re-executed programs of parallel runs do not share one. A run counts as valid
only if the validator accepts the solution and the persisted program imports
its backend's solver library.
"""

import argparse
import ast
import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from mcp_solver.agent.cli import (
    NEW_SESSION,
    kill_process_group,
    signal_process_group,
)
from mcp_solver.templates import SOLVERS

# File in each run directory holding the CLI subprocess's stderr.
STDERR_FILE = "cli_stderr.txt"
# Lines of that stderr copied into the result row of a failed run.
STDERR_TAIL_LINES = 20
# Seconds to wait for the remaining output after a timed-out run was killed.
KILL_GRACE = 5
# Seconds a CLI gets after SIGTERM (to kill its re-executed program) before
# its process group gets SIGKILL.
TERM_GRACE = 3

# Top-level module of each backend's solver library. A submitted program must
# import it; this rejects a constant-printing program without such an import
# (it does not establish that the solver was used).
SOLVER_MODULES: dict[str, str] = {
    "pysat": "pysat",
    "maxsat": "pysat",
    "z3": "z3",
    "cpmpy": "cpmpy",
    "clingo": "clingo",
    "didp": "didppy",
}

# CLI processes running now. In their own sessions they no longer receive the
# terminal's Ctrl-C, so main() stops them on KeyboardInterrupt or SIGTERM;
# after that (_stopping) no new CLI is started. Both under _live_lock.
_live: set[subprocess.Popen] = set()
_stopping = False
_live_lock = threading.Lock()


class BenchmarkStopped(Exception):
    """A solve was about to start after the benchmark was interrupted."""


def _text(captured: str | bytes | None) -> str:
    """Captured output as text (TimeoutExpired may carry bytes or None)."""
    if captured is None:
        return ""
    if isinstance(captured, bytes):
        return captured.decode("utf-8", errors="replace")
    return captured


def run_cli(cmd: list[str], timeout: float, cwd: Path) -> subprocess.CompletedProcess:
    """``subprocess.run(cmd, capture_output=True, text=True, ...)`` whose
    timeout kills the CLI with all its descendants (uv, python, the engine).

    On POSIX the CLI starts in its own session and the whole process group
    is killed; ``subprocess.TimeoutExpired`` is raised as by
    ``subprocess.run``, carrying the stderr captured until the kill.
    """
    # Spawn and register atomically with respect to _stop_all().
    with _live_lock:
        if _stopping:
            raise BenchmarkStopped
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            start_new_session=NEW_SESSION,
        )
        _live.add(proc)
    with proc:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            kill_process_group(proc, grace=TERM_GRACE)
            # A process outside the group (e.g. a kernel in its own session)
            # may still hold the pipes open; do not wait for it.
            try:
                exc.output, exc.stderr = proc.communicate(timeout=KILL_GRACE)
            except subprocess.TimeoutExpired as late:
                exc.output, exc.stderr = late.output, late.stderr
            raise exc from None
        except BaseException:
            kill_process_group(proc, grace=TERM_GRACE)
            raise
        finally:
            with _live_lock:
                _live.discard(proc)
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def _stop_all() -> None:
    """Stop every running CLI (SIGTERM, then SIGKILL after TERM_GRACE
    seconds, to the whole group) and start no new one."""
    global _stopping
    with _live_lock:
        _stopping = True
        running = list(_live)
    for proc in running:
        signal_process_group(proc, signal.SIGTERM)
    deadline = time.monotonic() + TERM_GRACE
    for proc in running:
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=max(0.0, deadline - time.monotonic()))
    for proc in running:
        kill_process_group(proc)


def _interrupt_on_sigterm(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def imports_solver_library(code: str, solver: str) -> bool:
    """True if *code* imports *solver*'s library (``import X``, ``import X.y``
    or ``from X[.y] import ...`` anywhere in the program).

    Decided on the syntax tree, so a mention in a comment or a string does
    not count; code that does not parse counts as no import.
    """
    module = SOLVER_MODULES[solver]
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        else:
            continue
        if any(name == module or name.startswith(module + ".") for name in names):
            return True
    return False


def discover(problems_root: Path, solvers: list[str]) -> list[tuple[str, Path]]:
    """Return (solver, problem_md) pairs for every non-smoke problem."""
    found = []
    for solver in solvers:
        directory = problems_root / solver
        if not directory.is_dir():
            continue
        for md in sorted(directory.glob("*.md")):
            if md.name != "test.md":
                found.append((solver, md))
    return found


def build_command(
    solver: str, problem_md: Path, args: argparse.Namespace, workdir: Path
) -> list[str]:
    """The CLI command for one run, with every path argument absolute.

    The command runs with *workdir* as its working directory, so relative
    inputs are resolved against the harness's own working directory first:
    problem files, --dev PATH or a path in $MCP_SOLVER_DEV (passed on as
    --dev, which the CLI ranks above the environment; ``1``/``true``/``auto``
    stay with the child, whose auto-detection does not depend on the cwd),
    a model .json file, and an alias that a local ``<alias>.json`` overrides
    (the engine's lookup order).
    """
    workdir = workdir.resolve()
    model = args.model
    if model.endswith(".json"):
        model = str(Path(model).expanduser().resolve())
    elif "/" not in model and Path(f"{model}.json").is_file():
        model = str(Path(f"{model}.json").resolve())
    dev = args.dev
    if dev is None:
        env_dev = os.environ.get("MCP_SOLVER_DEV")
        if env_dev and env_dev.lower() not in ("1", "true", "auto"):
            dev = env_dev
    cmd = [
        sys.executable,
        "-m",
        "mcp_solver.agent.cli",
        solver,
        "--problem",
        str(problem_md.resolve()),
        "--workdir",
        str(workdir),
        "--model",
        model,
        "--stats-json",
        str(workdir / "stats.json"),
        "-q",
    ]
    if dev is not None:
        cmd.append("--dev")
        if dev != "auto":
            cmd.append(str(Path(dev).expanduser().resolve()))
    if args.step_limit:
        cmd += ["--step-limit", str(args.step_limit)]
    if args.token_limit is not None:
        cmd += ["--token-limit", str(args.token_limit)]
    return cmd


def run_one(
    solver: str,
    problem_md: Path,
    args: argparse.Namespace,
    out_root: Path,
    run_idx: int,
) -> dict:
    """Solve one problem in a CLI subprocess and validate its output.

    The subprocess's stderr goes to ``<run dir>/cli_stderr.txt`` (also on a
    timeout, with whatever was captured); for a nonzero exit or a result
    that is not valid, its last lines are also in the row as ``stderr_tail``.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workdir = (out_root / solver / f"{problem_md.stem}_{stamp}_r{run_idx}").resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    stats_file = workdir / "stats.json"
    cmd = build_command(solver, problem_md, args, workdir)

    row = {
        "solver": solver,
        "problem": problem_md.stem,
        "model": args.model,
        "timestamp": stamp,
        "workdir": str(workdir),
        "dev": args.dev,
    }
    start = time.monotonic()
    try:
        proc = run_cli(cmd, timeout=args.timeout, cwd=workdir)
        row["exit_code"] = proc.returncode
        row["solution"] = proc.stdout.strip()
        stderr = _text(proc.stderr)
    except subprocess.TimeoutExpired as exc:
        row["exit_code"] = -1
        row["solution"] = ""
        row["error"] = f"timeout after {args.timeout}s"
        stderr = _text(exc.stderr)
    row["wall_seconds"] = round(time.monotonic() - start, 1)
    try:
        (workdir / STDERR_FILE).write_text(stderr, encoding="utf-8")
    except OSError as exc:
        row["stderr_file_error"] = str(exc)

    if stats_file.is_file():
        try:
            stats = json.loads(stats_file.read_text())
        except (json.JSONDecodeError, OSError):
            # A run killed mid-write can leave a truncated stats file; one bad
            # run must not abort the whole benchmark.
            stats = {}
        row["total_tokens"] = stats.get("token_consumption", {}).get("total_tokens")
        row["exec_calls"] = stats.get("tool_usage", {}).get("python_exec", 0)
        if stats.get("step_limit_reached"):
            row["step_limit_reached"] = True
        if stats.get("token_limit_reached"):
            row["token_limit_reached"] = True

    _validate(row, problem_md)
    _check_solver_usage(row, solver, workdir / f"{problem_md.stem}_code.py")
    if row["exit_code"] != 0 or row.get("valid") is False:
        row["stderr_tail"] = "\n".join(stderr.splitlines()[-STDERR_TAIL_LINES:])
    return row


def _validate(row: dict, problem_md: Path) -> None:
    """Set ``valid`` and ``message`` in *row* from the problem's validator."""
    validator = problem_md.with_name(f"{problem_md.stem}_ground_truth.py")
    if not validator.is_file():
        row["valid"] = None
        row["message"] = "no validator"
    elif not row["solution"]:
        row["valid"] = False
        row["message"] = "no solution output"
    else:
        try:
            check = subprocess.run(
                [sys.executable, str(validator)],
                input=row["solution"],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            row["valid"] = False
            row["message"] = "validator timed out after 120s"
            return
        out = check.stdout.strip()
        verdict = None
        for candidate in (out, out.splitlines()[-1] if out else ""):
            try:
                verdict = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue
        if isinstance(verdict, dict):
            row["valid"] = bool(verdict.get("valid"))
            row["message"] = verdict.get("message", "")
        else:
            row["valid"] = False
            row["message"] = f"validator produced no verdict: {check.stderr[:200]}"


def _check_solver_usage(row: dict, solver: str, program: Path) -> None:
    """Set ``solver_imported`` in *row*; a valid row whose persisted program
    does not import the backend's solver library becomes not valid."""
    try:
        code = program.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        code = ""
    row["solver_imported"] = imports_solver_library(code, solver)
    if row.get("valid") and not row["solver_imported"]:
        row["valid"] = False
        row["message"] = (
            f"the solution validates, but the submitted program {program.name}"
            f" does not import {SOLVER_MODULES[solver]} (the {solver} backend's"
            f" solver library); validator said: {row.get('message', '')}"
        )


def print_summary(rows: list[dict]) -> None:
    """Print a per-solver summary table to stderr."""
    by_solver: dict[str, list[dict]] = {}
    for row in rows:
        by_solver.setdefault(row["solver"], []).append(row)
    print("\nsolver   valid/run   tokens(avg)   exec(avg)   wall(avg)", file=sys.stderr)
    for solver, group in sorted(by_solver.items()):
        n = len(group)
        valid = sum(1 for r in group if r.get("valid"))
        tokens = [r["total_tokens"] for r in group if r.get("total_tokens") is not None]
        execs = [r["exec_calls"] for r in group if "exec_calls" in r]
        walls = [r["wall_seconds"] for r in group]
        avg = lambda xs: sum(xs) / len(xs) if xs else 0  # noqa: E731
        print(
            f"{solver:<8} {valid}/{n:<9} {avg(tokens):>11,.0f} {avg(execs):>11.1f}"
            f" {avg(walls):>10.1f}s",
            file=sys.stderr,
        )
    failures = [r for r in rows if r.get("valid") is False]
    for r in failures:
        print(
            f"  FAIL {r['solver']}/{r['problem']}: {r.get('message', '')[:120]}",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mcp-solver-bench",
        description="Run test problems through the solving CLI and validate results.",
    )
    parser.add_argument(
        "solvers",
        nargs="*",
        default=list(SOLVERS),
        help=f"solvers to benchmark (default: all of {', '.join(SOLVERS)})",
    )
    parser.add_argument(
        "--problems",
        metavar="DIR",
        default="tests/problems",
        help="root directory of per-solver problem folders (default: %(default)s)",
    )
    parser.add_argument("--model", default="gpt56terra", help="agent model name")
    parser.add_argument("--runs", type=int, default=1, help="iterations per problem")
    parser.add_argument("--jobs", type=int, default=4, help="parallel solves")
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="safety backstop in seconds per solve (the operative bound is --step-limit)",
    )
    parser.add_argument(
        "--step-limit",
        type=int,
        default=30,
        help="maximum agent steps per solve (default: %(default)s)",
    )
    parser.add_argument(
        "--token-limit",
        type=int,
        default=None,
        metavar="N",
        help="maximum total tokens per solve, passed to the CLI (0 = no limit;"
        " default: the CLI's default)",
    )
    parser.add_argument(
        "--dev",
        nargs="?",
        const="auto",
        default=None,
        metavar="PATH",
        help="dev mode: pass through to the CLI so templates come from a"
        " local checkout (bare --dev auto-detects; default: CLI decides)",
    )
    parser.add_argument(
        "--out",
        metavar="DIR",
        default=None,
        help="output directory (default: bench_results/<timestamp>)",
    )
    args = parser.parse_args(argv)

    solvers = args.solvers
    unknown = set(solvers) - set(SOLVERS)
    if unknown:
        parser.error(f"unknown solver(s): {', '.join(sorted(unknown))}")

    problems = discover(Path(args.problems), solvers)
    if not problems:
        parser.error(f"no problems found under {args.problems} for {solvers}")

    out_root = Path(
        args.out or f"bench_results/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    results_file = out_root / "results.jsonl"

    jobs = [(s, p, i) for s, p in problems for i in range(args.runs)]
    print(
        f"mcp-solver-bench: {len(jobs)} runs ({len(problems)} problems x {args.runs}),"
        f" model {args.model}, {args.jobs} parallel -> {out_root}",
        file=sys.stderr,
    )

    rows = []
    # SIGTERM takes the KeyboardInterrupt path, which stops the running CLIs.
    on_main = threading.current_thread() is threading.main_thread()
    if on_main:
        previous = signal.signal(signal.SIGTERM, _interrupt_on_sigterm)
    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            try:
                futures = {
                    pool.submit(run_one, s, p, args, out_root, i): (s, p)
                    for s, p, i in jobs
                }
                for future in as_completed(futures):
                    row = future.result()
                    rows.append(row)
                    with results_file.open("a") as fh:
                        fh.write(json.dumps(row) + "\n")
                    status = {True: "ok", False: "FAIL", None: "?"}[row.get("valid")]
                    print(
                        f"[{len(rows)}/{len(jobs)}] {status:<4}"
                        f" {row['solver']}/{row['problem']} ({row['wall_seconds']}s)",
                        file=sys.stderr,
                    )
            except KeyboardInterrupt:
                pool.shutdown(wait=False, cancel_futures=True)
                _stop_all()
                raise
    finally:
        if on_main:
            # None: the previous handler was not installed from Python.
            signal.signal(signal.SIGTERM, previous or signal.SIG_DFL)

    print_summary(rows)
    print(str(results_file))
    # valid=None means "no validator", which is not a failure.
    return 0 if all(r.get("valid") is not False for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
