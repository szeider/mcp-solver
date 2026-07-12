"""Benchmark harness: run test problems through the solving CLI and validate.

Each problem is solved in its own subprocess (the engine does not support
concurrent solves in one process), its JSON solution is piped into the
problem's ``*_ground_truth.py`` validator, and one result row per run is
appended to ``results.jsonl`` in the output directory.
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from mcp_solver.templates import SOLVERS


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


def run_one(
    solver: str,
    problem_md: Path,
    args: argparse.Namespace,
    out_root: Path,
    run_idx: int,
) -> dict:
    """Solve one problem in a CLI subprocess and validate its output."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workdir = out_root / solver / f"{problem_md.stem}_{stamp}_r{run_idx}"
    workdir.mkdir(parents=True, exist_ok=True)
    stats_file = workdir / "stats.json"

    cmd = [
        sys.executable,
        "-m",
        "mcp_solver.agent.cli",
        solver,
        "--problem",
        str(problem_md),
        "--workdir",
        str(workdir),
        "--model",
        args.model,
        "--stats-json",
        str(stats_file),
        "-q",
    ]
    if args.local_package:
        cmd += ["--local-package", args.local_package]
    if args.step_limit:
        cmd += ["--step-limit", str(args.step_limit)]

    row = {
        "solver": solver,
        "problem": problem_md.stem,
        "model": args.model,
        "timestamp": stamp,
        "workdir": str(workdir),
    }
    start = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
        row["exit_code"] = proc.returncode
        row["solution"] = proc.stdout.strip()
    except subprocess.TimeoutExpired:
        row["exit_code"] = -1
        row["solution"] = ""
        row["error"] = f"timeout after {args.timeout}s"
    row["wall_seconds"] = round(time.monotonic() - start, 1)

    if stats_file.is_file():
        stats = json.loads(stats_file.read_text())
        row["total_tokens"] = stats.get("token_consumption", {}).get("total_tokens")
        row["exec_calls"] = stats.get("tool_usage", {}).get("python_exec", 0)
        if stats.get("step_limit_reached"):
            row["step_limit_reached"] = True

    validator = problem_md.with_name(f"{problem_md.stem}_ground_truth.py")
    if not validator.is_file():
        row["valid"] = None
        row["message"] = "no validator"
    elif not row["solution"]:
        row["valid"] = False
        row["message"] = "no solution output"
    else:
        check = subprocess.run(
            [sys.executable, str(validator)],
            input=row["solution"],
            capture_output=True,
            text=True,
            timeout=120,
        )
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
    return row


def print_summary(rows: list[dict]) -> None:
    """Print a per-solver summary table to stderr."""
    by_solver: dict[str, list[dict]] = {}
    for row in rows:
        by_solver.setdefault(row["solver"], []).append(row)
    print("\nsolver   valid/run   tokens(avg)   exec(avg)   wall(avg)", file=sys.stderr)
    for solver, group in sorted(by_solver.items()):
        n = len(group)
        valid = sum(1 for r in group if r.get("valid"))
        tokens = [r["total_tokens"] for r in group if r.get("total_tokens")]
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
    parser.add_argument("--local-package", metavar="PATH", default=None)
    parser.add_argument(
        "--out",
        metavar="DIR",
        default=None,
        help="output directory (default: bench_results/<timestamp>)",
    )
    args = parser.parse_args(argv)

    solvers = args.solvers or list(SOLVERS)
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
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(run_one, s, p, args, out_root, i): (s, p) for s, p, i in jobs
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            with results_file.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            status = {True: "ok", False: "FAIL", None: "?"}[row.get("valid")]
            print(
                f"[{len(rows)}/{len(jobs)}] {status:<4} {row['solver']}/{row['problem']}"
                f" ({row['wall_seconds']}s)",
                file=sys.stderr,
            )

    print_summary(rows)
    print(str(results_file))
    return 0 if all(r.get("valid") for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
