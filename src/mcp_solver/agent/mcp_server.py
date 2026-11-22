"""mcp-solver-serve: the v4 MCP server — a solver toolkit for host LLMs.

The HOST does the solving. Claude Desktop, Claude Code, or mcp-minion (the
batchable open-source host substitute) connects here and writes, runs, and
verifies solver programs itself against a persistent IPython kernel. The
server contributes the solver value: backend selection, solver packages
injected into the kernel, the modeling instructions (templates), and
the ``submit_code`` finish line (compile check, then one standalone run of
the program, then its self-check as information). It calls no LLM and needs
no API key.

Kernel model: the solving kernel has the fixed engine kernel id
``"solver"`` (:data:`SOLVER_KERNEL`). Every ``select_backend`` call does one
``python_reset(kernel_id="solver", packages=<backend packages>)``, which
creates the kernel on first use and otherwise restarts it (state cleared,
packages swapped, also after a crash), so solving problem after problem
never accumulates kernel processes. Once a backend is selected,
``python_exec``, ``python_status``, ``python_interrupt`` and ``python_reset``
calls without ``kernel_id`` act on that kernel; a routed ``python_reset``
with packages keeps the backend's solver libraries. An explicit
``kernel_id`` is passed to the engine unchanged and never changes the
routing. Before any ``select_backend``, calls without ``kernel_id`` reach the
engine unchanged and act on its default kernel ``_default``.

Successive revelation to the host:
- Tier 0: tool descriptions — enough to start (call ``select_backend`` first).
- Tier 1: ``mcp-solver://guide`` and, on backend selection, the full modeling
  instructions for that backend (also browsable as
  ``mcp-solver://template/{solver}``).
- Tier 2: successful submissions become resources
  (``mcp-solver://submissions/{id}``), linked from the submit_code result.

The kernel tools are proxied over stdio to the engine's ``ipython_mcp``
server (agentic-python-coder), using only its public MCP contract; one
long-lived engine connection per server process, so kernel state persists
across the host's tool calls.
"""

import asyncio
import contextlib
import functools
import json
import logging
import math
import os
import re
import signal
import sys
import tempfile
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceError, ToolError
from mcp.types import CallToolResult, ResourceLink, TextContent

import mcp_solver
from mcp_solver.agent import engine
from mcp_solver.agent.cli import (
    CAPTURE_BUDGET_BYTES,
    ProgramRun,
    build_with_packages,
    run_isolated,
    signal_process_group,
)
from mcp_solver.templates import SOLVERS, get_template

SELECT_BACKEND_DESCRIPTION = """\
Start solving a constraint problem: call this FIRST, once per problem.

Sets up a persistent IPython kernel preloaded with the backend's solver
library, and returns the modeling instructions for that backend. Calling it
again recycles the kernel for a new problem (previous session state is
cleared).

Backends:
- pysat: Boolean satisfiability — feasibility, combinatorial search
- maxsat: weighted optimization over Boolean constraints
- z3: SMT — integers/reals/bitvectors, proofs, program verification
- cpmpy: finite-domain constraint programming — scheduling, assignment, puzzles
- clingo: Answer Set Programming — logic rules, defaults, reachability
- didp: dynamic programming (state-space search) — routing, sequencing, packing

Unsure which backend fits? Read the resource mcp-solver://guide first.

Then iterate with python_exec (write, run, verify against the problem
statement). Before calling submit_code, run the complete program with
python_exec, calling main(check=True), and check its output; submit_code is
for the finished, tested program, not for a first try. UNSAT / "no solution
exists" is a valid outcome.
"""

GUIDE = """\
# mcp-solver: how to solve a problem

Workflow: (1) `select_backend(solver)` — sets up the solving kernel with the
right solver library and returns modeling instructions; (2) iterate
`python_exec` (it runs in that kernel automatically): encode, run the real
solver, verify the result independently against the problem statement;
(3) finish with `submit_code` (the final self-contained program in the
template's shape: `solve()`, `verify(solution)`, `main(check=False)` that
runs `verify()` only when `check` is true and prints the solution, and a
`__main__` guard that calls `main()`). Before calling submit_code, run the
complete program with python_exec, calling `main(check=True)`, and check its
output; submit_code is for the finished, tested program, not for a first
try. The self-check is a heuristic: it must hold for every solution of the
model, not only the one the solver happened to return.

Choosing a backend:
- **pysat** — pure Boolean structure: placements, colorings, covers,
  pigeonhole-style feasibility. Fastest when everything is yes/no variables.
- **maxsat** — like pysat but with costs or preferences: maximize satisfied
  soft constraints under hard ones. Finds the true optimum, not a heuristic.
- **z3** — arithmetic and verification: integer/real/bitvector constraints,
  cryptarithmetic, proving a property holds (UNSAT of the negation),
  finding counterexamples, induction obligations.
- **cpmpy** — finite-domain CP: rosters, timetables, routing-style puzzles,
  global constraints (all-different, cumulative), optimization over integers.
- **clingo** — rule-based knowledge: transitive closure/reachability,
  defaults with exceptions, choice under logical rules, #minimize/#maximize.
- **didp** — dynamic programming as state-space search (didppy): sequential
  decisions with a natural state — routing with time windows, sequencing,
  scheduling, knapsack/packing. Often beats CP on these; optimality proofs
  via anytime beam search.

Tips: encode exactly the stated problem (every number matters); UNSAT is a
legitimate answer, re-check the encoding before reporting it; keep the final
program self-contained (all imports, no session state) and print only the
solution JSON. Long solver runs: python_exec defaults to a 30 s limit —
raise its `timeout` (up to 300) and give the solver an explicit time limit
rather than concluding the model is wrong. submit_code runs the finished
program on its own with a 240 s limit, environment setup included, so a
solver time limit in the submitted program should be at most about 200 s.
"""

# Engine connection (one per server process; kernels persist across calls).
_engine = None

# The solving kernel's engine id. select_backend creates or restarts it with
# python_reset; calls without kernel_id are routed to it once a backend is
# selected.
SOLVER_KERNEL = "solver"

# Solver packages of the selected backend; None until the first successful
# select_backend (no routing before that). The server dispatches tool calls
# concurrently, so select_backend and a routed python_reset run under this
# lock (python_exec does not take it, so a reset is never blocked by a
# running exec).
_backend_packages: list[str] | None = None
_kernel_lock = asyncio.Lock()
# The solving kernel's current packages: the backend's plus those a routed
# python_reset added. submit_code runs the submitted program with them.
_kernel_packages: list[str] | None = None

# Processes (Popen) of submit_code's standalone runs in flight; the
# lifespan kills those still registered at shutdown.
_standalone_runs: set = set()

# Server-side execution cap of the engine's python_exec (seconds); clamp
# here too so an oversized host value cannot stretch the client-side wait.
_MAX_EXEC_TIMEOUT = 300

# Bounded store of successful submissions (Tier 2 resources).
_submissions: dict[str, str] = {}
_MAX_SUBMISSIONS = 50

# submit_code runs an accepted program once on its own (as the CLI
# re-executes it): wall-clock limit in seconds (below the 300 s after which
# mcp-minion as host gives up on a tool call) and the characters of output kept in the reply (the end
# of stdout, where programs print the solution JSON).
STANDALONE_TIMEOUT = 240
_STDOUT_CAP = 20_000
_STDERR_CAP = 3_000

RUN_OK_MESSAGE = (
    "The server ran the program on its own in a clean environment with the"
    ' backend\'s packages; its output is in "stdout" and is the result, so'
    " there is no need to run the program again outside these tools."
)
RUN_OK_CUT_MESSAGE = (
    "The server ran the program on its own in a clean environment with the"
    f" backend's packages. Its output was longer than {_STDOUT_CAP:,}"
    ' characters, so "stdout" shows only its end; run the program to obtain'
    " the complete output."
)
NOT_EXECUTED_MESSAGE = "Compile check only, not executed: no backend selected."

# After an accepted plain run, submit_code runs the program's self-check,
# main(check=True), as information: it never decides acceptance. The driver
# below loads the program as a module (so its __main__ guard does not run)
# and calls main(check=True); without a main() that accepts check=True it
# exits with _NO_CHECK_EXIT after writing _NO_CHECK_MARKER, and once
# main(check=True) has returned (or exited with code 0) it writes
# _CHECK_DONE_MARKER, so an exit during the import is not taken for a
# passed check. The check runs within what is left of STANDALONE_TIMEOUT,
# measured right before its process starts; with less than
# _MIN_CHECK_SECONDS left it is not run.
_MIN_CHECK_SECONDS = 5
_NO_CHECK_EXIT = 86
_NO_CHECK_MARKER = "mcp-solver: no main() that accepts check=True"
_CHECK_DONE_MARKER = "mcp-solver: main(check=True) returned"
_CHECK_DRIVER = f"""\
import importlib.util
import inspect
import sys
from pathlib import Path

_path = Path(__file__).resolve().parent / "submission.py"
_spec = importlib.util.spec_from_file_location("submission", _path)
_module = importlib.util.module_from_spec(_spec)
sys.modules["submission"] = _module
_spec.loader.exec_module(_module)
_main = getattr(_module, "main", None)
try:
    inspect.signature(_main).bind(check=True)
except (TypeError, ValueError):
    sys.stderr.write({_NO_CHECK_MARKER!r} + "\\n")
    sys.exit({_NO_CHECK_EXIT})
try:
    _main(check=True)
except SystemExit as _exit:
    if _exit.code not in (None, 0):
        raise
sys.stdout.flush()
sys.stderr.write("\\n" + {_CHECK_DONE_MARKER!r} + "\\n")
sys.stderr.flush()
"""

# Solve statistics per episode (episode = one select_backend up to the next
# one, or server shutdown). The host's tokens are invisible to the server,
# but its tool usage is not: counts, exec failures, submissions, wall time.
# One JSONL line per episode goes to $MCP_SOLVER_STATS when set; a compact
# snapshot rides along in submit_code's structuredContent.
_episode: dict | None = None


def _episode_open(solver: str) -> None:
    global _episode
    _episode_close("superseded")
    _episode = {
        "solver": solver,
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "t0": time.monotonic(),
        "tool_calls": {"select_backend": 1},
        "exec_failures": 0,
        "submit_attempts": 0,
        "submit_ok": 0,
        # python_exec calls before the first submit_code (None until then):
        # 0 means the host submitted without running anything in the kernel.
        "exec_before_first_submit": None,
        "code_bytes": None,
        # Self-check of the last accepted submission (None: none accepted
        # after a standalone run): passed, failed or not_available.
        "self_check": None,
    }


def _episode_record(end: str) -> dict:
    record = {k: v for k, v in _episode.items() if k != "t0"}
    record["wall_seconds"] = round(time.monotonic() - _episode["t0"], 1)
    record["end"] = end
    return record


def _episode_sync() -> None:
    """Snapshot the open episode to <stats>.open after every counted call.

    Hosts often kill the server without a clean shutdown (claude -p does),
    so waiting for episode close would lose single-episode sessions
    entirely. The snapshot is overwritten in place and removed on close.
    """
    path = os.environ.get("MCP_SOLVER_STATS")
    if not path or _episode is None:
        return
    try:
        with open(path + ".open", "w", encoding="utf-8") as fh:
            fh.write(json.dumps(_episode_record("open")) + "\n")
    except OSError as e:
        print(f"mcp-solver-serve: cannot write stats to {path}: {e}", file=sys.stderr)


def _episode_close(end: str) -> None:
    global _episode
    if _episode is None:
        return
    record = _episode_record(end)
    _episode = None
    path = os.environ.get("MCP_SOLVER_STATS")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        with contextlib.suppress(FileNotFoundError):
            os.remove(path + ".open")
    except OSError as e:
        print(f"mcp-solver-serve: cannot write stats to {path}: {e}", file=sys.stderr)


def _episode_count(tool: str) -> None:
    if _episode is not None:
        calls = _episode["tool_calls"]
        calls[tool] = calls.get(tool, 0) + 1
        _episode_sync()


def _episode_snapshot(episode: dict | None) -> dict | None:
    if episode is None:
        return None
    return {
        "exec_calls": episode["tool_calls"].get("python_exec", 0),
        "exec_failures": episode["exec_failures"],
        "submit_attempts": episode["submit_attempts"],
        "exec_before_first_submit": episode["exec_before_first_submit"],
        "self_check": episode["self_check"],
        "wall_seconds": round(time.monotonic() - episode["t0"], 1),
    }


def _dev_path() -> str | None:
    """Dev-mode source checkout for templates (env or auto-detect).

    Raises DevModeError when MCP_SOLVER_DEV asks for auto-detection and no
    checkout exists (the same error the CLI reports).
    """
    return engine.resolve_dev_path()


@asynccontextmanager
async def _lifespan(server: MCPServer) -> AsyncIterator[None]:
    """Hold one engine (ipython_mcp) connection for the server's lifetime."""
    global _engine, _backend_packages, _kernel_packages
    manager = await engine.start_engine()
    _engine = manager
    try:
        yield
    finally:
        for proc in list(_standalone_runs):
            signal_process_group(proc, signal.SIGKILL)
        _standalone_runs.clear()
        _episode_close("shutdown")
        _engine = None
        _backend_packages = None
        _kernel_packages = None
        await manager.__aexit__(None, None, None)


mcp = MCPServer(
    "mcp-solver",
    version=mcp_solver.__version__,
    instructions=(
        "Constraint-solving toolkit: call select_backend(solver) first, then"
        " write and verify a solver program via python_exec, and finish with"
        " submit_code. Backend choice: see resource mcp-solver://guide."
    ),
    lifespan=_lifespan,
)


async def _engine_call(name: str, arguments: dict) -> str:
    """Proxy one tool call to the engine; return the engine tool's text.

    An error outcome (an MCP ``isError`` result, or a client-level failure
    such as engine down or timeout) becomes a ToolError, i.e. an error
    result for the host. Engine-level outcomes reported as normal results,
    including failed executions (``"success": false``), pass through to the
    host untouched.
    """
    if _engine is None:
        raise ToolError("engine not connected (server starting or shut down)")
    outcome = await _engine.call_tool(name, arguments)
    if outcome.is_error:
        raise ToolError(outcome.text or f"engine tool {name!r} failed")
    return outcome.text


def _route(kernel_id: str | None) -> str | None:
    """The kernel id to send: an explicit id unchanged, else the solving
    kernel once a backend is selected, else None (the engine's default)."""
    if kernel_id:
        return kernel_id
    return SOLVER_KERNEL if _backend_packages is not None else None


def _merge_packages(backend: list[str], extra: list[str]) -> list[str]:
    """The backend's packages first, then *extra*, without duplicates."""
    return list(dict.fromkeys([*backend, *extra]))


def _parse_engine_json(text: str) -> dict:
    """Parse an engine tool's JSON text; {} when it isn't a JSON object."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


@mcp.resource("mcp-solver://guide", mime_type="text/markdown")
def guide() -> str:
    """Backend selection and solving workflow guide."""
    return GUIDE


@mcp.resource("mcp-solver://template/{solver}", mime_type="text/markdown")
def template_resource(solver: str) -> str:
    """The full modeling instructions for a backend."""
    if solver not in SOLVERS:
        raise ResourceError(f"unknown solver {solver!r}; expected one of {SOLVERS}")
    try:
        root = _dev_path()
    except engine.DevModeError as e:
        raise ResourceError(str(e)) from None
    return get_template(solver, root=root)


@mcp.resource("mcp-solver://submissions/{submission_id}", mime_type="text/x-python")
def submission_resource(submission_id: str) -> str:
    """A previously submitted (compile-checked) solver program."""
    code = _submissions.get(submission_id)
    if code is None:
        raise ResourceError(f"unknown submission {submission_id!r}")
    return code


@mcp.tool(description=SELECT_BACKEND_DESCRIPTION, structured_output=False)
async def select_backend(solver: str) -> str:
    global _backend_packages, _kernel_packages
    if solver not in SOLVERS:
        raise ToolError(f"unknown solver {solver!r}; expected one of {SOLVERS}")

    try:
        dev = _dev_path()
    except engine.DevModeError as e:
        raise ToolError(str(e)) from None
    packages = build_with_packages(solver)

    # One reset creates the solving kernel (first use) or restarts it
    # (backend switch, new problem, after a crash); "created" tells which.
    async with _kernel_lock:
        parsed = _parse_engine_json(
            await _engine_call(
                "python_reset", {"kernel_id": SOLVER_KERNEL, "packages": packages}
            )
        )
        if not parsed.get("success"):
            raise ToolError(
                f"kernel setup for {solver!r} failed:"
                f" {parsed.get('error') or 'unknown engine error'}"
            )
        recycled = parsed.get("created") is False
        _backend_packages = packages
        _kernel_packages = packages
        _episode_open(solver)
        _episode_sync()

    state_note = (
        "The solving kernel was recycled: previous session state is cleared."
        if recycled
        else "A fresh solving kernel is ready."
    )
    kernel_note = (
        f"{state_note} Kernel id: {SOLVER_KERNEL}. Packages:"
        f" {', '.join(packages)}. python_exec,"
        " python_status, python_interrupt and python_reset without kernel_id"
        " act on this kernel. Before calling submit_code, run the complete"
        " program with python_exec, calling main(check=True), and check its"
        " output; submit_code is for the finished, tested program (the"
        " template's shape, self-contained), not for a first try."
    )
    template = get_template(solver, root=dev)
    # Plain text on purpose: the template IS the payload, and clients that
    # prefer structuredContent over text (Claude Code does) would otherwise
    # hide it from the model. Field-tested 2026-07-12.
    return f"{template}\n\n---\n\n{kernel_note}"


@mcp.tool(structured_output=False)
async def python_exec(
    code: str, timeout: float = 30, kernel_id: str | None = None
) -> str:
    """Execute Python code in the persistent solving kernel.

    Without kernel_id it runs in the kernel set up by select_backend; an
    explicit kernel_id runs it in that kernel instead. State persists across
    calls (variables, imports). Raise timeout (seconds, max 300) for long
    solver runs.
    """
    _episode_count("python_exec")
    if not math.isfinite(timeout):
        timeout = 30.0
    args: dict = {"code": code, "timeout": max(1.0, min(timeout, _MAX_EXEC_TIMEOUT))}
    target = _route(kernel_id)
    if target:
        args["kernel_id"] = target
    text = await _engine_call("python_exec", args)
    if _episode is not None and _parse_engine_json(text).get("success") is False:
        _episode["exec_failures"] += 1
        _episode_sync()
    return text


@mcp.tool(structured_output=False)
async def python_reset(
    packages: list[str] | None = None, kernel_id: str | None = None
) -> str:
    """Restart the solving kernel: state cleared, solver packages kept.

    Without kernel_id (or with kernel_id="solver") it acts on the kernel set
    up by select_backend: a non-empty packages list is installed next to the
    backend's solver libraries (replacing packages added by an earlier
    reset); without it the current packages stay. Another kernel_id restarts
    that kernel, or creates it under that name, with the packages as given;
    it does not change which kernel calls without kernel_id use. To switch
    backends, call select_backend instead.
    """
    global _kernel_packages
    _episode_count("python_reset")
    if _route(kernel_id) == SOLVER_KERNEL:
        async with _kernel_lock:
            backend = _backend_packages
            if backend is not None:
                args: dict = {"kernel_id": SOLVER_KERNEL}
                if packages:
                    args["packages"] = _merge_packages(backend, packages)
                text = await _engine_call("python_reset", args)
                if packages and _parse_engine_json(text).get("success") is True:
                    _kernel_packages = args["packages"]
                return text
    args = {}
    if kernel_id:
        args["kernel_id"] = kernel_id
    if packages:
        args["packages"] = packages
    text = await _engine_call("python_reset", args)
    if kernel_id and _backend_packages is not None:
        text = _with_routing_note(text, kernel_id)
    return text


def _with_routing_note(text: str, kernel_id: str) -> str:
    """Append the server's routing to a successful reset of another kernel.

    The engine's message says that calls without kernel_id use its default
    kernel; behind this server they go to the solving kernel once a backend
    is selected. Other replies are returned unchanged.
    """
    parsed = _parse_engine_json(text)
    message = parsed.get("message")
    if parsed.get("success") is not True or not isinstance(message, str):
        return text
    parsed["message"] = (
        f"{message} Through mcp-solver-serve, calls without kernel_id go to the"
        f' solving kernel "{SOLVER_KERNEL}"; pass kernel_id="{kernel_id}" to use'
        " this kernel."
    )
    return json.dumps(parsed)


@mcp.tool(structured_output=False)
async def python_status(kernel_id: str | None = None) -> str:
    """Check kernel state: active kernels, defined variables, packages.

    Without kernel_id it reports the kernel set up by select_backend.
    """
    _episode_count("python_status")
    args: dict = {}
    target = _route(kernel_id)
    if target:
        args["kernel_id"] = target
    return await _engine_call("python_status", args)


@mcp.tool(structured_output=False)
async def python_interrupt(kernel_id: str | None = None) -> str:
    """Interrupt running code in the solving kernel (state is preserved)."""
    _episode_count("python_interrupt")
    args: dict = {}
    target = _route(kernel_id)
    if target:
        args["kernel_id"] = target
    return await _engine_call("python_interrupt", args)


def _run_standalone(
    code: str,
    packages: list[str],
    on_start=None,
    *,
    deadline: float | None = None,
    check: bool = False,
) -> ProgramRun | None:
    """Run *code* once on its own: a fresh temporary directory, an isolated
    uv environment with *packages*, the filtered environment, its own
    process group, STANDALONE_TIMEOUT. With *check* it runs the self-check
    driver instead, which imports the program and calls main(check=True).
    With *deadline* (a time.monotonic() value) the run gets the time left
    until then, measured right before it starts; None is returned without a
    run when less than _MIN_CHECK_SECONDS are left. Blocking; submit_code
    calls it in a worker thread (*on_start* receives the Popen, see
    run_isolated)."""
    with tempfile.TemporaryDirectory(
        prefix="mcp-solver-run-", ignore_cleanup_errors=True
    ) as tmp:
        program = Path(tmp) / "submission.py"
        program.write_text(code, encoding="utf-8")
        if check:
            program = Path(tmp) / "self_check.py"
            program.write_text(_CHECK_DRIVER, encoding="utf-8")
        timeout = STANDALONE_TIMEOUT
        if deadline is not None:
            timeout = deadline - time.monotonic()
            if timeout < _MIN_CHECK_SECONDS:
                return None
        try:
            return run_isolated(
                program,
                packages,
                timeout,
                capture=True,
                cwd=tmp,
                on_start=on_start,
            )
        except FileNotFoundError:
            return ProgramRun(None, "", "uv not found on PATH", False)


def _tail(text: str | None, limit: int) -> tuple[str, bool]:
    """The last *limit* characters of *text*, and whether it was cut."""
    text = text or ""
    return (text[-limit:], True) if len(text) > limit else (text, False)


_ANSI = re.compile(r"\x1b\[[0-9;]*m")
# clingo's own error lines, e.g. "<block>:3:5-9: error: syntax error, ..."
_CLINGO_ERROR = re.compile(r"^\S.*?:\d+:\d+(?:-[\d:]+)?: error: .+$")
_EXC_LINE = re.compile(r"^([A-Za-z_][\w.]*)(?::\s|$)")
_FIX = "Fix it with python_exec and submit again."
_MAX_EXC_LINE = 500


def _final_exception(stderr: str) -> tuple[str, str, bool] | None:
    """The propagated exception of a Python traceback in *stderr*.

    Returns (exception type without module, its line, whether a frame of the
    traceback is in a function named verify), or None without a traceback.
    The last "Traceback" header is the exception that ended the program
    (chained exceptions print theirs first); its exception line is the first
    unindented line after the indented frames.
    """
    lines = _ANSI.sub("", stderr).splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.startswith("Traceback (most")]
    if not starts:
        return None
    in_verify = False
    for ln in lines[starts[-1] + 1 :]:
        if not ln.strip():
            continue
        if ln[0].isspace():
            frame = ln.strip()
            if frame.startswith("File ") and frame.endswith(", in verify"):
                in_verify = True
            continue
        match = _EXC_LINE.match(ln)
        if not match:
            return None
        line = ln.strip()
        if len(line) > _MAX_EXC_LINE:
            line = line[:_MAX_EXC_LINE] + " ..."
        return match.group(1).rsplit(".", 1)[-1], line, in_verify
    return None


def _quoted_name(line: str) -> str | None:
    """The first quoted name in an exception line ('x' in "name 'x' ...")."""
    match = re.search(r"'([^']+)'", line)
    return match.group(1) if match else None


def _sentence(text: str) -> str:
    return text if text.endswith((".", "!", "?")) else text + "."


def _run_failure(run: ProgramRun, packages: list[str]) -> str | None:
    """The rejection text for a failed standalone run, None when it succeeded.

    Leads with what happened (for an exception its last traceback line),
    then one sentence for that kind of failure; a cause is named only where
    the failure shows it (kernel-session state only for NameError and
    import failures).
    """
    if run.output_exceeded:
        killed = ", and was killed" if run.exit_code is None else ""
        return (
            f"Too much output: the program wrote more than"
            f" {CAPTURE_BUDGET_BYTES:,} bytes (stdout and stderr together) when"
            f" the server ran it on its own{killed}. It must print only the"
            " solution JSON; remove progress, debug and solver log output. " + _FIX
        )
    if run.exit_code is None and not run.timed_out:
        reason = _sentence((run.stderr or "").strip() or "unknown reason")
        return (
            f"The run could not start: {reason} This is a problem of the"
            " server's setup, not of the program: install uv"
            " (https://docs.astral.sh/uv/) where mcp-solver-serve runs, restart"
            " the server, and submit the program unchanged."
        )
    if run.timed_out:
        return (
            f"Timed out: the program did not finish within {STANDALONE_TIMEOUT} s"
            " when the server ran it on its own, and was killed. Run the"
            " complete program with python_exec (raise its timeout) to find"
            " where the time goes. " + _FIX
        )
    stderr = run.stderr or ""
    if run.exit_code == 0:
        if (run.stdout or "").strip():
            return None
        return (
            "No output: the program exited with code 0 but printed nothing"
            " when the server ran it on its own. It must print the solution"
            " JSON. " + _FIX
        )
    exc = _final_exception(stderr)
    if exc is None:
        last = next((ln for ln in reversed(stderr.splitlines()) if ln.strip()), "")
        what = (
            f"Exit code {run.exit_code}"
            if run.exit_code is not None
            else "The run could not start"
        )
        detail = f" Last line of stderr: {last.strip()}" if last else ""
        return f"{what} when the server ran the program on its own.{detail} " + _FIX
    kind, line, in_verify = exc
    head = _sentence(line)
    if kind == "AssertionError":
        why = (
            "The program's own verify() rejected its solution, so the solution"
            " or the check is wrong."
            if in_verify
            else "An assert in the program failed, so the computed result or"
            " the assertion is wrong."
        )
    elif kind == "NameError":
        name = _quoted_name(line)
        why = (
            f"The program uses {name!r} without defining or importing it: a typo,"
            " or a variable, function or import from the kernel session, which"
            " a submitted program cannot rely on."
            if name
            else "The program uses a name without defining or importing it."
        )
    elif kind in ("ModuleNotFoundError", "ImportError"):
        name = _quoted_name(line)
        what = repr(name) if name else "it"
        why = (
            f"The run has only the packages {', '.join(packages)}, without"
            f" {what}; if the kernel session had it, add its package with"
            " python_reset(packages=[...]), or do without it."
        )
    else:
        errors = [ln.strip() for ln in stderr.splitlines() if _CLINGO_ERROR.match(ln)]
        if errors:
            shown = "; ".join(errors[:5]) + ("; ..." if len(errors) > 5 else "")
            head += f" Clingo reported: {_sentence(shown)}"
            why = (
                "Clingo rejected the ASP program text; its positions are"
                " line:column in that text, not in the Python file."
            )
        else:
            why = (
                f"The program raised {kind} when the server ran it on its own;"
                ' the traceback is in "stderr".'
            )
    return f"{head} {why} {_FIX}"


SELF_CHECK_PASSED = "passed"
SELF_CHECK_FAILED = "failed"
SELF_CHECK_NOT_AVAILABLE = "not_available"
# The last sentence of every not-available and failed self-check message.
_SELF_CHECK_INFO = (
    " The submission is accepted on its plain run; the self-check is information only."
)


def _capped(text: str) -> str:
    """*text* cut to _MAX_EXC_LINE characters, marked with " ..." if cut."""
    return text if len(text) <= _MAX_EXC_LINE else text[:_MAX_EXC_LINE] + " ..."


def _self_check_outcome(run: ProgramRun | None) -> tuple[str, str]:
    """The self-check's status and message for the reply.

    *run* is the run of main(check=True), None when it was not run because
    less than _MIN_CHECK_SECONDS of the budget were left. "passed" needs the
    driver's completion marker: main(check=True) returned or exited with
    code 0.
    """
    budget = f"{STANDALONE_TIMEOUT} s budget"
    not_available = "Self-check not available: "
    stderr = (run.stderr or "") if run is not None else ""
    if run is None:
        what = (
            f"less than {_MIN_CHECK_SECONDS} s of the {budget} were left for"
            " main(check=True) after the plain run."
        )
    elif run.output_exceeded:
        return SELF_CHECK_FAILED, _self_check_failed(
            f"wrote more than {CAPTURE_BUDGET_BYTES:,} bytes of output"
        )
    elif run.timed_out:
        what = f"main(check=True) did not finish before the {budget} ran out."
    elif run.exit_code is None:
        reason = _capped((stderr.strip() or "unknown reason").rstrip("."))
        what = f"its run could not start ({reason})."
    elif run.exit_code == _NO_CHECK_EXIT and _NO_CHECK_MARKER in stderr:
        what = "the program has no main() that accepts check=True."
    elif run.exit_code == 0 and _CHECK_DONE_MARKER in stderr:
        return SELF_CHECK_PASSED, (
            "Self-check passed: main(check=True) ran and returned without error."
        )
    elif run.exit_code == 0:
        what = (
            "the program exited with code 0 before main(check=True) returned"
            " (for example during its import)."
        )
    else:
        exc = _final_exception(stderr)
        if exc is not None:
            return SELF_CHECK_FAILED, _self_check_failed(f"raised {exc[1]}")
        last = next(
            (ln.strip() for ln in reversed(stderr.splitlines()) if ln.strip()), ""
        )
        detail = f"ended with exit code {run.exit_code}"
        if last:
            detail += f" (last line of stderr: {_capped(last)})"
        return SELF_CHECK_FAILED, _self_check_failed(detail)
    return SELF_CHECK_NOT_AVAILABLE, not_available + what + _SELF_CHECK_INFO


def _self_check_failed(what: str) -> str:
    return (
        f"Self-check failed: main(check=True) {_sentence(what)} Either the"
        " solution or verify() is wrong: check both against the problem"
        " statement, and if the solution is wrong, fix the program and submit"
        " it again." + _SELF_CHECK_INFO
    )


@mcp.tool()
async def submit_code(code: str) -> CallToolResult:
    """Submit the final, verified, self-contained solver program.

    Before calling submit_code, run the complete program with python_exec,
    calling main(check=True), and check its output; submit_code is for the
    finished, tested program, not for a first try.
    The server checks that it compiles and, once a backend is selected,
    runs it once more on its own in a clean environment (no kernel state,
    at most 240 s including environment setup, at most 4 MB of output) and
    returns its output; a program that fails there is rejected with the
    reason and may be fixed and submitted again. After an accepted run the
    server calls main(check=True) in a separate fresh isolated environment
    with the same packages and filtered environment variables, within what
    is left of the 240 s, and reports the self-check as passed, failed or
    not available; the self-check does not decide acceptance.
    Accepted programs are stored and linked as a resource (the last 50).
    """
    # The episode and the packages of the backend this submission is made
    # under, taken before the first await: a select_backend during the
    # submission changes neither its run nor which episode it is counted on.
    episode = _episode
    packages = _kernel_packages
    # Count the attempt BEFORE _episode_count syncs the .open snapshot, so a
    # failed submission (which never syncs again) is not lost on a host kill.
    if episode is not None:
        if episode["submit_attempts"] == 0:
            episode["exec_before_first_submit"] = episode["tool_calls"].get(
                "python_exec", 0
            )
        episode["submit_attempts"] += 1
    _episode_count("submit_code")
    result_text = await _engine_call("submit_code", {"code": code})
    verdict = _parse_engine_json(result_text)

    accepted = bool(verdict.get("ok"))
    # A rejection ({"ok": false, "error": ...} from the engine) is an error
    # result, so every host sees it as a failed call.
    text = result_text if accepted else str(verdict.get("error") or result_text)
    reply = dict(verdict)
    if accepted and packages is None:
        reply["executed"] = False
        reply["message"] = NOT_EXECUTED_MESSAGE
    elif accepted:
        # Run without the kernel lock and off the event loop: kernel tools
        # (python_status, python_interrupt, a reset) stay responsive.
        # A cancelled call (or a disconnect) kills the run: the worker thread
        # cannot be interrupted, its process group can.
        procs: list = []
        cancelled = False

        def started(proc) -> None:
            procs.append(proc)
            _standalone_runs.add(proc)
            # Checked after the add: a start that comes after the
            # cancellation (and after the finally below has deregistered
            # procs) is killed and deregistered here, nothing else would.
            if cancelled:
                signal_process_group(proc, signal.SIGKILL)
                _standalone_runs.discard(proc)

        # The plain run decides acceptance. After an accepted one, the
        # self-check, main(check=True), runs within what is left of the
        # budget and is only reported: no failure of it, its setup
        # included, rejects the submission (cancellation still propagates).
        check_run = None
        try:
            deadline = time.monotonic() + STANDALONE_TIMEOUT
            run = await asyncio.to_thread(
                _run_standalone, code, list(packages), started
            )
            failure = _run_failure(run, list(packages))
            if failure is None:
                try:
                    check_run = await asyncio.to_thread(
                        functools.partial(
                            _run_standalone,
                            code,
                            list(packages),
                            started,
                            deadline=deadline,
                            check=True,
                        )
                    )
                except Exception as e:
                    check_run = ProgramRun(
                        None, None, f"{type(e).__name__}: {e}", False
                    )
        except asyncio.CancelledError:
            cancelled = True
            for proc in procs:
                signal_process_group(proc, signal.SIGKILL)
            raise
        finally:
            _standalone_runs.difference_update(procs)
        stdout, out_cut = _tail(run.stdout, _STDOUT_CAP)
        notes = [f"stdout cut to its last {_STDOUT_CAP} characters."] if out_cut else []
        if failure is None:
            message = reply.get("message")
            ran = RUN_OK_CUT_MESSAGE if out_cut else RUN_OK_MESSAGE
            status, check_message = _self_check_outcome(check_run)
            reply.update(
                executed=True,
                exit_code=0,
                stdout=stdout,
                message=f"{message} {ran}" if message else ran,
                self_check={"status": status, "message": check_message},
            )
            if episode is not None:
                episode["self_check"] = status
        else:
            accepted = False
            stderr, err_cut = _tail(run.stderr, _STDERR_CAP)
            if err_cut:
                notes.append(f"stderr cut to its last {_STDERR_CAP} characters.")
            reply = {
                "ok": False,
                # False when the run could not start (no uv on PATH).
                "executed": run.exit_code is not None
                or run.timed_out
                or run.output_exceeded,
                "exit_code": run.exit_code,
                "timed_out": run.timed_out,
                "stdout": stdout,
                "stderr": stderr,
                "error": failure,
            }
        if notes:
            reply["note"] = " ".join(notes)
    if reply != verdict:
        text = json.dumps(reply)
    content: list = [TextContent(type="text", text=text)]
    if accepted:
        if episode is not None:
            episode["submit_ok"] += 1
            episode["code_bytes"] = len(code.encode())
            if episode is _episode:
                _episode_sync()
        submission_id = uuid.uuid4().hex
        _submissions[submission_id] = code
        while len(_submissions) > _MAX_SUBMISSIONS:
            _submissions.pop(next(iter(_submissions)))
        content.append(
            ResourceLink(
                type="resource_link",
                uri=f"mcp-solver://submissions/{submission_id}",
                name=f"submission_{submission_id[:8]}.py",
                description="The accepted solver program.",
                mime_type="text/x-python",
                size=len(code.encode()),
            )
        )
    structured = reply
    stats = _episode_snapshot(episode)
    if stats:
        structured["stats"] = stats
    return CallToolResult(
        content=content,
        structured_content=structured or None,
        is_error=not accepted,
    )


def main() -> None:
    """Entry point for mcp-solver-serve (stdio transport)."""
    # The mcp library logs every request at INFO ("Processing request of
    # type CallToolRequest"), which is pure noise on the host's console
    # or log; keep warnings and errors.
    logging.getLogger("mcp").setLevel(logging.WARNING)
    mcp.run()


if __name__ == "__main__":
    main()
