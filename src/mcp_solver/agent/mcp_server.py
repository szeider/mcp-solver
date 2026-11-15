"""mcp-solver-serve: the v4 MCP server — a solver toolkit for host LLMs.

The HOST does the solving. Claude Desktop, Claude Code, or mcp-minion (the
batchable open-source host substitute) connects here and writes, runs, and
verifies solver programs itself against a persistent IPython kernel. The
server contributes the solver value: backend selection, solver packages
injected into the kernel, the modeling instructions (templates), and
the ``submit_code`` finish line (compile check, then one standalone run of
the program). It calls no LLM and needs no API key.

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
import json
import logging
import math
import os
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

from mcp_solver.agent import engine
from mcp_solver.agent.cli import ProgramRun, build_with_packages, run_isolated
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
statement), and finish by calling submit_code with the final, verified,
self-contained program. UNSAT / "no solution exists" is a valid outcome.
"""

GUIDE = """\
# mcp-solver: how to solve a problem

Workflow: (1) `select_backend(solver)` — sets up the solving kernel with the
right solver library and returns modeling instructions; (2) iterate
`python_exec` (it runs in that kernel automatically): encode, run the real
solver, verify the result independently against the problem statement;
(3) finish with `submit_code` (the final self-contained program in the
template's shape: `solve()`, `verify(solution)`, a `__main__` guard that
verifies before printing; the server runs it on its own and returns its
output, and rejects it if verify() fails).

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
rather than concluding the model is wrong.
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
    ' backend\'s packages; its output is in "stdout", so there is no need to'
    " run it anywhere else."
)
NOT_EXECUTED_MESSAGE = "Compile check only, not executed: no backend selected."

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
        "code_bytes": None,
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


def _episode_snapshot() -> dict | None:
    if _episode is None:
        return None
    return {
        "exec_calls": _episode["tool_calls"].get("python_exec", 0),
        "exec_failures": _episode["exec_failures"],
        "submit_attempts": _episode["submit_attempts"],
        "wall_seconds": round(time.monotonic() - _episode["t0"], 1),
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
        _episode_close("shutdown")
        _engine = None
        _backend_packages = None
        _kernel_packages = None
        await manager.__aexit__(None, None, None)


mcp = MCPServer(
    "mcp-solver",
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
        " act on this kernel."
        " Finish with submit_code(final self-contained program in the"
        " template's shape: solve(), verify(solution), __main__ guard);"
        " the server then runs it on its own and returns its output."
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


def _run_standalone(code: str, packages: list[str]) -> ProgramRun:
    """Run *code* once on its own: a fresh temporary directory, an isolated
    uv environment with *packages*, the filtered environment, its own
    process group, STANDALONE_TIMEOUT. Blocking; submit_code calls it in a
    worker thread."""
    with tempfile.TemporaryDirectory(
        prefix="mcp-solver-run-", ignore_cleanup_errors=True
    ) as tmp:
        program = Path(tmp) / "submission.py"
        program.write_text(code, encoding="utf-8")
        try:
            return run_isolated(
                program, packages, STANDALONE_TIMEOUT, capture=True, cwd=tmp
            )
        except FileNotFoundError:
            return ProgramRun(None, "", "uv not found on PATH", False)


def _tail(text: str | None, limit: int) -> tuple[str, bool]:
    """The last *limit* characters of *text*, and whether it was cut."""
    text = text or ""
    return (text[-limit:], True) if len(text) > limit else (text, False)


def _run_failure(run: ProgramRun) -> str | None:
    """Why a standalone run failed (a clause), or None when it succeeded."""
    if run.timed_out:
        return f"it did not finish within {STANDALONE_TIMEOUT} s and was killed"
    if run.exit_code != 0:
        return f"exit code {run.exit_code}"
    if not (run.stdout or "").strip():
        return "it printed nothing"
    return None


@mcp.tool()
async def submit_code(code: str) -> CallToolResult:
    """Submit the final, verified, self-contained solver program.

    Call when the program is complete and its verify() passes; if the reply
    rejects it, fix it and call again.
    The server checks that it compiles and, once a backend is selected,
    runs it on its own in a clean environment (no kernel state) and returns
    its output; a program that fails there is rejected with the reason.
    Accepted programs are stored and linked as a resource (the last 50).
    """
    # Count the attempt BEFORE _episode_count syncs the .open snapshot, so a
    # failed submission (which never syncs again) is not lost on a host kill.
    if _episode is not None:
        _episode["submit_attempts"] += 1
    _episode_count("submit_code")
    result_text = await _engine_call("submit_code", {"code": code})
    verdict = _parse_engine_json(result_text)

    accepted = bool(verdict.get("ok"))
    # A rejection ({"ok": false, "error": ...} from the engine) is an error
    # result, so every host sees it as a failed call.
    text = result_text if accepted else str(verdict.get("error") or result_text)
    reply = dict(verdict)
    packages = _kernel_packages
    if accepted and packages is None:
        reply["executed"] = False
        reply["message"] = NOT_EXECUTED_MESSAGE
    elif accepted:
        # Run without the kernel lock and off the event loop: kernel tools
        # (python_status, python_interrupt, a reset) stay responsive.
        run = await asyncio.to_thread(_run_standalone, code, list(packages))
        stdout, out_cut = _tail(run.stdout, _STDOUT_CAP)
        notes = [f"stdout cut to its last {_STDOUT_CAP} characters."] if out_cut else []
        failure = _run_failure(run)
        if failure is None:
            message = reply.get("message")
            reply.update(
                executed=True,
                exit_code=0,
                stdout=stdout,
                message=f"{message} {RUN_OK_MESSAGE}" if message else RUN_OK_MESSAGE,
            )
        else:
            accepted = False
            stderr, err_cut = _tail(run.stderr, _STDERR_CAP)
            if err_cut:
                notes.append(f"stderr cut to its last {_STDERR_CAP} characters.")
            reply = {
                "ok": False,
                "executed": True,
                "exit_code": run.exit_code,
                "timed_out": run.timed_out,
                "stdout": stdout,
                "stderr": stderr,
                "error": (
                    "The program compiles but failed when run on its own in a"
                    f" clean environment ({failure}); the usual cause is that it"
                    " relies on variables or imports from the kernel session,"
                    " but a submitted program must be self-contained: fix it"
                    " with python_exec and submit again."
                ),
            }
        if notes:
            reply["note"] = " ".join(notes)
    if reply != verdict:
        text = json.dumps(reply)
    content: list = [TextContent(type="text", text=text)]
    if accepted:
        if _episode is not None:
            _episode["submit_ok"] += 1
            _episode["code_bytes"] = len(code.encode())
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
    stats = _episode_snapshot()
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
