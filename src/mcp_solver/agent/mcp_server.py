"""mcp-solver-serve: the v4 MCP server — a solver toolkit for host LLMs.

The HOST does the solving. Claude Desktop, Claude Code, or mcp-minion (the
batchable open-source host substitute) connects here and writes, runs, and
verifies solver programs itself against a persistent IPython kernel. The
server contributes the solver value: backend selection, solver packages and
helpers injected into the kernel, the modeling instructions (templates), and
the compile-gated ``submit_code`` finish line. It calls no LLM and needs no
API key.

Kernel model: ``select_backend`` sets up ONE solving kernel and recycles it
(state cleared, packages swapped) on every later call, so solving problem
after problem never accumulates kernel processes. Bare ``python_exec``/
``python_interrupt`` calls are routed to that kernel automatically; an
explicit ``kernel_id`` overrides the routing. A manual ``python_reset``
without ``kernel_id`` creates an additional, independent kernel (which then
becomes the routing target) — those live until the server exits.

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
import math
import os
import sys
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import CallToolResult, ResourceLink, TextContent
from pydantic import AnyUrl

import mcp_solver
from mcp_solver.agent.cli import (
    _is_unpublishable,
    _local_checkout,
    build_with_packages,
)
from mcp_solver.templates import SOLVERS, get_template

SELECT_BACKEND_DESCRIPTION = """\
Start solving a constraint problem: call this FIRST, once per problem.

Sets up a persistent IPython kernel preloaded with the backend's solver
library and helper functions, and returns the modeling instructions for that
backend. Calling it again recycles the kernel for a new problem (previous
session state is cleared).

Backends:
- pysat: Boolean satisfiability — feasibility, combinatorial search
- maxsat: weighted optimization over Boolean constraints
- z3: SMT — integers/reals/bitvectors, proofs, program verification
- cpmpy: finite-domain constraint programming — scheduling, assignment, puzzles
- clingo: Answer Set Programming — logic rules, defaults, reachability

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
(3) finish with `submit_code` (the final, verified, self-contained program).

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

Tips: encode exactly the stated problem (every number matters); UNSAT is a
legitimate answer, re-check the encoding before reporting it; keep the final
program self-contained (all imports, no session state) and print only the
solution JSON. Long solver runs: python_exec defaults to a 30 s limit —
raise its `timeout` (up to 300) and give the solver an explicit time limit
rather than concluding the model is wrong.
"""

# Engine connection (one per server process; kernels persist across calls).
_engine = None

# The solving kernel: created by the first select_backend, recycled by later
# ones. Bare python_exec/python_interrupt calls are routed here; a manual
# python_reset that creates or resets a kernel retargets the routing.
# FastMCP dispatches tool calls concurrently, so every read-await-write of
# _kernel_id happens under this lock.
_kernel_id: str | None = None
_kernel_lock = asyncio.Lock()

# Server-side execution cap of the engine's python_exec (seconds); clamp
# here too so an oversized host value cannot stretch the client-side wait.
_MAX_EXEC_TIMEOUT = 300

# Bounded store of successful submissions (Tier 2 resources).
_submissions: dict[str, str] = {}
_MAX_SUBMISSIONS = 50

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
    """Dev-mode source checkout for helpers/templates (env or auto-detect)."""
    env = os.environ.get("MCP_SOLVER_DEV")
    if env:
        if env.lower() in ("1", "true", "auto"):
            return _local_checkout()
        return env
    return _local_checkout()


@asynccontextmanager
async def _lifespan(server: FastMCP) -> AsyncIterator[None]:
    """Hold one engine (ipython_mcp) connection for the server's lifetime."""
    global _engine, _kernel_id
    from mcp_minion import MCPManager

    manager = MCPManager(
        {
            "ipython": {
                "command": sys.executable,
                "args": ["-m", "agentic_python_coder.mcp_server"],
            }
        }
    )
    await manager.__aenter__()
    if not manager.has_tool("submit_code"):
        await manager.__aexit__(None, None, None)
        raise RuntimeError(
            "the ipython_mcp engine is unavailable or lacks submit_code;"
            " install agentic-python-coder >=3.4.1 in this environment"
        )
    _engine = manager
    try:
        yield
    finally:
        _episode_close("shutdown")
        _engine = None
        _kernel_id = None
        await manager.__aexit__(None, None, None)


mcp = FastMCP(
    "mcp-solver",
    instructions=(
        "Constraint-solving toolkit: call select_backend(solver) first, then"
        " write and verify a solver program via python_exec, and finish with"
        " submit_code. Backend choice: see resource mcp-solver://guide."
    ),
    lifespan=_lifespan,
)


async def _engine_call(name: str, arguments: dict) -> str:
    """Proxy one tool call to the engine; unwrap the client's JSON envelope.

    The envelope's ``error`` key covers client-level failures (engine down,
    timeout). Engine-level outcomes — including failed executions — arrive
    inside ``result`` as the engine tool's own JSON text and pass through to
    the host untouched.
    """
    if _engine is None:
        raise ToolError("engine not connected (server starting or shut down)")
    raw = await _engine.call_tool(name, arguments)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ToolError(f"engine returned a malformed envelope: {e}") from None
    if not isinstance(payload, dict):
        raise ToolError("engine returned a malformed envelope (not a JSON object)")
    if "error" in payload:
        raise ToolError(str(payload["error"]))
    return str(payload.get("result", ""))


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
        raise ValueError(f"unknown solver {solver!r}; expected one of {SOLVERS}")
    return get_template(solver, root=_dev_path())


@mcp.resource("mcp-solver://submissions/{submission_id}", mime_type="text/x-python")
def submission_resource(submission_id: str) -> str:
    """A previously submitted (compile-checked) solver program."""
    code = _submissions.get(submission_id)
    if code is None:
        raise ValueError(f"unknown submission {submission_id!r}")
    return code


@mcp.tool(description=SELECT_BACKEND_DESCRIPTION, structured_output=False)
async def select_backend(solver: str) -> str:
    global _kernel_id
    if solver not in SOLVERS:
        raise ToolError(f"unknown solver {solver!r}; expected one of {SOLVERS}")

    dev = _dev_path()
    if dev is None and _is_unpublishable(mcp_solver.__version__):
        raise ToolError(
            f"mcp-solver {mcp_solver.__version__} is not a published release"
            " and no source checkout was found; set MCP_SOLVER_DEV to a"
            " checkout path"
        )
    packages = build_with_packages(solver, dev)

    # Recycle the existing solving kernel; fresh kernel on first call or
    # when the stored one is gone (engine reports success=false for it).
    recycled = False
    parsed: dict = {}
    async with _kernel_lock:
        if _kernel_id is not None:
            parsed = _parse_engine_json(
                await _engine_call(
                    "python_reset", {"packages": packages, "kernel_id": _kernel_id}
                )
            )
            recycled = bool(parsed.get("success"))
        if not recycled:
            parsed = _parse_engine_json(
                await _engine_call("python_reset", {"packages": packages})
            )
            if not parsed.get("success"):
                raise ToolError(
                    f"kernel setup for {solver!r} failed:"
                    f" {parsed.get('error') or 'unknown engine error'}"
                )
        _kernel_id = parsed.get("kernel_id") or _kernel_id
        _episode_open(solver)
        _episode_sync()

    state_note = (
        "The solving kernel was recycled: previous session state is cleared."
        if recycled
        else "A fresh solving kernel is ready."
    )
    kernel_note = (
        f"{state_note} Packages: {', '.join(packages)}. python_exec runs in"
        " this kernel automatically."
        " Finish with submit_code(final verified self-contained program)."
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

    Runs in the kernel set up by select_backend unless kernel_id overrides
    it. State persists across calls (variables, imports). Raise timeout
    (seconds, max 300) for long solver runs.
    """
    _episode_count("python_exec")
    if not math.isfinite(timeout):
        timeout = 30.0
    args: dict = {"code": code, "timeout": max(1.0, min(timeout, _MAX_EXEC_TIMEOUT))}
    target = kernel_id or _kernel_id
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
    """Create a new kernel (no kernel_id) or reset one (with kernel_id).

    Prefer select_backend, which installs the right solver packages for you.
    The kernel created or reset here becomes the target of bare python_exec
    calls.
    """
    global _kernel_id
    _episode_count("python_reset")
    args = {"packages": packages or []}
    if kernel_id:
        args["kernel_id"] = kernel_id
    async with _kernel_lock:
        text = await _engine_call("python_reset", args)
        parsed = _parse_engine_json(text)
        if parsed.get("success") and parsed.get("kernel_id"):
            _kernel_id = str(parsed["kernel_id"])
    return text


@mcp.tool(structured_output=False)
async def python_status(kernel_id: str | None = None) -> str:
    """Check kernel state: active kernels, defined variables, packages."""
    _episode_count("python_status")
    args: dict = {}
    if kernel_id:
        args["kernel_id"] = kernel_id
    return await _engine_call("python_status", args)


@mcp.tool(structured_output=False)
async def python_interrupt(kernel_id: str | None = None) -> str:
    """Interrupt running code in the solving kernel (state is preserved)."""
    _episode_count("python_interrupt")
    args: dict = {}
    target = kernel_id or _kernel_id
    if target:
        args["kernel_id"] = target
    return await _engine_call("python_interrupt", args)


@mcp.tool()
async def submit_code(code: str) -> CallToolResult:
    """Submit the final, verified, self-contained solver program.

    Call ONCE at the end, only after executing the program via python_exec
    and passing your verification. The code is syntax-checked; on success it
    is stored and linked as a resource (the server keeps the last 50).
    """
    # Count the attempt BEFORE _episode_count syncs the .open snapshot, so a
    # failed submission (which never syncs again) is not lost on a host kill.
    if _episode is not None:
        _episode["submit_attempts"] += 1
    _episode_count("submit_code")
    result_text = await _engine_call("submit_code", {"code": code})
    verdict = _parse_engine_json(result_text)

    content: list = [TextContent(type="text", text=result_text)]
    if verdict.get("ok"):
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
                uri=AnyUrl(f"mcp-solver://submissions/{submission_id}"),
                name=f"submission_{submission_id[:8]}.py",
                description="The accepted solver program (compile-checked).",
                mimeType="text/x-python",
                size=len(code.encode()),
            )
        )
    structured = dict(verdict) if verdict else {}
    stats = _episode_snapshot()
    if stats:
        structured["stats"] = stats
    return CallToolResult(
        content=content,
        structuredContent=structured or None,
    )


def main() -> None:
    """Entry point for mcp-solver-serve (stdio transport)."""
    mcp.run()


if __name__ == "__main__":
    main()
