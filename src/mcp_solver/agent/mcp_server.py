"""mcp-solver-serve: the v4.1 MCP server (successive revelation).

Tier 0: one ``solve`` tool whose description (kept well under 1.2KB — some
hosts truncate at 2KB) is all a host needs to call it correctly.
Tier 1: the resource ``mcp-solver://guide`` — a short backend-selection guide.
Tier 2: per-solve ``resource_link`` content pointing at the submitted solver
program and the run log.

Stateless per the 2025-11-25 MCP spec: no sampling, no elicitation. Each solve
runs the CLI in its own subprocess with a fresh working directory, and per-step
progress is forwarded as MCP progress notifications.
"""

import asyncio
import re
import sys
import tempfile
import uuid
from pathlib import Path

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ResourceLink, TextContent
from pydantic import AnyUrl

from mcp_solver.templates import SOLVERS

# One solve is bounded by the agent step limit; this is only a safety backstop.
SOLVE_TIMEOUT_SECONDS = 900
STEP_LIMIT = 30

# Kept under 1.2KB; hosts reveal deeper detail via mcp-solver://guide (Tier 1).
SOLVE_DESCRIPTION = """\
Solve a constraint, optimization, or verification problem stated in natural
language. A solver-writing agent encodes the problem for a real solver, runs
and verifies it, and returns the solution as JSON plus a link to the generated
solver program.

Backends (pick one):
- pysat: Boolean satisfiability — feasibility, combinatorial search
- maxsat: weighted optimization over Boolean constraints
- z3: SMT — integers/reals/bitvectors, proofs, program verification
- cpmpy: finite-domain constraint programming — scheduling, assignment, puzzles
- clingo: Answer Set Programming — logic rules, defaults, reachability

Unsure which backend fits? Read the resource mcp-solver://guide first.

State the problem completely: all numbers, all constraints, and the exact
output JSON format you want back. A solve typically takes 30-120 seconds;
progress is reported. UNSAT / "no solution exists" / "property proved" are
valid answers and reported as such, not errors.
"""

GUIDE = """\
# Choosing an mcp-solver backend

- **pysat** — pure Boolean structure: placements, colorings, covers,
  pigeonhole-style feasibility. Fastest when everything is yes/no variables.
- **maxsat** — like pysat but with costs or preferences: maximize satisfied
  soft constraints under hard ones (task assignment, selection, scheduling
  with penalties). Finds the true optimum, not a heuristic.
- **z3** — arithmetic and verification: integer/real/bitvector constraints,
  cryptarithmetic, proving a property holds (UNSAT of the negation),
  finding counterexamples, induction obligations.
- **cpmpy** — finite-domain CP: rosters, timetables, routing-style puzzles,
  global constraints (all-different, cumulative), optimization over integers.
- **clingo** — rule-based knowledge: transitive closure/reachability,
  defaults with exceptions, choice under logical rules, #minimize/#maximize.

Tips for good results:
- Include every number and constraint explicitly; the agent solves exactly
  what is written.
- Specify the desired output JSON schema (field names, nesting) — the answer
  will follow it exactly.
- Impossible instances are legitimate: expect {"satisfiable": false} or an
  explicit UNSAT/proved report rather than an error.
"""

mcp = FastMCP(
    "mcp-solver",
    instructions=(
        "Constraint solving via one tool: solve(solver, problem). Read the"
        " resource mcp-solver://guide for backend selection."
    ),
)

# run_id -> working directory of a completed solve (serves Tier 2 resources).
_runs: dict[str, Path] = {}


@mcp.resource("mcp-solver://guide", mime_type="text/markdown")
def guide() -> str:
    """Backend-selection guide for the solve tool."""
    return GUIDE


@mcp.resource("mcp-solver://runs/{run_id}/{filename}")
def run_file(run_id: str, filename: str) -> str:
    """Serve an artifact (program, log, stats) from a completed solve."""
    workdir = _runs.get(run_id)
    if workdir is None:
        raise ValueError(f"unknown run {run_id!r}")
    path = workdir / filename
    # The artifact must be a direct child of the run directory (no traversal).
    if path.parent != workdir or not path.is_file():
        raise ValueError(f"unknown artifact {filename!r} for run {run_id}")
    return path.read_text(encoding="utf-8")


_STEP_LINE = re.compile(r"^mcp-solver: step (\d+): (.*)$")


async def _forward_progress(stream: asyncio.StreamReader, ctx: Context) -> str:
    """Relay CLI stderr: step lines become progress notifications.

    Returns the full stderr text (for error reporting).
    """
    collected: list[str] = []
    while True:
        raw = await stream.readline()
        if not raw:
            return "".join(collected)
        line = raw.decode(errors="replace")
        collected.append(line)
        match = _STEP_LINE.match(line.strip())
        if match:
            step, detail = int(match.group(1)), match.group(2)
            await ctx.report_progress(step, STEP_LIMIT, detail)


@mcp.tool(description=SOLVE_DESCRIPTION)
async def solve(solver: str, problem: str, ctx: Context) -> list:
    if solver not in SOLVERS:
        raise ToolError(f"unknown solver {solver!r}; expected one of {SOLVERS}")
    if not problem.strip():
        raise ToolError("problem must be a non-empty problem statement")

    run_id = uuid.uuid4().hex[:8]
    workdir = Path(tempfile.mkdtemp(prefix=f"mcp-solver-{run_id}-"))
    (workdir / "problem.md").write_text(problem, encoding="utf-8")

    cmd = [
        sys.executable,
        "-m",
        "mcp_solver.agent.cli",
        solver,
        "--problem",
        str(workdir / "problem.md"),
        "--workdir",
        str(workdir),
        "--step-limit",
        str(STEP_LIMIT),
        "--stats-json",
        str(workdir / "stats.json"),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        async with asyncio.timeout(SOLVE_TIMEOUT_SECONDS):
            stderr_task = asyncio.create_task(_forward_progress(proc.stderr, ctx))
            stdout_bytes = await proc.stdout.read()
            stderr_text = await stderr_task
            returncode = await proc.wait()
    except TimeoutError:
        proc.kill()
        raise ToolError(f"solve timed out after {SOLVE_TIMEOUT_SECONDS}s") from None

    solution = stdout_bytes.decode(errors="replace").strip()
    if returncode != 0:
        tail = "\n".join(stderr_text.strip().splitlines()[-5:])
        raise ToolError(f"solve failed (exit {returncode}):\n{tail}")

    _runs[run_id] = workdir
    content: list = [TextContent(type="text", text=solution)]
    program = workdir / "problem_code.py"
    if program.is_file():
        content.append(
            ResourceLink(
                type="resource_link",
                uri=AnyUrl(f"mcp-solver://runs/{run_id}/{program.name}"),
                name=program.name,
                description="The verified solver program that produced this solution.",
                mimeType="text/x-python",
            )
        )
    for log in sorted(workdir.glob("run_*.json")):
        content.append(
            ResourceLink(
                type="resource_link",
                uri=AnyUrl(f"mcp-solver://runs/{run_id}/{log.name}"),
                name=log.name,
                description="Complete agent run log (steps, tool calls, tokens).",
                mimeType="application/json",
            )
        )
    return content


def main() -> None:
    """Entry point for mcp-solver-serve (stdio transport)."""
    mcp.run()


if __name__ == "__main__":
    main()
