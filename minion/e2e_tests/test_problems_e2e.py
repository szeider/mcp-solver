"""End-to-end benchmark problems through minion + mcp-solver-serve.

Every problem of every backend in tests/problems/<solver>/ is runnable as a
real minion episode: the agent selects the backend over MCP, models the
problem in the solving kernel, submits, and the final answer is validated
by the problem's ground-truth validator. The run folders are built by the
same helpers as the mcp-solver-runfolder tool, so an e2e test run and a
manually generated run folder are the same episode.

These tests make real API calls with a solver-grade model and take about a
minute per problem. Requires OPENROUTER_API_KEY (env, cwd .env, or
~/.mcp-minion). Select what to run, e.g.:

    uv run pytest minion/e2e_tests -k didp -v
    uv run pytest minion/e2e_tests -k "cpmpy and not order" -v
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv
from mcp_solver.agent.runfolder import (
    DEFAULT_MODEL,
    SOLVER_MARKERS,
    write_run_folder,
)
from mcp_solver.templates import SOLVERS

from mcp_minion.cli import load_run_folder, save_submission

from .test_e2e import run_agent_from_folder

load_dotenv()
# Same fallback the minion CLI uses; does not override an existing env var.
load_dotenv(Path.home() / ".mcp-minion")

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROBLEMS_ROOT = REPO_ROOT / "tests" / "problems"

PAIRS = [
    (solver, md.stem)
    for solver in SOLVERS
    for md in sorted((PROBLEMS_ROOT / solver).glob("*.md"))
    if md.name != "test.md"
]


def extract_json(text: str) -> dict | None:
    """First parseable JSON object embedded in *text*, or None."""
    decoder = json.JSONDecoder()
    for start, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    return None


def make_run_folder(tmp_path: Path, solver: str, problem: str) -> Path:
    problem_text = (PROBLEMS_ROOT / solver / f"{problem}.md").read_text(
        encoding="utf-8"
    )
    return write_run_folder(
        tmp_path / f"{solver}_{problem}", solver, problem_text, REPO_ROOT
    )


@pytest.mark.parametrize(
    ("solver", "problem"), PAIRS, ids=[f"{s}-{p}" for s, p in PAIRS]
)
def test_problem_end_to_end(solver: str, problem: str, tmp_path: Path) -> None:
    folder = make_run_folder(tmp_path, solver, problem)
    result, logger = run_agent_from_folder(folder)

    # The agent must actually have used tools (not answered from memory).
    assert result.tool_calls_made >= 2

    # The final answer must contain the solution JSON...
    solution = extract_json(result.answer)
    assert solution is not None, f"no JSON object in answer: {result.answer!r}"

    # ...which the problem's independent ground-truth validator accepts.
    validator = PROBLEMS_ROOT / solver / f"{problem}_ground_truth.py"
    proc = subprocess.run(
        [sys.executable, str(validator)],
        input=json.dumps(solution),
        capture_output=True,
        text=True,
        timeout=120,
    )
    verdict = json.loads(proc.stdout.strip().splitlines()[-1])
    assert verdict["valid"], f"validator rejected {solution}: {verdict['message']}"

    # The submitted program lands on disk and genuinely used the requested
    # backend's library (no silent backend substitution).
    submission_path = save_submission(result.steps, folder)
    assert submission_path is not None, "no accepted submission in the run"
    code = submission_path.read_text(encoding="utf-8").lower()
    markers = SOLVER_MARKERS[solver]
    assert any(m in code for m in markers), (
        f"submission does not look like {solver} code (markers {markers})"
    )

    log_data = json.loads(logger.log_path.read_text())
    assert log_data["status"] == "completed"


def test_every_problem_has_a_validator() -> None:
    """Each problem ships its ground-truth validator (no API needed)."""
    assert PAIRS, f"no problems discovered under {PROBLEMS_ROOT}"
    for solver, problem in PAIRS:
        assert (PROBLEMS_ROOT / solver / f"{problem}_ground_truth.py").is_file(), (
            f"missing validator for {solver}/{problem}"
        )


def test_every_solver_has_markers() -> None:
    """Marker table covers every backend (no API needed)."""
    assert set(SOLVERS) <= set(SOLVER_MARKERS)


def test_generated_run_folder_loads(tmp_path: Path) -> None:
    """Generated run folders parse cleanly (no API needed)."""
    solver, problem = PAIRS[0]
    folder = make_run_folder(tmp_path, solver, problem)
    config, mcp_servers, prompt = load_run_folder(folder)
    assert config.model == DEFAULT_MODEL
    assert "mcp-solver" in mcp_servers
    assert f"{solver} backend ONLY" in prompt
