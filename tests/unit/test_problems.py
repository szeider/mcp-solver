"""Benchmark problems must not hand out their answer (no API, no network).

Every ```json block in a problem statement is a shape-only example: piped to
the problem's validator, it must be rejected. Otherwise a program that prints
the example as a constant would pass the benchmark.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

PROBLEMS_ROOT = Path(__file__).resolve().parents[2] / "tests" / "problems"
JSON_BLOCK = re.compile(r"```json[ \t]*\n(.*?)```", re.DOTALL)

PROBLEMS = sorted(
    md
    for md in PROBLEMS_ROOT.glob("*/*.md")
    if md.with_name(f"{md.stem}_ground_truth.py").is_file()
)


def _verdict(validator: Path, solution: str) -> dict:
    """Run *validator* like the harness does and return its verdict."""
    proc = subprocess.run(
        [sys.executable, str(validator)],
        input=solution,
        capture_output=True,
        text=True,
        timeout=120,
    )
    out = proc.stdout.strip()
    for candidate in (out, out.splitlines()[-1] if out else ""):
        try:
            verdict = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(verdict, dict):
            return verdict
    return {"valid": False, "message": f"no verdict: {proc.stderr[:200]}"}


def test_problems_discovered():
    assert len(PROBLEMS) >= 30


@pytest.mark.parametrize(
    "problem", PROBLEMS, ids=[f"{p.parent.name}/{p.stem}" for p in PROBLEMS]
)
def test_json_examples_fail_validation(problem):
    blocks = JSON_BLOCK.findall(problem.read_text(encoding="utf-8"))
    assert blocks, f"{problem.name} has no ```json example of the output format"
    validator = problem.with_name(f"{problem.stem}_ground_truth.py")
    for i, block in enumerate(blocks, 1):
        verdict = _verdict(validator, block.strip())
        assert verdict.get("valid") is False, (
            f"json block {i} of {problem.name} passes its validator:"
            f" {verdict.get('message', '')}"
        )
