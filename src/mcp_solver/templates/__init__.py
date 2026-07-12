"""Solver project templates (markdown), shipped as package data."""

from importlib.resources import files

SOLVERS = ("pysat", "maxsat", "z3", "cpmpy", "clingo")


def get_template(solver: str) -> str:
    """Return the project template text for a solver."""
    if solver not in SOLVERS:
        raise ValueError(f"Unknown solver {solver!r}; expected one of {SOLVERS}")
    return (files(__package__) / f"{solver}.md").read_text(encoding="utf-8")
