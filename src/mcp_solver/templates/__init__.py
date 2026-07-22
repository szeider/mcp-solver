"""Solver project templates (markdown), shipped as package data."""

from importlib.resources import files
from pathlib import Path

SOLVERS = ("pysat", "maxsat", "z3", "cpmpy", "clingo", "didp")


def get_template(solver: str, root: str | None = None) -> str:
    """Return the project template text for a solver.

    When *root* is given (dev mode), read the template from that source
    checkout at ``<root>/src/mcp_solver/templates/<solver>.md`` if it exists;
    otherwise fall back to the packaged resource.
    """
    if solver not in SOLVERS:
        raise ValueError(f"Unknown solver {solver!r}; expected one of {SOLVERS}")
    if root is not None:
        override = Path(root) / "src" / "mcp_solver" / "templates" / f"{solver}.md"
        if override.is_file():
            return override.read_text(encoding="utf-8")
    return (files(__package__) / f"{solver}.md").read_text(encoding="utf-8")
