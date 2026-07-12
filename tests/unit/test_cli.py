"""Unit tests for the v4 mcp-solver CLI (no engine, no network)."""

import sys

import pytest

import mcp_solver
from mcp_solver.agent import cli

# --- argument parsing: solver validation ---------------------------------


def test_valid_solver_parses():
    args = cli.build_parser().parse_args(["z3", "solve", "this"])
    assert args.solver == "z3"
    assert args.task == ["solve", "this"]


@pytest.mark.parametrize("solver", ["pysat", "maxsat", "z3", "cpmpy", "clingo"])
def test_all_solvers_accepted(solver):
    args = cli.build_parser().parse_args([solver, "hi"])
    assert args.solver == solver


def test_unknown_solver_rejected():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["gurobi", "hi"])


def test_defaults():
    args = cli.build_parser().parse_args(["z3", "hi"])
    assert args.model == "gpt56terra"
    assert args.workdir is None
    assert args.step_limit is None
    assert args.quiet is False
    assert args.local_package is None


# --- task / problem exclusivity ------------------------------------------


def test_task_text_joined():
    parser = cli.build_parser()
    args = parser.parse_args(["z3", "hello", "world"])
    task, basename = cli.resolve_task(args, parser)
    assert task == "hello world"
    assert basename == "task"


def test_problem_file_used(tmp_path):
    problem = tmp_path / "sudoku.md"
    problem.write_text("Solve the sudoku.", encoding="utf-8")
    parser = cli.build_parser()
    args = parser.parse_args(["z3", "--problem", str(problem)])
    task, basename = cli.resolve_task(args, parser)
    assert task == "Solve the sudoku."
    assert basename == "sudoku"


def test_task_and_problem_conflict(tmp_path):
    problem = tmp_path / "p.md"
    problem.write_text("x", encoding="utf-8")
    parser = cli.build_parser()
    args = parser.parse_args(["z3", "hello", "--problem", str(problem)])
    with pytest.raises(SystemExit):
        cli.resolve_task(args, parser)


def test_neither_task_nor_problem():
    parser = cli.build_parser()
    args = parser.parse_args(["z3"])
    with pytest.raises(SystemExit):
        cli.resolve_task(args, parser)


# --- with_packages construction ------------------------------------------


def test_with_packages_pysat_pins_helpers():
    pkgs = cli.build_with_packages("pysat")
    assert pkgs == ["python-sat", f"mcp-solver=={mcp_solver.__version__}"]


@pytest.mark.parametrize(
    ("solver", "lib"),
    [
        ("pysat", "python-sat"),
        ("maxsat", "python-sat"),
        ("z3", "z3-solver"),
        ("cpmpy", "cpmpy"),
        ("clingo", "clingo"),
    ],
)
def test_with_packages_solver_libraries(solver, lib):
    pkgs = cli.build_with_packages(solver)
    assert pkgs[0] == lib
    # helpers package is always injected, for every backend.
    assert pkgs[-1] == f"mcp-solver=={mcp_solver.__version__}"


def test_local_package_argument_overrides_pin():
    pkgs = cli.build_with_packages("z3", local_package="/path/to/checkout")
    assert pkgs == ["z3-solver", "/path/to/checkout"]


def test_local_package_env_overrides_pin(monkeypatch):
    monkeypatch.setenv("MCP_SOLVER_LOCAL_PACKAGE", "/env/checkout")
    pkgs = cli.build_with_packages("clingo")
    assert pkgs == ["clingo", "/env/checkout"]


def test_argument_beats_env(monkeypatch):
    monkeypatch.setenv("MCP_SOLVER_LOCAL_PACKAGE", "/env/checkout")
    assert cli.helpers_package("/arg/checkout") == "/arg/checkout"


# --- missing-engine error path -------------------------------------------


def test_missing_engine_returns_1(monkeypatch, capsys):
    # Setting the module to None makes ``import agentic_python_coder`` raise
    # ImportError, exercising the polite-error branch without the engine.
    monkeypatch.setitem(sys.modules, "agentic_python_coder", None)
    rc = cli.main(["z3", "hello"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "uv pip install 'mcp-solver[agent]'" in err
