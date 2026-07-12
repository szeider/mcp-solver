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
    assert args.dev is None
    assert args.no_dev is False


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


def test_missing_problem_file_clean_error(capsys):
    parser = cli.build_parser()
    args = parser.parse_args(["z3", "--problem", "does/not/exist.md"])
    with pytest.raises(SystemExit):
        cli.resolve_task(args, parser)
    assert "problem file not found" in capsys.readouterr().err


# --- with_packages construction ------------------------------------------


def test_with_packages_pysat_pins_helpers(monkeypatch):
    # Outside a source checkout the helpers fall back to the PyPI pin.
    monkeypatch.setattr(cli, "_local_checkout", lambda: None)
    pkgs = cli.build_with_packages("pysat")
    assert pkgs == ["python-sat", f"mcp-solver=={mcp_solver.__version__}"]


def test_source_checkout_auto_detected():
    # The test suite runs from an editable install, so detection must fire
    # and point at the repo root.
    checkout = cli._local_checkout()
    assert checkout is not None
    assert cli.helpers_package(checkout) == checkout


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
def test_with_packages_solver_libraries(solver, lib, monkeypatch):
    monkeypatch.setattr(cli, "_local_checkout", lambda: None)
    pkgs = cli.build_with_packages(solver)
    assert pkgs[0] == lib
    # helpers package is always injected, for every backend.
    assert pkgs[-1] == f"mcp-solver=={mcp_solver.__version__}"


def test_dev_path_used_as_helpers(monkeypatch):
    pkgs = cli.build_with_packages("z3", dev_path="/path/to/checkout")
    assert pkgs == ["z3-solver", "/path/to/checkout"]


# --- dev-path resolution -------------------------------------------------


def _resolve(argv, monkeypatch, env=None):
    """Parse *argv* and return the resolved dev path (clearing env first)."""
    monkeypatch.delenv("MCP_SOLVER_DEV", raising=False)
    if env is not None:
        monkeypatch.setenv("MCP_SOLVER_DEV", env)
    parser = cli.build_parser()
    args = parser.parse_args(argv)
    return cli.resolve_dev_path(args, parser)


def test_no_dev_forces_pin_even_in_checkout(monkeypatch):
    # A checkout is detected, but --no-dev overrides it → published behavior.
    monkeypatch.setattr(cli, "_local_checkout", lambda: "/repo")
    assert _resolve(["z3", "--no-dev", "hi"], monkeypatch) is None


def test_dev_path_used_verbatim(monkeypatch):
    assert (
        _resolve(["z3", "--dev", "/my/checkout", "hi"], monkeypatch) == "/my/checkout"
    )


def test_dev_bare_auto_detects(monkeypatch):
    # Bare --dev (no path) placed last so it does not swallow the task token.
    monkeypatch.setattr(cli, "_local_checkout", lambda: "/repo")
    assert _resolve(["z3", "hi", "--dev"], monkeypatch) == "/repo"


def test_dev_bare_errors_without_checkout(monkeypatch):
    monkeypatch.setattr(cli, "_local_checkout", lambda: None)
    with pytest.raises(SystemExit):
        _resolve(["z3", "hi", "--dev"], monkeypatch)


def test_env_truthy_auto_detects(monkeypatch):
    monkeypatch.setattr(cli, "_local_checkout", lambda: "/repo")
    assert _resolve(["z3", "hi"], monkeypatch, env="1") == "/repo"


def test_env_path_used(monkeypatch):
    monkeypatch.setattr(cli, "_local_checkout", lambda: "/repo")
    assert _resolve(["z3", "hi"], monkeypatch, env="/env/checkout") == "/env/checkout"


def test_env_truthy_errors_without_checkout(monkeypatch):
    monkeypatch.setattr(cli, "_local_checkout", lambda: None)
    with pytest.raises(SystemExit):
        _resolve(["z3", "hi"], monkeypatch, env="auto")


def test_auto_default_in_checkout(monkeypatch):
    # Nothing set: fall back to the detected checkout.
    monkeypatch.setattr(cli, "_local_checkout", lambda: "/repo")
    assert _resolve(["z3", "hi"], monkeypatch) == "/repo"


def test_auto_default_none_outside_checkout(monkeypatch):
    monkeypatch.setattr(cli, "_local_checkout", lambda: None)
    assert _resolve(["z3", "hi"], monkeypatch) is None


def test_arg_beats_env(monkeypatch):
    monkeypatch.setattr(cli, "_local_checkout", lambda: "/repo")
    assert (
        _resolve(["z3", "--dev", "/arg", "hi"], monkeypatch, env="/env/checkout")
        == "/arg"
    )


def test_no_dev_beats_env(monkeypatch):
    monkeypatch.setattr(cli, "_local_checkout", lambda: "/repo")
    assert _resolve(["z3", "--no-dev", "hi"], monkeypatch, env="/env/checkout") is None


def test_dev_and_no_dev_mutually_exclusive():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["z3", "--dev", "--no-dev", "hi"])


# --- missing-engine error path -------------------------------------------


def test_missing_engine_returns_1(monkeypatch, capsys):
    # Setting the module to None makes ``import agentic_python_coder`` raise
    # ImportError, exercising the polite-error branch without the engine.
    # A checkout is detected here, so dev mode is on and the fast-fail is
    # skipped; the run reaches the engine-import branch.
    monkeypatch.delenv("MCP_SOLVER_DEV", raising=False)
    monkeypatch.setattr(cli, "_local_checkout", lambda: "/repo")
    monkeypatch.setitem(sys.modules, "agentic_python_coder", None)
    rc = cli.main(["z3", "hello"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "uv pip install 'mcp-solver[agent]'" in err


# --- fast-fail: unpublishable version with dev mode off ------------------


def test_fast_fail_unpublishable_no_dev(monkeypatch, capsys):
    # No checkout, no env, and a pre-release version → fail before the engine.
    monkeypatch.delenv("MCP_SOLVER_DEV", raising=False)
    monkeypatch.setattr(mcp_solver, "__version__", "4.0.0a0")
    monkeypatch.setattr(cli, "_local_checkout", lambda: None)
    rc = cli.main(["z3", "hello"])
    assert rc == 2
    assert "--dev" in capsys.readouterr().err


def test_no_fast_fail_for_published_version(monkeypatch, capsys):
    # A plain numeric version is publishable, so the pin is fine and the run
    # proceeds (to the engine-import branch, which is absent here → rc 1).
    monkeypatch.delenv("MCP_SOLVER_DEV", raising=False)
    monkeypatch.setattr(mcp_solver, "__version__", "4.0.0")
    monkeypatch.setattr(cli, "_local_checkout", lambda: None)
    monkeypatch.setitem(sys.modules, "agentic_python_coder", None)
    rc = cli.main(["z3", "hello"])
    assert rc == 1


# --- template root override (dev mode) -----------------------------------


def test_get_template_root_override(tmp_path):
    from mcp_solver.templates import get_template

    tdir = tmp_path / "src" / "mcp_solver" / "templates"
    tdir.mkdir(parents=True)
    (tdir / "pysat.md").write_text("FAKE OVERRIDE", encoding="utf-8")
    # The override file wins when present under the root.
    assert get_template("pysat", root=str(tmp_path)) == "FAKE OVERRIDE"
    # Missing override falls back to the packaged resource.
    assert get_template("z3", root=str(tmp_path)) != "FAKE OVERRIDE"
    # No root → packaged resource.
    assert get_template("pysat") != "FAKE OVERRIDE"
