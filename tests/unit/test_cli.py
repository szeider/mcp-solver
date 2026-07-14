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
    assert args.step_limit == 30
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
    assert pkgs == ["python-sat", "pypblib", f"mcpsolver=={mcp_solver.__version__}"]


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
    assert pkgs[-1] == f"mcpsolver=={mcp_solver.__version__}"


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


# --- model resolution ------------------------------------------------------


def test_raw_openrouter_id_passes_through():
    model_id, params = cli.resolve_model("openai/gpt-5.6-terra")
    assert model_id == "openai/gpt-5.6-terra"
    assert params == {}


def test_alias_resolved_from_engine():
    # The default alias must resolve via the engine's bundled models/.
    model_id, params = cli.resolve_model("gpt56terra")
    assert model_id == "openai/gpt-5.6-terra"
    # gpt56terra is a no-sampling-params model: only max_tokens forwarded.
    assert set(params) <= {"max_tokens"}


def test_unknown_alias_raises():
    with pytest.raises(ValueError, match="unknown model alias"):
        cli.resolve_model("no-such-alias")


def test_model_params_no_sampling():
    params = cli._model_params(
        {"no_sampling_params": True, "max_tokens": 128, "temperature": 0}
    )
    assert params == {"max_tokens": 128}


def test_model_params_standard():
    params = cli._model_params({"temperature": 0, "max_tokens": 64, "top_k": 5})
    assert params == {"temperature": 0, "max_tokens": 64}


# --- stats construction ----------------------------------------------------


def _fake_result(steps, **kwargs):
    from mcp_minion import AgentResult

    return AgentResult(answer="done", steps=steps, **kwargs)


def test_build_stats_shape():
    steps = [
        {"step": 1, "tool_calls": [{"name": "python_exec"}, {"name": "python_exec"}]},
        {"step": 2, "tool_calls": [{"name": "submit_code"}]},
        {"step": 3, "tool_calls": None},
    ]
    result = _fake_result(steps, input_tokens=100, output_tokens=20, tool_calls_made=3)
    stats = cli.build_stats(result, elapsed=1.5, model_id="m")
    assert stats["token_consumption"]["total_tokens"] == 120
    assert stats["tool_usage"] == {"python_exec": 2, "submit_code": 1}
    assert stats["execution_time_seconds"] == 1.5
    assert stats["step_limit_reached"] is False
    assert stats["steps"] == 3


# --- end-to-end main() with a mocked solve --------------------------------


def _fake_solve_result(code="print('{}')"):
    steps = [
        {
            "step": 1,
            "content": None,
            "tool_calls": [
                {
                    "name": "submit_code",
                    "arguments": {"code": code},
                    "result": '{"result": "{\\"ok\\": true}"}',
                }
            ],
        },
        {"step": 2, "content": "done", "tool_calls": None},
    ]
    return _fake_result(steps, input_tokens=10, output_tokens=5, tool_calls_made=1)


def test_main_persists_submission_and_runs_it(monkeypatch, tmp_path, capsys):
    problem = tmp_path / "queens.md"
    problem.write_text("Place 8 queens.", encoding="utf-8")

    async def fake_solve(task, api_key, config, logger, quiet):
        return _fake_solve_result("print('SOLUTION')")

    ran = {}

    def fake_run_program(program, with_packages):
        ran["program"] = program
        ran["with_packages"] = with_packages
        return 0

    monkeypatch.setattr(cli, "solve", fake_solve)
    monkeypatch.setattr(cli, "run_program", fake_run_program)
    monkeypatch.setattr(cli, "find_api_key", lambda: "sk-test")

    rc = cli.main(["z3", "--problem", str(problem), "--workdir", str(tmp_path), "-q"])
    assert rc == 0
    artifact = tmp_path / "queens_code.py"
    assert artifact.read_text(encoding="utf-8") == "print('SOLUTION')"
    assert ran["program"] == artifact


def test_main_no_submission_returns_3(monkeypatch, tmp_path, capsys):
    problem = tmp_path / "p.md"
    problem.write_text("x", encoding="utf-8")

    async def fake_solve(task, api_key, config, logger, quiet):
        return _fake_result([{"step": 1, "content": "gave up", "tool_calls": None}])

    monkeypatch.setattr(cli, "solve", fake_solve)
    monkeypatch.setattr(cli, "find_api_key", lambda: "sk-test")

    rc = cli.main(["z3", "--problem", str(problem), "--workdir", str(tmp_path), "-q"])
    assert rc == 3
    assert "submit_code" in capsys.readouterr().err


def test_main_no_api_key_returns_1(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "find_api_key", lambda: None)
    rc = cli.main(["z3", "hello", "--workdir", str(tmp_path)])
    assert rc == 1
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err


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
