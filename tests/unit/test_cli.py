"""Unit tests for the v4 mcp-solver CLI (no engine, no network)."""

import json
import sys
from pathlib import Path

import pytest

import mcp_solver
from mcp_solver.agent import cli, engine

# --- argument parsing: solver validation ---------------------------------


def test_valid_solver_parses():
    args = cli.build_parser().parse_args(["z3", "solve", "this"])
    assert args.solver == "z3"
    assert args.task == ["solve", "this"]


@pytest.mark.parametrize("solver", ["pysat", "maxsat", "z3", "cpmpy", "clingo", "didp"])
def test_all_solvers_accepted(solver):
    args = cli.build_parser().parse_args([solver, "hi"])
    assert args.solver == solver


def test_unknown_solver_rejected():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["gurobi", "hi"])


def test_solver_packages_covers_all_solvers():
    # SOLVER_PACKAGES and SOLVERS are maintained by hand in two files;
    # a backend present in one but not the other fails at runtime.
    from mcp_solver.templates import SOLVERS

    assert set(cli.SOLVER_PACKAGES) == set(SOLVERS)


def test_defaults():
    args = cli.build_parser().parse_args(["z3", "hi"])
    assert args.model == "gpt56terra"
    assert args.workdir is None
    assert args.step_limit == 30
    assert args.quiet is False
    assert args.dev is None
    assert args.no_dev is False
    assert args.token_limit == cli.DEFAULT_TOKEN_LIMIT == 2_000_000


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
    assert "cannot read problem file" in capsys.readouterr().err


def test_problem_path_is_directory_clean_error(tmp_path, capsys):
    # A directory (IsADirectoryError, an OSError) must also exit cleanly.
    parser = cli.build_parser()
    args = parser.parse_args(["z3", "--problem", str(tmp_path)])
    with pytest.raises(SystemExit):
        cli.resolve_task(args, parser)
    assert "cannot read problem file" in capsys.readouterr().err


# --- with_packages construction ------------------------------------------


def test_with_packages_pysat_pins_helpers(monkeypatch):
    # Outside a source checkout the helpers fall back to the PyPI pin.
    monkeypatch.setattr(engine, "local_checkout", lambda: None)
    pkgs = cli.build_with_packages("pysat")
    assert pkgs == [
        "python-sat[pb]",
        f"agentic-mcp-solver=={mcp_solver.__version__}",
    ]


def test_source_checkout_auto_detected():
    # The test suite runs from an editable install, so detection must fire
    # and point at the repo root.
    checkout = engine.local_checkout()
    assert checkout is not None
    assert cli.helpers_package(checkout) == checkout


@pytest.mark.parametrize(
    ("solver", "lib"),
    [
        ("pysat", "python-sat[pb]"),
        ("maxsat", "python-sat[pb]"),
        ("z3", "z3-solver"),
        ("cpmpy", "cpmpy"),
        ("clingo", "clingo"),
        ("didp", "didppy>=0.11.1,<0.12"),
    ],
)
def test_with_packages_solver_libraries(solver, lib, monkeypatch):
    monkeypatch.setattr(engine, "local_checkout", lambda: None)
    pkgs = cli.build_with_packages(solver)
    assert pkgs[0] == lib
    # helpers package is always injected, for every backend.
    assert pkgs[-1] == f"agentic-mcp-solver=={mcp_solver.__version__}"


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
    monkeypatch.setattr(engine, "local_checkout", lambda: "/repo")
    assert _resolve(["z3", "--no-dev", "hi"], monkeypatch) is None


def test_dev_path_used_verbatim(monkeypatch):
    assert (
        _resolve(["z3", "--dev", "/my/checkout", "hi"], monkeypatch) == "/my/checkout"
    )


def test_dev_bare_auto_detects(monkeypatch):
    # Bare --dev (no path) placed last so it does not swallow the task token.
    monkeypatch.setattr(engine, "local_checkout", lambda: "/repo")
    assert _resolve(["z3", "hi", "--dev"], monkeypatch) == "/repo"


def test_dev_bare_errors_without_checkout(monkeypatch, capsys):
    monkeypatch.setattr(engine, "local_checkout", lambda: None)
    with pytest.raises(SystemExit):
        _resolve(["z3", "hi", "--dev"], monkeypatch)
    assert "no source checkout" in capsys.readouterr().err


def test_env_truthy_auto_detects(monkeypatch):
    monkeypatch.setattr(engine, "local_checkout", lambda: "/repo")
    assert _resolve(["z3", "hi"], monkeypatch, env="1") == "/repo"


def test_env_path_used(monkeypatch):
    monkeypatch.setattr(engine, "local_checkout", lambda: "/repo")
    assert _resolve(["z3", "hi"], monkeypatch, env="/env/checkout") == "/env/checkout"


def test_env_truthy_errors_without_checkout(monkeypatch):
    monkeypatch.setattr(engine, "local_checkout", lambda: None)
    with pytest.raises(SystemExit):
        _resolve(["z3", "hi"], monkeypatch, env="auto")


def test_auto_default_in_checkout(monkeypatch):
    # Nothing set: fall back to the detected checkout.
    monkeypatch.setattr(engine, "local_checkout", lambda: "/repo")
    assert _resolve(["z3", "hi"], monkeypatch) == "/repo"


def test_auto_default_none_outside_checkout(monkeypatch):
    monkeypatch.setattr(engine, "local_checkout", lambda: None)
    assert _resolve(["z3", "hi"], monkeypatch) is None


def test_arg_beats_env(monkeypatch):
    monkeypatch.setattr(engine, "local_checkout", lambda: "/repo")
    assert (
        _resolve(["z3", "--dev", "/arg", "hi"], monkeypatch, env="/env/checkout")
        == "/arg"
    )


def test_no_dev_beats_env(monkeypatch):
    monkeypatch.setattr(engine, "local_checkout", lambda: "/repo")
    assert _resolve(["z3", "--no-dev", "hi"], monkeypatch, env="/env/checkout") is None


def test_dev_and_no_dev_mutually_exclusive():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["z3", "--dev", "--no-dev", "hi"])


# --- model resolution ------------------------------------------------------


def test_raw_openrouter_id_passes_through():
    model = cli.resolve_model("openai/gpt-5.6-terra")
    assert model == cli.ModelSettings(model="openai/gpt-5.6-terra")


def test_alias_resolved_from_engine():
    # The default alias must resolve via the engine's bundled models/.
    model = cli.resolve_model("gpt56terra")
    assert model.model == "openai/gpt-5.6-terra"
    # gpt56terra is a no-sampling-params model: only max_tokens forwarded.
    assert set(model.api_params) <= {"max_tokens"}


def test_bundled_alias_with_request_timeout():
    # gemini31 carries request_timeout in the engine's bundled config.
    model = cli.resolve_model("gemini31")
    assert model.model.startswith("google/")
    assert model.request_timeout is not None
    assert "request_timeout" not in model.api_params


def test_unknown_alias_raises():
    with pytest.raises(ValueError, match="unknown model alias"):
        cli.resolve_model("no-such-alias")


def test_model_settings_no_sampling():
    model = cli.model_settings(
        {
            "path": "x/y",
            "no_sampling_params": True,
            "max_tokens": 128,
            "temperature": 0,
            "top_k": 5,
            "provider": {"order": ["a"]},
        }
    )
    assert model.api_params == {"max_tokens": 128}
    # top_k is a sampling parameter; provider applies to every model.
    assert model.extra_body == {"provider": {"order": ["a"]}}


def test_model_settings_standard():
    model = cli.model_settings(
        {"path": "x/y", "temperature": 0, "max_tokens": 64, "top_k": 5}
    )
    assert model.api_params == {"temperature": 0, "max_tokens": 64}
    assert model.extra_body == {"top_k": 5}
    assert model.request_timeout is None


def _model_file(tmp_path: Path) -> Path:
    path = tmp_path / "mymodel.json"
    path.write_text(
        json.dumps(
            {
                "path": "vendor/model-x",
                "temperature": 0.2,
                "max_tokens": 4096,
                "reasoning": {"effort": "high"},
                "request_timeout": 90,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_model_json_file_resolved(tmp_path):
    model = cli.resolve_model(str(_model_file(tmp_path)))
    assert model.model == "vendor/model-x"
    assert model.api_params == {"temperature": 0.2, "max_tokens": 4096}
    assert model.extra_body == {"reasoning": {"effort": "high"}}
    assert model.request_timeout == 90


def test_missing_model_json_file_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown model alias"):
        cli.resolve_model(str(tmp_path / "nope.json"))


# --- API key lookup --------------------------------------------------------


def test_find_api_key_reads_engine_dotenv(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert cli.find_api_key() is None
    (tmp_path / ".mcp-minion").write_text("OPENROUTER_API_KEY=sk-minion\n")
    assert cli.find_api_key() == "sk-minion"
    # The engine's file comes before ~/.mcp-minion, the environment first.
    engine_env = tmp_path / ".config" / "coder" / ".env"
    engine_env.parent.mkdir(parents=True)
    engine_env.write_text("OPENROUTER_API_KEY=sk-engine\n")
    assert cli.find_api_key() == "sk-engine"
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env")
    assert cli.find_api_key() == "sk-env"


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
    assert stats["token_limit_reached"] is False
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
                    "result": '{"ok": true}',
                    "is_error": False,
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
    monkeypatch.setattr(cli, "find_api_key", lambda *_: "sk-test")

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
    monkeypatch.setattr(cli, "find_api_key", lambda *_: "sk-test")

    rc = cli.main(["z3", "--problem", str(problem), "--workdir", str(tmp_path), "-q"])
    assert rc == 3
    assert "submit_code" in capsys.readouterr().err


def test_main_agent_error_after_submission_runs_it_and_returns_4(
    monkeypatch, tmp_path, capsys
):
    # The agent stopped early (LLM API failure) after an accepted submission:
    # the submission is persisted and re-executed, and the exit code flags it.
    problem = tmp_path / "queens.md"
    problem.write_text("Place 8 queens.", encoding="utf-8")
    stats_file = tmp_path / "stats.json"

    async def fake_solve(task, api_key, config, logger, quiet):
        result = _fake_solve_result("print('EARLIER')")
        # Set as an attribute so the test also runs against a minion
        # whose AgentResult has no ``error`` field yet.
        result.error = "APIError: 502 Bad Gateway"
        return result

    ran = {}

    def fake_run_program(program, with_packages):
        ran["program"] = program
        return 0

    monkeypatch.setattr(cli, "solve", fake_solve)
    monkeypatch.setattr(cli, "run_program", fake_run_program)
    monkeypatch.setattr(cli, "find_api_key", lambda *_: "sk-test")

    rc = cli.main(
        [
            "z3",
            "--problem",
            str(problem),
            "--workdir",
            str(tmp_path),
            "--stats-json",
            str(stats_file),
            "-q",
        ]
    )
    assert rc == cli.EXIT_AGENT_ERROR_WITH_SUBMISSION == 4
    artifact = tmp_path / "queens_code.py"
    assert artifact.read_text(encoding="utf-8") == "print('EARLIER')"
    assert ran["program"] == artifact
    assert stats_file.is_file()
    err = capsys.readouterr().err
    assert "502 Bad Gateway" in err
    assert "last accepted submission" in err


def test_main_agent_error_after_submission_keeps_program_failure_code(
    monkeypatch, tmp_path
):
    # A failing re-execution keeps the program's own exit code.
    async def fake_solve(task, api_key, config, logger, quiet):
        result = _fake_solve_result()
        result.error = "boom"
        return result

    monkeypatch.setattr(cli, "solve", fake_solve)
    monkeypatch.setattr(cli, "run_program", lambda program, with_packages: 7)
    monkeypatch.setattr(cli, "find_api_key", lambda *_: "sk-test")

    rc = cli.main(["z3", "hi", "--workdir", str(tmp_path), "-q"])
    assert rc == 7


def test_main_agent_error_without_submission_returns_1(monkeypatch, tmp_path, capsys):
    async def fake_solve(task, api_key, config, logger, quiet):
        result = _fake_result([{"step": 1, "content": None, "tool_calls": None}])
        result.error = "APIError: 401 Unauthorized"
        return result

    def fail_run_program(program, with_packages):
        raise AssertionError("nothing to run")

    monkeypatch.setattr(cli, "solve", fake_solve)
    monkeypatch.setattr(cli, "run_program", fail_run_program)
    monkeypatch.setattr(cli, "find_api_key", lambda *_: "sk-test")

    rc = cli.main(["z3", "hi", "--workdir", str(tmp_path), "-q"])
    assert rc == 1
    assert "401 Unauthorized" in capsys.readouterr().err
    assert not (tmp_path / "task_code.py").exists()


def test_main_unexpected_exception_one_line_returns_1(monkeypatch, tmp_path, capsys):
    async def fake_solve(task, api_key, config, logger, quiet):
        raise ExceptionGroup(
            "unhandled errors in a TaskGroup",
            [ConnectionResetError("connection\nreset by peer")],
        )

    monkeypatch.setattr(cli, "solve", fake_solve)
    monkeypatch.setattr(cli, "find_api_key", lambda *_: "sk-test")

    rc = cli.main(["z3", "hi", "--workdir", str(tmp_path), "-q"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err
    failure = [line for line in err.splitlines() if "solve failed" in line]
    assert failure == [
        "mcp-solver: the solve failed: ConnectionResetError: connection reset by peer"
    ]


def test_main_no_api_key_returns_1(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "find_api_key", lambda *_: None)
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
    monkeypatch.setattr(engine, "local_checkout", lambda: "/repo")
    monkeypatch.setitem(sys.modules, "agentic_python_coder", None)
    rc = cli.main(["z3", "hello"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "uv pip install 'agentic-mcp-solver[agent]'" in err


# --- fast-fail: unpublishable version with dev mode off ------------------


def test_fast_fail_unpublishable_no_dev(monkeypatch, capsys):
    # No checkout, no env, and a pre-release version → fail before the engine.
    monkeypatch.delenv("MCP_SOLVER_DEV", raising=False)
    monkeypatch.setattr(mcp_solver, "__version__", "4.0.0a0")
    monkeypatch.setattr(engine, "local_checkout", lambda: None)
    rc = cli.main(["z3", "hello"])
    assert rc == 2
    assert "--dev" in capsys.readouterr().err


def test_no_fast_fail_for_published_version(monkeypatch, capsys):
    # A plain numeric version is publishable, so the pin is fine and the run
    # proceeds (to the engine-import branch, which is absent here → rc 1).
    monkeypatch.delenv("MCP_SOLVER_DEV", raising=False)
    monkeypatch.setattr(mcp_solver, "__version__", "4.0.0")
    monkeypatch.setattr(engine, "local_checkout", lambda: None)
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


# --- AgentConfig wiring: model settings and token limit --------------------


def test_main_wires_model_settings_and_token_limit(monkeypatch, tmp_path):
    seen = {}

    async def fake_solve(task, api_key, config, logger, quiet):
        seen["config"] = config
        return _fake_solve_result()

    monkeypatch.setattr(cli, "solve", fake_solve)
    monkeypatch.setattr(cli, "run_program", lambda program, with_packages: 0)
    monkeypatch.setattr(cli, "find_api_key", lambda *_: "sk-test")

    model_file = _model_file(tmp_path)
    argv = ["z3", "hi", "--workdir", str(tmp_path), "-q", "--model", str(model_file)]
    assert cli.main([*argv, "--token-limit", "5000"]) == 0
    config = seen["config"]
    assert config.model == "vendor/model-x"
    assert config.api_params == {"temperature": 0.2, "max_tokens": 4096}
    assert config.extra_body == {"reasoning": {"effort": "high"}}
    assert config.request_timeout == 90
    assert config.max_total_tokens == 5000

    # Default limit, and 0 = no limit.
    assert cli.main(argv) == 0
    assert seen["config"].max_total_tokens == cli.DEFAULT_TOKEN_LIMIT
    assert cli.main([*argv, "--token-limit", "0"]) == 0
    assert seen["config"].max_total_tokens == 0


def test_negative_token_limit_rejected(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["z3", "hi", "--workdir", str(tmp_path), "--token-limit", "-1"])


def test_main_token_cap_saves_and_runs_last_submission(monkeypatch, tmp_path, capsys):
    # The token cap ends the run like the step limit: the last accepted
    # submission is saved and executed, and stats + stderr say why it ended.
    stats_file = tmp_path / "stats.json"

    async def fake_solve(task, api_key, config, logger, quiet):
        result = _fake_solve_result("print('CAPPED')")
        result.token_cap_reached = True
        return result

    ran = {}

    def fake_run_program(program, with_packages):
        ran["program"] = program
        return 0

    monkeypatch.setattr(cli, "solve", fake_solve)
    monkeypatch.setattr(cli, "run_program", fake_run_program)
    monkeypatch.setattr(cli, "find_api_key", lambda *_: "sk-test")

    rc = cli.main(
        [
            "z3",
            "hi",
            "--workdir",
            str(tmp_path),
            "--token-limit",
            "1000",
            "--stats-json",
            str(stats_file),
            "-q",
        ]
    )
    assert rc == 0
    assert (tmp_path / "task_code.py").read_text(encoding="utf-8") == "print('CAPPED')"
    assert ran["program"] == tmp_path / "task_code.py"
    stats = json.loads(stats_file.read_text())
    assert stats["token_limit_reached"] is True
    assert stats["token_limit"] == 1000
    assert "token limit (1,000 tokens)" in capsys.readouterr().err


def test_main_skips_rejected_submission(monkeypatch, tmp_path):
    # An error outcome of submit_code (e.g. the toolkit server's rejection)
    # never becomes the persisted program.
    async def fake_solve(task, api_key, config, logger, quiet):
        result = _fake_solve_result("print('GOOD')")
        result.steps.insert(
            1,
            {
                "step": 2,
                "content": None,
                "tool_calls": [
                    {
                        "name": "submit_code",
                        "arguments": {"code": "def f(:"},
                        "result": "Syntax error at line 1",
                        "is_error": True,
                    }
                ],
            },
        )
        return result

    monkeypatch.setattr(cli, "solve", fake_solve)
    monkeypatch.setattr(cli, "run_program", lambda program, with_packages: 0)
    monkeypatch.setattr(cli, "find_api_key", lambda *_: "sk-test")
    assert cli.main(["z3", "hi", "--workdir", str(tmp_path), "-q"]) == 0
    assert (tmp_path / "task_code.py").read_text(encoding="utf-8") == "print('GOOD')"


def test_main_kernel_setup_failure_returns_1(monkeypatch, tmp_path, capsys):
    # minion reports a failed eager python_reset as AgentResult.error.
    async def fake_solve(task, api_key, config, logger, quiet):
        return _fake_result(
            [], error="kernel setup failed: python_reset with packages [...]"
        )

    monkeypatch.setattr(cli, "solve", fake_solve)
    monkeypatch.setattr(cli, "find_api_key", lambda *_: "sk-test")
    assert cli.main(["z3", "hi", "--workdir", str(tmp_path), "-q"]) == 1
    assert "kernel setup failed" in capsys.readouterr().err


# --- re-execution environment ----------------------------------------------


def test_run_program_uses_kernel_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret")
    monkeypatch.setenv("UV_INDEX_URL", "https://example.invalid/simple")
    seen = {}

    class Done:
        returncode = 0

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["env"] = kwargs.get("env")
        return Done()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    program = tmp_path / "p.py"
    assert cli.run_program(program, ["z3-solver"]) == 0
    assert seen["cmd"][-2:] == ["python", str(program)]
    assert seen["env"] == engine.kernel_env()
    assert "OPENROUTER_API_KEY" not in seen["env"]
    assert seen["env"]["UV_INDEX_URL"] == "https://example.invalid/simple"


# --- custom endpoint: base_url and api_key_env -----------------------------


def test_model_settings_endpoint_defaults():
    model = cli.model_settings({"path": "x/y"})
    assert model.base_url is None and model.api_key_env is None
    assert model.key_var == "OPENROUTER_API_KEY"


def test_main_custom_endpoint_uses_its_url_and_key(monkeypatch, tmp_path):
    model_file = tmp_path / "local.json"
    model_file.write_text(
        json.dumps(
            {
                "path": "local/model",
                "base_url": "http://localhost:8000/v1",
                "api_key_env": "MY_ENDPOINT_KEY",
            }
        )
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)  # no dotenv files
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-openrouter")
    monkeypatch.setenv("MY_ENDPOINT_KEY", "sk-local")
    seen = {}

    async def fake_solve(task, api_key, config, logger, quiet):
        seen["api_key"] = api_key
        seen["config"] = config
        return _fake_solve_result()

    monkeypatch.setattr(cli, "solve", fake_solve)
    monkeypatch.setattr(cli, "run_program", lambda program, with_packages: 0)
    argv = ["z3", "hi", "--workdir", str(tmp_path), "-q", "--model"]
    assert cli.main([*argv, str(model_file)]) == 0
    assert seen["config"].base_url == "http://localhost:8000/v1"
    assert seen["api_key"] == "sk-local"

    # Without base_url/api_key_env: OpenRouter and its key, as before.
    assert cli.main([*argv, str(_model_file(tmp_path))]) == 0
    assert seen["config"].base_url == "https://openrouter.ai/api/v1"
    assert seen["api_key"] == "sk-openrouter"


def test_main_custom_endpoint_missing_key_names_variable(monkeypatch, tmp_path, capsys):
    model_file = tmp_path / "local.json"
    model_file.write_text(
        json.dumps({"path": "local/model", "api_key_env": "MY_ENDPOINT_KEY"})
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-openrouter")
    monkeypatch.delenv("MY_ENDPOINT_KEY", raising=False)
    rc = cli.main(["z3", "hi", "--workdir", str(tmp_path), "--model", str(model_file)])
    assert rc == 1
    assert "MY_ENDPOINT_KEY" in capsys.readouterr().err


# --- base install without the [agent] extra --------------------------------

_NO_AGENT_SCRIPT = """
import sys

BLOCKED = ("mcp", "mcp_minion", "agentic_python_coder")


class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None


sys.meta_path.insert(0, Block())
from mcp_solver.agent import cli

cli.build_parser()
try:
    cli.main(["--help"])
except SystemExit as exc:
    print("HELP-EXIT", exc.code)
print("SOLVE-RC", cli.main(["z3", "hello", "--dev"]))
"""


def test_base_install_without_agent_extra(tmp_path):
    # A fresh interpreter with mcp, mcp-minion and the engine unimportable
    # (as in a base install): --help works and a solve attempt prints the
    # install guidance instead of a traceback.
    import os
    import subprocess

    env = {k: v for k, v in os.environ.items() if k != "MCP_SOLVER_DEV"}
    proc = subprocess.run(
        [sys.executable, "-c", _NO_AGENT_SCRIPT],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
        env=env,
    )
    assert "Traceback" not in proc.stderr, proc.stderr
    assert "HELP-EXIT 0" in proc.stdout
    assert "--token-limit" in proc.stdout
    assert "SOLVE-RC 1" in proc.stdout
    assert "the agent layer is not installed" in proc.stderr
    assert "[agent]" in proc.stderr
