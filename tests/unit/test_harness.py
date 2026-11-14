"""Unit tests for the benchmark harness command construction (no solves)."""

import argparse
import sys
from pathlib import Path

from mcp_solver.agent import harness


def _args(**overrides):
    values = {
        "model": "gpt56terra",
        "dev": None,
        "step_limit": 30,
        "token_limit": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _value(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_build_command_makes_paths_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    cmd = harness.build_command(
        "z3",
        Path("problems/z3/puzzle.md"),
        _args(dev="checkout", model="models/m.json"),
        Path("out/z3/run1"),
    )
    assert cmd[:4] == [sys.executable, "-m", "mcp_solver.agent.cli", "z3"]
    assert _value(cmd, "--problem") == str(tmp_path / "problems/z3/puzzle.md")
    assert _value(cmd, "--workdir") == str(tmp_path / "out/z3/run1")
    assert _value(cmd, "--stats-json") == str(tmp_path / "out/z3/run1/stats.json")
    assert _value(cmd, "--dev") == str(tmp_path / "checkout")
    assert _value(cmd, "--model") == str(tmp_path / "models/m.json")
    assert _value(cmd, "--step-limit") == "30"
    assert "--token-limit" not in cmd


def test_build_command_alias_bare_dev_and_token_limit(tmp_path):
    cmd = harness.build_command(
        "pysat", tmp_path / "p.md", _args(dev="auto", token_limit=0), tmp_path
    )
    assert _value(cmd, "--model") == "gpt56terra"
    assert cmd[cmd.index("--dev") + 1] == "--step-limit"  # bare --dev
    assert _value(cmd, "--token-limit") == "0"


def test_run_one_uses_run_dir_as_cwd(tmp_path, monkeypatch):
    seen = {}

    class Proc:
        returncode = 3
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kwargs):
        seen.setdefault("calls", []).append((cmd, kwargs))
        return Proc()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    problem = tmp_path / "puzzle.md"
    problem.write_text("x")
    args = _args()
    args.timeout = 10
    row = harness.run_one("z3", Path("puzzle.md"), args, Path("out"), 0)
    (cmd, kwargs), *_ = seen["calls"]
    workdir = Path(row["workdir"])
    assert workdir.is_absolute() and workdir.is_dir()
    assert kwargs["cwd"] == workdir
    assert _value(cmd, "--workdir") == str(workdir)
    assert row["valid"] is None  # no validator next to the problem


def test_build_command_relative_dev_env_made_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MCP_SOLVER_DEV", "checkout")
    cmd = harness.build_command("z3", tmp_path / "p.md", _args(), tmp_path / "run")
    assert _value(cmd, "--dev") == str(tmp_path / "checkout")
    # An explicit --dev for the harness wins, as in the CLI.
    cmd = harness.build_command(
        "z3", tmp_path / "p.md", _args(dev="/arg"), tmp_path / "run"
    )
    assert _value(cmd, "--dev") == "/arg"


def test_build_command_auto_dev_env_left_to_child(tmp_path, monkeypatch):
    # MCP_SOLVER_DEV=1 keeps meaning auto-detect: the child inherits it.
    monkeypatch.setenv("MCP_SOLVER_DEV", "1")
    cmd = harness.build_command("z3", tmp_path / "p.md", _args(), tmp_path / "run")
    assert "--dev" not in cmd


def test_build_command_local_alias_override_made_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MCP_SOLVER_DEV", raising=False)
    (tmp_path / "custom.json").write_text('{"path": "vendor/x"}')
    cmd = harness.build_command(
        "z3", tmp_path / "p.md", _args(model="custom"), tmp_path / "run"
    )
    assert _value(cmd, "--model") == str(tmp_path / "custom.json")
    # Without a local file the alias passes through; full IDs are untouched.
    for model in ("gpt56terra", "openai/gpt-5.6-terra"):
        cmd = harness.build_command(
            "z3", tmp_path / "p.md", _args(model=model), tmp_path / "run"
        )
        assert _value(cmd, "--model") == model


def _fake_cli(monkeypatch, *, returncode=0, stdout="", stderr="", timeout=False):
    import subprocess

    calls = []

    class Proc:
        pass

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if timeout:
            raise subprocess.TimeoutExpired(cmd, 10, output=b"", stderr=stderr)
        proc = Proc()
        proc.returncode, proc.stdout, proc.stderr = returncode, stdout, stderr
        return proc

    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    return calls


def _run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MCP_SOLVER_DEV", raising=False)
    (tmp_path / "puzzle.md").write_text("x")
    args = _args()
    args.timeout = 10
    row = harness.run_one("z3", Path("puzzle.md"), args, Path("out"), 0)
    return row, Path(row["workdir"]) / harness.STDERR_FILE


def test_run_one_failed_run_keeps_stderr(tmp_path, monkeypatch):
    lines = [f"line {i}" for i in range(30)]
    _fake_cli(monkeypatch, returncode=1, stderr="\n".join(lines) + "\n")
    row, stderr_file = _run(tmp_path, monkeypatch)
    assert stderr_file.read_text() == "\n".join(lines) + "\n"
    assert row["stderr_tail"].splitlines() == lines[-harness.STDERR_TAIL_LINES :]


def test_run_one_timeout_keeps_captured_stderr(tmp_path, monkeypatch):
    # TimeoutExpired can carry bytes even with text=True.
    _fake_cli(monkeypatch, stderr=b"mcp-solver: step 1: python_exec\n", timeout=True)
    row, stderr_file = _run(tmp_path, monkeypatch)
    assert row["exit_code"] == -1
    assert stderr_file.read_text() == "mcp-solver: step 1: python_exec\n"
    assert row["stderr_tail"] == "mcp-solver: step 1: python_exec"


def test_run_one_invalid_result_gets_tail(tmp_path, monkeypatch):
    # Exit 0 but no solution output: invalid, so the tail is recorded.
    (tmp_path / "puzzle_ground_truth.py").write_text("print('{}')")
    _fake_cli(monkeypatch, returncode=0, stdout="", stderr="no output\n")
    row, _ = _run(tmp_path, monkeypatch)
    assert row["valid"] is False
    assert row["stderr_tail"] == "no output"


def test_run_one_success_writes_stderr_without_tail(tmp_path, monkeypatch):
    _fake_cli(monkeypatch, returncode=0, stdout="", stderr="all fine\n")
    row, stderr_file = _run(tmp_path, monkeypatch)
    assert stderr_file.read_text() == "all fine\n"
    assert row["valid"] is None  # no validator
    assert "stderr_tail" not in row
