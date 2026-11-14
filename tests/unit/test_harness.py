"""Unit tests for the benchmark harness command construction (no solves)."""

import argparse
import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

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
    monkeypatch.setattr(harness, "run_cli", fake_run)
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

    monkeypatch.setattr(harness, "run_cli", fake_run)
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


# --- timeouts kill the whole process tree ----------------------------------

# Starts a sleeping grandchild, records both pids, then sleeps itself.
_SPAWNER = """
import os, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
with open(sys.argv[1], "w") as f:
    f.write(f"{os.getpid()} {child.pid}\\n")
time.sleep(600)
"""


def _gone(pid: int, wait: float = 3.0) -> bool:
    """True once *pid* no longer exists (killed orphans may linger briefly
    as zombies until init reaps them)."""
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            pass
        time.sleep(0.05)
    return False


def _pids(*files: Path) -> list[int]:
    return [int(x) for f in files if f.exists() for x in f.read_text().split()]


def _reap(pids: list[int]) -> None:
    """Kill whatever a failed test left running."""
    for pid in pids:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)


posix_only = pytest.mark.skipif(os.name != "posix", reason="process groups, /bin/sh")


@posix_only
def test_run_cli_timeout_kills_descendants(tmp_path):
    pid_file = tmp_path / "pids.txt"
    # A shell between run_cli and the spawner, as uv sits between the CLI and
    # the program: the spawner is a grandchild, its sleeper a great-grandchild
    # (the trailing command stops sh from exec-ing the spawner).
    script = tmp_path / "spawner.py"
    script.write_text(_SPAWNER)
    cmd = ["/bin/sh", "-c", f'"{sys.executable}" "{script}" "{pid_file}"; true']
    try:
        start = time.monotonic()
        with pytest.raises(subprocess.TimeoutExpired) as info:
            harness.run_cli(cmd, timeout=2, cwd=tmp_path)
        assert info.value.timeout == 2
        assert time.monotonic() - start < 2 + harness.TERM_GRACE + harness.KILL_GRACE
        assert len(_pids(pid_file)) == 2
        for pid in _pids(pid_file):
            assert _gone(pid)
    finally:
        _reap(_pids(pid_file))


@posix_only
def test_run_one_timeout_row_with_real_process(tmp_path, monkeypatch):
    pid_file = tmp_path / "pids.txt"
    script = tmp_path / "spawner.py"
    script.write_text(_SPAWNER)
    monkeypatch.setattr(
        harness,
        "build_command",
        lambda *a: [sys.executable, str(script), str(pid_file)],
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / "puzzle.md").write_text("x")
    args = _args()
    args.timeout = 1
    try:
        row = harness.run_one("z3", Path("puzzle.md"), args, Path("out"), 0)
        assert row["exit_code"] == -1
        assert row["error"] == "timeout after 1s"
        assert row["solution"] == ""
        assert len(_pids(pid_file)) == 2
        for pid in _pids(pid_file):
            assert _gone(pid)
    finally:
        _reap(_pids(pid_file))


# The real CLI re-execution (cli.run_program) inside a harness solve.
_FAKE_CLI = """
import os, sys
from pathlib import Path
from mcp_solver.agent import cli
Path(sys.argv[2]).write_text(str(os.getpid()))
sys.exit(cli.run_program(Path(sys.argv[1]), []))
"""


@posix_only
def test_run_cli_timeout_during_reexecution_kills_detached_program(
    tmp_path, monkeypatch
):
    # harness.run_cli -> fake CLI calling the real cli.run_program -> fake
    # ``uv`` (sh) -> program in its own session -> sleeping child. The
    # harness's SIGTERM makes the CLI kill the program's session before the
    # CLI's own group is killed.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake_uv = bindir / "uv"
    fake_uv.write_text(f'#!/bin/sh\nfor last; do :; done\n"{sys.executable}" "$last"\n')
    fake_uv.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    cli_pid, rerun_pids = tmp_path / "cli_pid.txt", tmp_path / "pids.txt"
    program = tmp_path / "p_code.py"
    program.write_text(_SPAWNER.replace("sys.argv[1]", repr(str(rerun_pids))))
    fake_cli = tmp_path / "fake_cli.py"
    fake_cli.write_text(_FAKE_CLI)
    cmd = [sys.executable, str(fake_cli), str(program), str(cli_pid)]
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            harness.run_cli(cmd, timeout=4, cwd=tmp_path)
        assert len(_pids(cli_pid)) == 1
        assert len(_pids(rerun_pids)) == 2  # the program had started
        for pid in _pids(cli_pid, rerun_pids):
            assert _gone(pid)
    finally:
        _reap(_pids(cli_pid, rerun_pids))


def test_run_cli_after_stop_spawns_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "_stopping", False)  # restored afterwards
    harness._stop_all()
    marker = tmp_path / "started"
    with pytest.raises(harness.BenchmarkStopped):
        harness.run_cli(
            [sys.executable, "-c", f"open({str(marker)!r}, 'w')"],
            timeout=10,
            cwd=tmp_path,
        )
    time.sleep(0.5)
    assert not marker.exists()


@posix_only
def test_main_sigterm_takes_interrupt_path(tmp_path, monkeypatch):
    # SIGTERM during a benchmark raises KeyboardInterrupt in main(), which
    # stops the running solves. A guard handler keeps pytest alive if main()
    # does not install its own.
    monkeypatch.setattr(harness, "_stopping", False)
    (tmp_path / "problems" / "z3").mkdir(parents=True)
    (tmp_path / "problems" / "z3" / "a.md").write_text("x")

    def fake_run_one(solver, problem_md, *rest):
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(1)
        return {
            "solver": solver,
            "problem": problem_md.stem,
            "valid": None,
            "wall_seconds": 0,
        }

    monkeypatch.setattr(harness, "run_one", fake_run_one)
    guard = signal.signal(signal.SIGTERM, lambda *a: None)
    try:
        with pytest.raises(KeyboardInterrupt):
            harness.main(
                [
                    "z3",
                    "--problems",
                    str(tmp_path / "problems"),
                    "--out",
                    str(tmp_path / "out"),
                ]
            )
        assert harness._stopping
    finally:
        signal.signal(signal.SIGTERM, guard)
