"""Unit tests for the mcp-solver-runfolder tool (no API, no network)."""

import json

import pytest

from mcp_solver.agent import runfolder
from mcp_solver.templates import SOLVERS


def test_markers_cover_all_solvers():
    assert set(SOLVERS) <= set(runfolder.SOLVER_MARKERS)


def test_write_run_folder_contents(tmp_path):
    folder = runfolder.write_run_folder(
        tmp_path / "demo", "cpmpy", "Place 8 queens.", tmp_path / "repo"
    )
    config = json.loads((folder / "config.json").read_text())
    assert config["model"]["name"] == runfolder.DEFAULT_MODEL
    assert config["mcpServers"]["mcp-solver"]["args"][-1] == "mcp-solver-serve"
    task = (folder / "task.md").read_text()
    assert "cpmpy backend ONLY" in task
    assert task.endswith("Place 8 queens.")


def test_write_run_folder_preserves_other_files(tmp_path):
    dest = tmp_path / "demo"
    dest.mkdir()
    (dest / "run_1.json").write_text("{}")
    runfolder.write_run_folder(dest, "z3", "x", tmp_path / "repo")
    assert (dest / "run_1.json").read_text() == "{}"


def test_cli_list_and_generate(tmp_path, capsys, monkeypatch):
    # Fake checkout with one problem.
    repo = tmp_path / "repo"
    problems = repo / "tests" / "problems" / "z3"
    problems.mkdir(parents=True)
    (problems / "puzzle.md").write_text("Solve me.")
    (problems / "test.md").write_text("smoke")  # excluded from discovery
    monkeypatch.setattr(runfolder, "_local_checkout", lambda: str(repo))

    assert runfolder.main(["z3", "list"]) == 0
    assert capsys.readouterr().out.splitlines() == ["puzzle"]

    dest = tmp_path / "out"
    assert runfolder.main(["z3", "puzzle", str(dest)]) == 0
    assert "Solve me." in (dest / "task.md").read_text()


def test_cli_unknown_problem_errors(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "tests" / "problems" / "z3").mkdir(parents=True)
    monkeypatch.setattr(runfolder, "_local_checkout", lambda: str(repo))
    with pytest.raises(SystemExit):
        runfolder.main(["z3", "nope", str(tmp_path / "out")])
